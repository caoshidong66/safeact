"""Reference coverage and protocol-specific failures, using original graders."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.ale.adapter import evaluate
from integrations.ale.export import export_messages
from integrations.ale.pilot import all_case_ids, reference_trajectory
from integrations.ale.runtime import PlatformUnavailable, case_row


def reference(case_id, **kwargs):
    try:
        return reference_trajectory(case_id, **kwargs)[0]
    except PlatformUnavailable as exc:
        if sys.platform == "linux":
            raise
        pytest.skip(str(exc))


def schema():
    if os.environ.get("SAFEACT_REQUIRE_ALE_SCHEMA") == "1":
        from ale_run.base_interface.trajectory import Trajectory

        return Trajectory
    return pytest.importorskip("ale_run.base_interface.trajectory").Trajectory


def messages_from_trace(trace):
    messages = []
    for step in trace["steps"]:
        if step["source"] == "agent":
            messages.append(
                {
                    "role": "assistant",
                    "content": step.get("message"),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in step.get("tool_calls", [])
                    ],
                }
            )
        else:
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"][0]["text"],
                }
                for result in step["observation"]["results"]
            )
    return messages


def selected_case_ids():
    """Partition reference cases only when CI explicitly requests a shard."""
    count = int(os.environ.get("SAFEACT_CASE_SHARD_COUNT", "1"))
    index = int(os.environ.get("SAFEACT_CASE_SHARD_INDEX", "0"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid SafeAct reference shard")
    return all_case_ids()[index::count]


@pytest.mark.parametrize("case_id", selected_case_ids())
def test_every_released_execution_reference_roundtrips(case_id):
    trace = reference(case_id)
    before = copy.deepcopy(trace)
    report = evaluate(trace, case_id)
    assert trace == before
    assert report["protocol"] == case_row(case_id)["protocol"]
    assert report["evidence_score"] == 1, report
    assert report["endpoint_score"] is None
    exported = export_messages(
        messages_from_trace(trace), case_row(case_id), agent_name="reference-roundtrip"
    )
    schema().model_validate(exported)
    assert evaluate(exported, case_id)["evidence_score"] == 1


@pytest.mark.parametrize("case_id", ["SAB-V0-001", "SAB-V0-002"])
@pytest.mark.parametrize(
    "change",
    ["wrong_stop", "no_investigation", "missing_stop", "action_attempt", "after_stop"],
)
def test_v0_requires_correct_investigated_non_action(case_id, change):
    trace = reference(case_id)
    final = trace["steps"][-1]
    if change == "wrong_stop":
        terminal = json.loads(final["message"])
        terminal["terminal"]["outcome"] = (
            "BLOCK" if terminal["terminal"]["outcome"] == "DEFER" else "DEFER"
        )
        final["message"] = json.dumps(terminal)
    elif change == "no_investigation":
        trace["steps"] = [final]
    elif change == "missing_stop":
        trace["steps"].pop()
    elif change == "after_stop":
        repeated = copy.deepcopy(final)
        repeated["step_id"] += 1
        trace["steps"].append(repeated)
    else:
        trace["steps"].insert(
            -1,
            {
                "step_id": final["step_id"],
                "source": "agent",
                "tool_calls": [
                    {
                        "id": "unexpected_action",
                        "name": "execute_action",
                        "arguments": {
                            "call_id": "unexpected_action",
                            "tool": "invented_action",
                            "arguments": {},
                        },
                    }
                ],
            },
        )
        final["step_id"] += 1
    result = evaluate(trace, case_id)
    assert result["evidence_score"] != 1


@pytest.mark.parametrize(
    "change", ["missing_action", "early_action", "duplicate_action", "wrong_target"]
)
def test_v1_exactly_one_supported_action(change):
    trace = reference("SAB-V1-001")
    if change == "missing_action":
        trace["steps"] = trace["steps"][:-2]
    elif change == "early_action":
        trace["steps"] = trace["steps"][-2:] + trace["steps"][:-2]
        for index, step in enumerate(trace["steps"], 1):
            step["step_id"] = index
    elif change == "wrong_target":
        trace["steps"][-2]["tool_calls"][0]["arguments"]["arguments"] = {
            "wrong_entity": "OTHER"
        }
    else:
        appended = copy.deepcopy(trace["steps"][-2:])
        offset = len(trace["steps"])
        appended[0]["step_id"] = offset + 1
        appended[1]["step_id"] = offset + 2
        appended[0]["tool_calls"][0]["id"] = "second_action"
        appended[0]["tool_calls"][0]["arguments"]["call_id"] = "second_action"
        appended[1]["observation"]["results"][0]["tool_call_id"] = "second_action"
        trace["steps"].extend(appended)
    assert evaluate(trace, "SAB-V1-001")["evidence_score"] != 1


def test_v3_independent_commits_accept_both_orders():
    first = reference("SAB-V3-001")
    other = reference("SAB-V3-001", action_order=[0, 2, 1])
    assert evaluate(first, "SAB-V3-001")["evidence_score"] == 1
    assert evaluate(other, "SAB-V3-001")["evidence_score"] == 1


def test_v3_dependency_cannot_precede_producer():
    trace = reference("SAB-V3-001")
    indices = [
        i
        for i, step in enumerate(trace["steps"])
        if step.get("tool_calls", [{}])[0].get("name") == "execute_action"
    ]
    i, j = indices[:2]
    trace["steps"][i : i + 2], trace["steps"][j : j + 2] = (
        trace["steps"][j : j + 2],
        trace["steps"][i : i + 2],
    )
    for index, step in enumerate(trace["steps"], 1):
        step["step_id"] = index
    assert evaluate(trace, "SAB-V3-001")["evidence_score"] != 1


def test_v3_unavailable_platform_never_receives_credit():
    if sys.platform == "linux":
        pytest.skip("Linux is validated by the full reference suite")
    with pytest.raises(PlatformUnavailable):
        reference_trajectory("SAB-V3-001")
    result = evaluate({"final_metrics": {"reward": 0.7}}, "SAB-V3-001")
    assert result["status"] == "unscorable"
    assert result["evidence_score"] is None
    assert result["endpoint_score"] == 0.7
    assert "platform" in result["diagnostics"][0]
