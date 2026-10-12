"""Regression tests against original SafeAct cases and ALE's native schema."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.ale.adapter import TraceError, evaluate, fingerprint, strict_json
from integrations.ale.export import export_messages
from integrations.ale.pilot import V2_PILOT_CASES, reference_trajectory
from integrations.ale.runtime import case_row, runtime


@pytest.fixture
def trace():
    return reference_trajectory("SAB-V2-003")[0]


@pytest.mark.parametrize("case_id", V2_PILOT_CASES)
def test_original_references_pass_and_input_is_unchanged(case_id):
    original, _public = reference_trajectory(case_id)
    before = copy.deepcopy(original)
    result = evaluate(original, case_id)
    assert result["status"] == "scored"
    assert result["evidence_score"] == 1
    assert result["endpoint_score"] is None
    assert original == before


def test_reference_matches_actual_ale_schema(trace):
    # Available when ALE's checkout has been installed. Do not invent a parallel schema.
    schema = pytest.importorskip("ale_run.base_interface.trajectory")
    parsed = schema.Trajectory.model_validate(trace)
    result = evaluate(parsed.model_dump(mode="json"), "SAB-V2-003")
    assert result["evidence_score"] == 1


def test_incomplete_workflow_preserves_endpoint_score(trace):
    first_action = next(
        i
        for i, step in enumerate(trace["steps"])
        if step.get("tool_calls", [{}])[0].get("name") == "execute_action"
    )
    trace["steps"] = trace["steps"][:first_action]
    trace["final_metrics"]["reward"] = 0.75
    result = evaluate(trace, "SAB-V2-003")
    assert result["status"] == "scored"
    assert result["evidence_score"] == 0
    assert result["endpoint_score"] == 0.75
    assert result["native_result"]["summary"]["action_executed_count"] == 0


def test_attempt_before_any_evidence_is_rejected_by_original_evaluator(trace):
    row = case_row("SAB-V2-003")
    action = next(
        step
        for step in trace["steps"]
        if step.get("tool_calls", [{}])[0].get("name") == "execute_action"
    )
    action = copy.deepcopy(action)
    action["step_id"] = 1
    envelope = action["tool_calls"][0]["arguments"]
    with runtime(row, [1], seed=b"negative-test") as (gateway, _run, _live):
        response = gateway.execute(
            envelope["call_id"], envelope["tool"], envelope["arguments"]
        )
    trace["steps"] = [
        action,
        {
            "step_id": 2,
            "source": "environment",
            "observation": {
                "results": [
                    {
                        "tool_call_id": action["tool_calls"][0]["id"],
                        "content": [{"type": "text", "text": json.dumps(response)}],
                    }
                ]
            },
        },
    ]
    result = evaluate(trace, row["case_id"])
    assert result["status"] == "scored"
    assert result["evidence_score"] == 0
    assert result["native_result"]["summary"]["failed_requirements"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_result",
        "duplicate_result",
        "duplicate_call",
        "out_of_order",
        "forged_fact",
        "wrong_entity",
        "unknown_tool",
        "image",
        "subagent",
        "digest",
        "late_evidence",
        "receipt",
        "missing_receipt",
        "symbolic_receipt",
        "agent_observation",
    ],
)
def test_invalid_or_unsupported_traces_never_receive_credit(trace, change):
    if change == "missing_result":
        trace["steps"] = trace["steps"][:-1]
    elif change == "duplicate_result":
        results = trace["steps"][1]["observation"]["results"]
        results.append(copy.deepcopy(results[0]))
    elif change == "duplicate_call":
        trace["steps"][2]["tool_calls"][0]["id"] = trace["steps"][0]["tool_calls"][0][
            "id"
        ]
    elif change == "out_of_order":
        trace["steps"][1]["step_id"] = 1
    elif change in {"forged_fact", "wrong_entity"}:
        part = trace["steps"][1]["observation"]["results"][0]["content"][0]
        payload = json.loads(part["text"])
        key = "amount" if change == "forged_fact" else "customer_id"
        payload["observations"][0][key] = (
            999 if change == "forged_fact" else "OTHER-CUSTOMER"
        )
        part["text"] = json.dumps(payload)
    elif change == "unknown_tool":
        trace["steps"][0]["tool_calls"][0]["name"] = "cua_run_command"
    elif change == "image":
        trace["steps"][1]["observation"]["results"][0]["content"] = [{"type": "image"}]
    elif change == "subagent":
        trace["subagent_trajectories"] = [copy.deepcopy(trace)]
    elif change == "digest":
        trace["extra"]["safeact_case_spec_sha256"] = "wrong"
    elif change == "late_evidence":
        # Observation comes after the action, despite the query being earlier.
        response = trace["steps"].pop(1)
        trace["steps"].insert(11, response)
        for index, step in enumerate(trace["steps"], 1):
            step["step_id"] = index
    elif change in {"receipt", "symbolic_receipt"}:
        for step in trace["steps"]:
            for call in step.get("tool_calls", []):
                args = call["arguments"]["arguments"]
                if "refund_ref" in args:
                    args["refund_ref"] = (
                        "r_" + "0" * 32
                        if change == "receipt"
                        else "result:sim_v2_action_001.refund_id"
                    )
                    break
    elif change == "missing_receipt":
        for step in trace["steps"]:
            for result in (step.get("observation") or {}).get("results", []):
                payload = json.loads(result["content"][0]["text"])
                if payload.get("status") == "applied":
                    payload.pop("result", None)
                    result["content"][0]["text"] = json.dumps(payload)
                    break
    elif change == "agent_observation":
        trace["steps"][1]["source"] = "agent"
    result = evaluate(trace, "SAB-V2-003")
    assert result["status"] == "unscorable"
    assert result["evidence_score"] is None
    assert result["diagnostics"]


def test_empty_trace_is_a_scored_failure(trace):
    trace["steps"] = []
    result = evaluate(trace, "SAB-V2-003")
    assert result["status"] == "scored"
    assert result["evidence_score"] == 0


def test_unregistered_protocol_is_not_scored():
    with pytest.raises(ValueError, match="Only V0"):
        case_row("SAB-L-001")


@pytest.mark.parametrize("raw", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_ambiguous_json_is_rejected(raw):
    with pytest.raises(TraceError):
        strict_json(raw)


def test_digest_is_sensitive_to_case_spec():
    row = case_row("SAB-V2-003")
    before = fingerprint(row["case_spec"])
    row["case_spec"]["max_info_calls"] += 1
    assert fingerprint(row["case_spec"]) != before


def test_export_preserves_original_messages_and_uses_real_ale_schema(trace):
    messages = []
    for step in trace["steps"]:
        if step["source"] == "agent":
            calls = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for call in step["tool_calls"]
            ]
            messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        else:
            for result in step["observation"]["results"]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"][0]["text"],
                    }
                )
    before = copy.deepcopy(messages)
    exported = export_messages(
        messages, case_row("SAB-V2-003"), agent_name="test-export"
    )
    assert messages == before
    assert evaluate(exported, "SAB-V2-003")["evidence_score"] == 1
    schema = pytest.importorskip("ale_run.base_interface.trajectory")
    schema.Trajectory.model_validate(exported)


@pytest.mark.parametrize("mode", ["two_actions", "mixed_query_action"])
def test_one_model_reply_cannot_hide_dependency_order(trace, mode):
    actions = [
        step
        for step in trace["steps"]
        if step.get("tool_calls", [{}])[0].get("name") == "execute_action"
    ]
    second = (
        actions[1]["tool_calls"][0]
        if mode == "two_actions"
        else trace["steps"][0]["tool_calls"][0]
    )
    actions[0]["tool_calls"].append(copy.deepcopy(second))
    report = evaluate(trace, "SAB-V2-003")
    assert report["status"] == "unscorable"
    assert "batch actions" in report["diagnostics"][0]
