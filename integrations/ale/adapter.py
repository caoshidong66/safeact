"""Replay trusted semantic-tool observations without changing ALE outcome scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .runtime import case_row, runtime

SUPPORTED_TOOLS = frozenset({"information_call", "execute_action", "list_tools"})
TOKEN = re.compile(r"^r_[0-9a-f]{32}$")


class TraceError(ValueError):
    """A trace violates the semantic-tool recording contract."""


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def strict_json(text):
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise TraceError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value):
        raise TraceError(f"Non-finite JSON number: {value}")

    return json.loads(
        text, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


def result_payload(result):
    content = result.get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or content[0].get("type") != "text"
    ):
        raise TraceError(
            "A semantic tool result must contain exactly one JSON text part"
        )
    try:
        payload = strict_json(content[0]["text"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceError(f"Invalid semantic tool result: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceError("A semantic tool result must be a JSON object")
    return payload


def check_shape(trace, row):
    if trace.get("schema_version") != "ALE-v1.0":
        raise TraceError("Expected ALE-v1.0")
    if trace.get("continued_trajectory_ref") or trace.get("subagent_trajectories"):
        raise TraceError("Spanning and subagent trajectories need causal merging first")
    extra = trace.get("extra", {})
    if not isinstance(extra, dict) or extra.get("safeact_case_id") != row["case_id"]:
        raise TraceError("Missing or mismatched extra.safeact_case_id")
    if extra.get("safeact_case_spec_sha256") != fingerprint(row["case_spec"]):
        raise TraceError("Missing or mismatched case specification digest")
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise TraceError("steps must be a list")
    previous = 0
    for step in steps:
        if not isinstance(step, dict):
            raise TraceError("Each step must be an object")
        step_id = step.get("step_id")
        if type(step_id) is not int or step_id <= previous:
            raise TraceError(
                "step_id must be positive and strictly increasing in recorded order"
            )
        previous = step_id
        if step.get("source") not in {"agent", "environment", "user", "system"}:
            raise TraceError("Invalid step source")
        calls = step.get("tool_calls", [])
        if not isinstance(calls, list) or any(
            not isinstance(call, dict) for call in calls
        ):
            raise TraceError("tool_calls must be a list of objects")
        if calls and step["source"] != "agent":
            raise TraceError("Only agent steps may contain tool calls")
        observation = step.get("observation")
        if observation is not None and step["source"] != "environment":
            raise TraceError("Only environment steps may contain tool observations")
    return steps


def compare_response(actual, expected, bindings):
    """Bind observed receipt fields only after a successful producer response."""
    candidate = dict(bindings)
    expected_result = expected.get("result")
    if expected.get("status") == "applied" and isinstance(expected_result, dict):
        actual_result = actual.get("result")
        if not isinstance(actual_result, dict) or set(actual_result) != set(
            expected_result
        ):
            raise TraceError(
                "Action receipt fields differ from the producer's declared schema"
            )
        for field, replay_value in expected_result.items():
            observed_value = actual_result[field]
            if (
                not isinstance(observed_value, str)
                or TOKEN.fullmatch(observed_value) is None
            ):
                raise TraceError(
                    "Action receipt must be an opaque value returned by the gateway"
                )
            if observed_value in candidate or replay_value in candidate.values():
                raise TraceError(
                    "An action receipt was reused by another producer or field"
                )
            candidate[observed_value] = replay_value

    def translate(value):
        if isinstance(value, dict):
            return {key: translate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [translate(item) for item in value]
        return candidate.get(value, value) if isinstance(value, str) else value

    if translate(actual) != expected:
        raise TraceError(
            "Recorded observation differs from the authoritative gateway response"
        )
    bindings.update(candidate)


def replay(trace, row, gateway, turn, live):
    steps = check_shape(trace, row)
    pending = {}
    seen = set()
    bindings = {}
    observed_calls = 0
    for step in steps:
        names = [call.get("name") for call in step.get("tool_calls", [])]
        if names.count("execute_action") > 1 or (
            "execute_action" in names and "information_call" in names
        ):
            raise TraceError(
                "A model reply cannot batch actions or mix queries with an action"
            )
        for call in step.get("tool_calls", []):
            call_id, name, arguments = (
                call.get("id"),
                call.get("name"),
                call.get("arguments"),
            )
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                raise TraceError("Tool call IDs must be nonempty and globally unique")
            if not isinstance(name, str) or name not in SUPPORTED_TOOLS:
                raise TraceError(f"Unsupported tool surface: {name}")
            if not isinstance(arguments, dict):
                raise TraceError("Tool arguments must be an object")
            if name != "list_tools" and (
                not isinstance(arguments.get("tool"), str)
                or not isinstance(arguments.get("arguments"), dict)
            ):
                raise TraceError(
                    "Business tool names and argument objects are required"
                )
            seen.add(call_id)
            expected = None
            if name == "execute_action":
                if pending:
                    raise TraceError(
                        "Action emitted before prior tool responses were observed"
                    )
                if set(arguments) != {"call_id", "tool", "arguments"}:
                    raise TraceError("Invalid execute_action envelope")
                turn[0] = step["step_id"]
                translated = live.map_values(
                    arguments["arguments"],
                    lambda value: (
                        bindings.get(value, value) if isinstance(value, str) else value
                    ),
                )
                expected = gateway.execute(
                    arguments["call_id"], arguments["tool"], translated
                )
            elif name == "information_call":
                if set(arguments) != {"tool", "arguments"}:
                    raise TraceError("Invalid information_call envelope")
            elif arguments:
                raise TraceError("list_tools accepts no arguments")
            pending[call_id] = (name, arguments, step["step_id"], expected)

        observation = step.get("observation")
        if observation is None:
            continue
        if not isinstance(observation, dict) or observation.get("error"):
            raise TraceError("Invalid or failed environment observation")
        results = observation.get("results", [])
        if not isinstance(results, list):
            raise TraceError("Observation results must be a list")
        for result in results:
            if not isinstance(result, dict):
                raise TraceError("Tool results must be objects")
            call_id = result.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise TraceError("Orphan or duplicate tool result")
            name, arguments, call_step, expected = pending.pop(call_id)
            if result.get("is_error", False):
                raise TraceError("Transport errors do not establish business evidence")
            actual = result_payload(result)
            if name == "information_call":
                turn[0] = call_step
                expected = gateway.call(arguments["tool"], arguments["arguments"])
            elif name == "list_tools":
                expected = gateway.describe()
            compare_response(actual, expected, bindings)
            observed_calls += 1
    if pending:
        raise TraceError("Missing tool results at end of trajectory")
    return gateway.score(), observed_calls


def evaluate(trace, case_id):
    """Return a sidecar score; callers must supply trusted host-collected traces."""
    if not isinstance(trace, dict):
        raise TraceError("Trajectory must be a JSON object")
    row = case_row(case_id)
    metrics = trace.get("final_metrics") or {}
    if not isinstance(metrics, dict):
        raise TraceError("final_metrics must be an object or null")
    endpoint = metrics.get("reward")
    if endpoint is not None and (
        type(endpoint) not in (int, float)
        or not math.isfinite(endpoint)
        or not 0 <= endpoint <= 1
    ):
        raise TraceError("ALE reward must be a finite number in [0, 1] or null")
    report = {
        "schema_version": "safeact-ale-sidecar-v1",
        "case_id": case_id,
        "case_spec_sha256": fingerprint(row["case_spec"]),
        "trajectory_sha256": fingerprint(trace),
        "endpoint_score": endpoint,
        "evidence_score": None,
        "status": "unscorable",
        "diagnostics": [],
    }
    turn = [0]
    with runtime(row, turn, seed=b"safeact-ale-offline-replay-v1") as (
        gateway,
        _run,
        live,
    ):
        try:
            result, count = replay(trace, row, gateway, turn, live)
        except TraceError as exc:
            report["diagnostics"].append(str(exc))
            return report
        report.update(
            status="scored",
            evidence_score=float(result["success"]),
            observed_tool_calls=count,
            native_result=result,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(
            strict_json(args.trajectory.read_text(encoding="utf-8")), args.case_id
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Infrastructure or input error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("case_id", "status", "endpoint_score", "evidence_score")
            }
        )
    )
    return 0 if report["status"] == "scored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
