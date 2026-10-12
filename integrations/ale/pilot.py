"""Generate ALE-shaped reference fixtures by executing original SafeAct cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import evaluate, fingerprint
from .runtime import ROOT, SUPPORTED_PROTOCOLS, PlatformUnavailable, case_row, runtime

V2_PILOT_CASES = ("SAB-V2-001", "SAB-V2-003", "SAB-V2-005", "SAB-V2-007", "SAB-V2-009")
PILOT_CASES = tuple(
    f"SAB-V{protocol}-{index:03d}" for protocol in range(4) for index in (1, 3, 5, 7, 9)
)


def all_case_ids():
    dataset = json.loads((ROOT / "data/safeact/cases.json").read_text(encoding="utf-8"))
    return tuple(
        row["case_id"]
        for row in dataset["cases"]
        if row["protocol"] in SUPPORTED_PROTOCOLS
    )


def reference_trajectory(case_id, *, action_order=None):
    row = case_row(case_id)
    turn = [0]
    steps = []
    with runtime(row, turn, seed=b"safeact-ale-reference-fixture-v1") as (
        gateway,
        run,
        live,
    ):
        builder = {
            "v0": run.build_simulated_no_action_execution,
            "v1": run.build_simulated_state_action_execution,
            "v2": run.build_simulated_multi_step_execution,
            "v3": run.build_simulated_multi_step_execution,
        }[row["protocol"]]
        record = builder({"env_id": row["env_id"]}, row["case_spec"])
        if action_order is not None:
            actions = [
                event
                for event in record["events"]
                if event["type"] == "CONSEQUENTIAL_CALL"
            ]
            if sorted(action_order) != list(range(len(actions))):
                raise ValueError("action_order must permute all consequential calls")
            reordered = iter(actions[index] for index in action_order)
            record["events"] = [
                next(reordered) if event["type"] == "CONSEQUENTIAL_CALL" else event
                for event in record["events"]
            ]
        values = {}
        for event in record["events"]:
            call_step = len(steps) + 1
            turn[0] = call_step
            arguments = live.map_values(
                event["arguments"],
                lambda value: (
                    values.get(value, value) if isinstance(value, str) else value
                ),
            )
            envelope = {"tool": event["tool"], "arguments": arguments}
            if event["type"] == "INFO_CALL":
                name = "information_call"
                response = gateway.call(event["tool"], arguments)
            else:
                name = "execute_action"
                envelope["call_id"] = event["call_id"]
                response = gateway.execute(event["call_id"], event["tool"], arguments)
                for field, value in response.get("result", {}).items():
                    values[f"result:{event['call_id']}.{field}"] = value
            call_id = f"ale_call_{call_step}"
            steps.append(
                {
                    "step_id": call_step,
                    "source": "agent",
                    "tool_calls": [
                        {"id": call_id, "name": name, "arguments": envelope}
                    ],
                }
            )
            steps.append(
                {
                    "step_id": call_step + 1,
                    "source": "environment",
                    "observation": {
                        "results": [
                            {
                                "tool_call_id": call_id,
                                "is_error": False,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(
                                            response, ensure_ascii=False
                                        ),
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        if row["protocol"] == "v0":
            turn[0] = len(steps) + 1
            terminal = record["terminal"]
            gateway.stop(terminal)
            steps.append(
                {
                    "step_id": turn[0],
                    "source": "agent",
                    "message": json.dumps({"terminal": terminal}),
                }
            )
        native = gateway.score()
        if not native["success"]:
            raise RuntimeError(f"Original reference failed: {case_id}: {native}")
        if row["protocol"] in {"v2", "v3"}:
            public, _patches = live.public_scenario(gateway.base.scenario)
        else:
            public = gateway.base.scenario
    trace = {
        "schema_version": "ALE-v1.0",
        "episode_id": f"reference-{case_id}",
        "agent": {"name": "safeact-reference-fixture", "model": None},
        "task_path": f"safeact/{case_id}",
        "variant_index": 0,
        "instruction": public.get("user_goal", ""),
        "steps": steps,
        "final_metrics": {"reward": None, "status": "completed"},
        "extra": {
            "safeact_case_id": case_id,
            "safeact_protocol": row["protocol"],
            "safeact_case_spec_sha256": fingerprint(row["case_spec"]),
            "fixture_only": True,
        },
    }
    return trace, public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--all-cases", action="store_true", help="Validate every released V0–V3 case"
    )
    parser.add_argument(
        "--allow-unavailable-platform",
        action="store_true",
        help="Record platform failures; default is fail closed",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    unavailable = []
    for case_id in all_case_ids() if args.all_cases else PILOT_CASES:
        try:
            trace, public = reference_trajectory(case_id)
        except PlatformUnavailable as exc:
            if not args.allow_unavailable_platform:
                raise
            unavailable.append({"case_id": case_id, "diagnostic": str(exc)})
            continue
        report = evaluate(trace, case_id)
        if report["evidence_score"] != 1.0:
            raise RuntimeError(f"Reference adapter failed: {case_id}")
        for suffix, payload in (
            ("trajectory", trace),
            ("public", public),
            ("evidence", report),
        ):
            path = args.output_dir / f"{case_id}.{suffix}.json"
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        reports.append(
            {
                key: report[key]
                for key in ("case_id", "status", "endpoint_score", "evidence_score")
            }
        )
    summary = {
        "fixture_only": True,
        "model_runs": 0,
        "ale_vm_runs": 0,
        "reference_cases": len(reports),
        "unavailable_cases": unavailable,
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
