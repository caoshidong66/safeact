#!/usr/bin/env python3
"""Check a standalone SafeActBench runtime without requesting model inference."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cli-matrix", action="store_true",
                        help="Also run the command-line runner and evaluator for every domain/protocol pair; writes runtime workspaces.")
    parser.add_argument("--tool-backends", action="store_true",
                        help="Probe each protocol/evidence backend through ToolGateway; writes Legacy tool artifacts.")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Run focused CLI/tool checks without repeating reference replay or mutation tests.")
    parser.add_argument("--expect-cases", type=int, default=656)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.repo_root.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "env/tools"), str(root / "scripts"), str(root / "agents"), str(root)]
    runner = importlib.import_module("run_agent")
    evaluator = importlib.import_module("evaluate_workspace")
    batch = importlib.import_module("run_safeact_agent_batch")
    for module in (runner, evaluator, batch):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Imported an external runtime module: {module.__file__}")
    tasks = batch.discover_tasks(root)
    errors = []
    results = []
    negative_results = []
    cli_results = []
    tool_results = []
    catalogs = {}
    cases = {}
    profiles = {}
    manifest = json.loads((root / "env/manifest.json").read_text())
    environments = {item["id"]: item for item in manifest["environments"]}
    if len(tasks) != args.expect_cases:
        errors.append({"check": "case_count", "actual": len(tasks), "expected": args.expect_cases})
    for task in tasks:
        catalog = task["catalog"]
        if catalog not in catalogs:
            payload = json.loads((root / "env" / catalog).read_text())
            catalogs[catalog] = {item.get("scenario_id") or item.get("episode_id"): item
                                 for item in payload.get("cases", payload.get("episodes", []))}
        cases[task["case_id"]] = catalogs[catalog][task["case_id"]]
        env_id = task["env_id"]
        if env_id not in profiles:
            profiles[env_id] = json.loads((root / "env" / environments[env_id]["profile_path"]).read_text())

    builders = {
        "legacy": runner.build_simulated_decision,
        "v0": runner.build_simulated_no_action_execution,
        "v1": runner.build_simulated_state_action_execution,
        "v2": runner.build_simulated_v2_execution,
        "v3": runner.build_simulated_multi_step_execution,
    }
    validators = {
        "v0": evaluator.validate_no_action_execution_record,
        "v1": evaluator.validate_single_commit_execution_record,
        "v2": evaluator.validate_linear_workflow_execution_record,
        "v3": evaluator.validate_multi_step_execution_record,
    }
    summary_keys = {"v0": "no_action_summary", "v1": "state_action_summary",
                    "v2": "linear_workflow_summary", "v3": "multi_step_summary"}
    success_keys = {"v0": "no_action_success", "v1": "safe_commit_success",
                    "v2": "workflow_success", "v3": "workflow_success"}
    success_statuses = {"v0": "pass_no_action", "v1": "pass_safe_commit",
                        "v2": "pass_linear_workflow_success", "v3": "pass_workflow_success"}
    exposures = {"legacy": runner.exposed_scenario, "v0": runner.exposed_no_action_episode,
                 "v1": runner.exposed_state_action_episode, "v2": runner.exposed_linear_workflow_episode,
                 "v3": runner.exposed_multi_step_episode}

    def score(task, payload):
        protocol = task["protocol"]
        case = cases[task["case_id"]]
        if protocol == "legacy":
            validation_errors = evaluator.validate_decision(payload, task["env_id"])
            scoring = evaluator.score_legacy_output(payload.get("decision"), case.get("gold_decision"),
                                                     payload.get("confirm"), case.get("gold_confirm"))
            record = {"status": "pass_gold_score" if not validation_errors and scoring["decision_correct"] is True
                      else "fail_gold_score", "contract_errors": validation_errors, "execution_errors": [],
                      "legacy_scoring": scoring}
        else:
            contract, execution, summary = validators[protocol](payload, case, task["env_id"])
            status = ("fail_contract" if contract else "fail_execution_record" if execution else
                      success_statuses[protocol] if summary.get(success_keys[protocol]) is True else "fail_reference")
            record = {"status": status, "contract_errors": contract, "execution_errors": execution,
                      summary_keys[protocol]: summary}
        return batch.task_succeeded(task, record), record

    gold_payloads = {}
    forbidden_public = {"gold_decision", "gold_confirm", "gold_claims", "gold_rationale",
                        "simulated_trace", "frozen_evidence_deltas", "expected_outcome",
                        "expected_stop_outcome", "expected_completed_commits", "expected_failed_commit"}
    def hidden_paths(value, path="$"):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden_public:
                    found.append(path + "." + key)
                found.extend(hidden_paths(item, path + "." + key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                found.extend(hidden_paths(item, path + f"[{i}]"))
        return found

    for index, task in enumerate(tasks, 1):
        case_id, protocol, env_id = task["case_id"], task["protocol"], task["env_id"]
        case = cases[case_id]
        try:
            payload = builders[protocol](profiles[env_id], case)
            gold_payloads[case_id] = payload
            if args.skip_reference:
                continue
            success, record = score(task, payload)
            public = exposures[protocol](case)
            leaked = hidden_paths(public)
            row = {"case_id": case_id, "protocol": protocol, "env_id": env_id,
                   "expected_outcome": task["expected_outcome"], "passed": success,
                   "contract_errors": record["contract_errors"], "execution_errors": record["execution_errors"],
                   "public_hidden_fields": leaked, "evaluation": record}
            results.append(row)
            if not success or leaked:
                errors.append({"check": "reference_replay", "case_id": case_id, "passed": success,
                               "contract_errors": record["contract_errors"],
                               "execution_errors": record["execution_errors"], "public_hidden_fields": leaked})
        except Exception:
            errors.append({"check": "reference_replay_exception", "case_id": case_id, "traceback": traceback.format_exc()})
        if index % 50 == 0:
            print(json.dumps({"check": "reference_replay", "checked": index, "total": len(tasks),
                              "errors": len(errors), "seconds": round(time.monotonic() - started, 2)}), flush=True)

    def negative(task, payload, name):
        success, record = score(task, payload)
        row = {"case_id": task["case_id"], "protocol": task["protocol"], "mutation": name,
               "rejected": not success, "evaluation": record}
        negative_results.append(row)
        if success:
            errors.append({"check": "negative_control", "case_id": task["case_id"], "mutation": name})

    for protocol in (() if args.skip_reference else ("legacy", "v0", "v1", "v2", "v3")):
        task = next(t for t in tasks if t["protocol"] == protocol)
        payload = copy.deepcopy(gold_payloads[task["case_id"]])
        if protocol == "legacy":
            payload["decision"] = "invalid_decision"
            mutation = "invalid_decision"
        elif protocol == "v0":
            payload["events"].append({"type": "CONSEQUENTIAL_CALL", "call_id": "invalid_commit",
                                      "tool": "nonexistent_action", "arguments": {}})
            mutation = "consequential_action_in_v0"
        else:
            action = next(e for e in payload["events"] if e.get("type") == "CONSEQUENTIAL_CALL")
            action["tool"] = "nonexistent_action"
            mutation = "nonexistent_consequential_tool"
        negative(task, payload, mutation)

    def break_reference(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item.startswith("result:"):
                    value[key] = "result:missing_predecessor.missing_receipt"
                    return True
                if break_reference(item):
                    return True
        elif isinstance(value, list):
            return any(break_reference(item) for item in value)
        return False

    for protocol in (() if args.skip_reference else ("v2", "v3")):
        for task in (t for t in tasks if t["protocol"] == protocol):
            payload = copy.deepcopy(gold_payloads[task["case_id"]])
            if any(break_reference(e.get("arguments", {})) for e in payload["events"]
                   if e.get("type") == "CONSEQUENTIAL_CALL"):
                negative(task, payload, "missing_dependency_receipt")
                break
        else:
            errors.append({"check": "negative_control_missing_reference", "protocol": protocol})

    listing = subprocess.run([sys.executable, "-B", str(root / "scripts/run_safeact_agent_batch.py"),
                              "--repo-root", str(root), "--list-only"], cwd=root,
                             capture_output=True, text=True, timeout=120)
    listed = json.loads(listing.stdout) if listing.returncode == 0 else {}
    if listing.returncode or listed.get("case_count") != len(tasks) or listed.get("tasks") != tasks:
        errors.append({"check": "list_cli", "returncode": listing.returncode, "stderr": listing.stderr})

    if args.cli_matrix:
        selected = {}
        for task in tasks:
            selected.setdefault((task["protocol"], task["env_id"]), task)
        for index, task in enumerate(selected.values(), 1):
            common = ["--env", task["env_id"], task["case_flag"], task["case_id"]]
            command = [sys.executable, "-B", str(root / "env/tools/run_agent.py"), *common, "--mode", "simulate"]
            run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
            ev = subprocess.run([sys.executable, "-B", str(root / "env/tools/evaluate_workspace.py"), *common],
                                cwd=root, capture_output=True, text=True, timeout=120)
            record = batch.parse_eval_output(ev.stdout) if ev.returncode == 0 else {}
            passed = run.returncode == ev.returncode == 0 and batch.task_succeeded(task, record)
            cli_results.append({"case_id": task["case_id"], "protocol": task["protocol"],
                                "env_id": task["env_id"], "passed": passed, "run_returncode": run.returncode,
                                "evaluate_returncode": ev.returncode, "run_stderr": run.stderr,
                                "evaluate_stderr": ev.stderr, "evaluation": record})
            if not passed:
                errors.append({"check": "cli_matrix", "case_id": task["case_id"], "run_returncode": run.returncode,
                               "evaluate_returncode": ev.returncode, "run_stderr": run.stderr,
                               "evaluate_stderr": ev.stderr, "stdout": ev.stdout[:2000]})
            print(json.dumps({"check": "cli_matrix", "checked": index, "total": len(selected),
                              "case_id": task["case_id"], "passed": passed}), flush=True)

    if args.tool_backends:
        adapter = importlib.import_module("coding_cli_safeact_agent")
        selected = {}
        for task in tasks:
            backend = task["evaluation_profile"]["evidence_backend"]
            domain = task["env_id"] if task["protocol"] == "legacy" and backend == "catalog_authored_evidence" else ""
            selected.setdefault((task["protocol"], backend, domain), task)
        for (protocol, backend, _domain), task in selected.items():
            case_id, env_id = task["case_id"], task["env_id"]
            try:
                scenario = exposures[protocol](cases[case_id])
                workspace = root / "env" / environments[env_id]["workspace_path"]
                prepare_workspace = getattr(runner, "prepare_workspace", None)
                if prepare_workspace is not None:
                    prepare_workspace(root / "env", env_id, workspace)
                gateway = adapter.ToolGateway(protocol, scenario, cases[case_id], workspace)
                if protocol == "legacy":
                    tool = gateway.allowed_tools[0]
                    arguments = {}
                else:
                    info = next(e for e in gold_payloads[case_id]["events"] if e.get("type") == "INFO_CALL")
                    tool, arguments = info["tool"], info["arguments"]
                response = gateway.call(tool, arguments)
                denied = gateway.call("nonexistent_information_tool", {})
                passed = response.get("status") == "ok" and denied.get("status") == "error"
                row = {"case_id": case_id, "protocol": protocol, "evidence_backend": backend,
                       "tool": tool, "arguments": arguments, "passed": passed,
                       "response_status": response.get("status"), "denied_response": denied,
                       "response": response}
                tool_results.append(row)
                if not passed:
                    errors.append({"check": "tool_backend", "case_id": case_id,
                                   "response": response, "denied": denied})
            except Exception:
                errors.append({"check": "tool_backend_exception", "case_id": case_id,
                               "traceback": traceback.format_exc()})
            print(json.dumps({"check": "tool_backend", "protocol": protocol, "backend": backend,
                              "checked": len(tool_results), "errors": len(errors)}), flush=True)

    result = {"passed": not errors, "scope": "Offline reference replay, integrity, and rejection checks; no model inference.",
              "counts": {"cases": len(tasks), "reference_replays": len(results),
                         "reference_replays_passed": sum(r["passed"] for r in results),
                         "negative_controls": len(negative_results), "cli_matrix_cases": len(cli_results),
                         "tool_backend_probes": len(tool_results),
                         "protocols": dict(Counter(t["protocol"] for t in tasks)),
                         "environments": dict(Counter(t["env_id"] for t in tasks))},
              "seconds": round(time.monotonic() - started, 2), "errors": errors,
              "reference_replays": results, "negative_controls": negative_results,
              "cli_matrix": cli_results, "tool_backends": tool_results}
    dump(args.output.resolve(), result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"reference_replays", "negative_controls", "cli_matrix", "tool_backends"}}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
