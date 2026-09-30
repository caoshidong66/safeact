#!/usr/bin/env python3
"""Return mock baseline results."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "research.get_baseline_results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="patient_level")
    parser.add_argument("--metric", default="auroc")
    args = parser.parse_args()

    baselines = load_json("resources/baselines.json")["baselines"]
    result = next(
        (
            item
            for item in baselines
            if item["dataset_id"].lower() == args.dataset.lower()
            and item["split"] == args.split
            and item["metric"] == args.metric
        ),
        None,
    )
    if result is None:
        append_log(TOOL_NAME, vars(args), "error", "Baseline result not found.")
        emit({"status": "error", "summary": "Baseline result not found."})
        return 2

    artifact_ref = write_artifact(
        TOOL_NAME,
        "baseline_results",
        {"baseline": result},
        source_refs=["resources/baselines.json"],
    )
    summary = f"Returned {args.metric} baselines for {args.dataset} on {args.split}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "baseline": result, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
