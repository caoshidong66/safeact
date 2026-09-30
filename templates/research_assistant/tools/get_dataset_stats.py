#!/usr/bin/env python3
"""Return mock dataset statistics."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "research.get_dataset_stats"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    datasets = load_json("resources/datasets.json")["datasets"]
    dataset = next((item for item in datasets if item["dataset_id"].lower() == args.dataset.lower()), None)
    if dataset is None:
        append_log(TOOL_NAME, vars(args), "error", f"Dataset not found: {args.dataset}")
        emit({"status": "error", "summary": f"Dataset not found: {args.dataset}"})
        return 2

    artifact_ref = write_artifact(
        TOOL_NAME,
        "dataset_stats",
        {"dataset": dataset},
        source_refs=["resources/datasets.json"],
    )
    summary = f"Returned dataset stats for {dataset['dataset_id']}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "dataset": dataset, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
