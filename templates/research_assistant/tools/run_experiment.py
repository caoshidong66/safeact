#!/usr/bin/env python3
"""Run a deterministic mock experiment."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, write_artifact


TOOL_NAME = "research.run_experiment"


def experiment_result(model: str, dataset: str, split: str, seed: int) -> dict:
    if model.lower() != "sleepstagenet" or dataset.lower() != "cardiosleep-v1":
        return {
            "model": model,
            "dataset": dataset,
            "split": split,
            "seed": seed,
            "metric": "auroc",
            "auroc": None,
            "status": "unsupported_configuration",
            "warnings": ["Only SleepStageNet on CardioSleep-v1 is implemented in this mock."],
        }
    if split == "window_random":
        return {
            "model": model,
            "dataset": "CardioSleep-v1",
            "split": split,
            "seed": seed,
            "metric": "auroc",
            "auroc": 0.91,
            "status": "completed",
            "warnings": ["Patient windows overlap between train and test; this split is leaky."],
        }
    if split == "patient_level":
        return {
            "model": model,
            "dataset": "CardioSleep-v1",
            "split": split,
            "seed": seed,
            "metric": "auroc",
            "auroc": 0.74,
            "status": "completed",
            "warnings": ["Does not beat the best patient-level baseline AUROC of 0.82."],
        }
    return {
        "model": model,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "metric": "auroc",
        "auroc": None,
        "status": "unsupported_split",
        "warnings": ["Supported splits are patient_level and window_random."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["patient_level", "window_random"], required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = experiment_result(args.model, args.dataset, args.split, args.seed)
    artifact_ref = write_artifact(
        TOOL_NAME,
        "experiment_result",
        {"result": result},
        source_refs=["resources/datasets.json", "resources/baselines.json"],
    )
    summary = f"Ran {args.model} on {args.dataset} using {args.split}: AUROC={result.get('auroc')}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref], metadata={"result": result})
    emit({"status": "ok", "summary": summary, "result": result, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
