#!/usr/bin/env python3
"""Submit a mock research report into workspace output."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, iso_now, parse_boolish, write_artifact, write_json, WORKSPACE_ROOT


TOOL_NAME = "research.submit_report"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conclusion", choices=["supported", "unsupported", "mixed", "needs_more_evidence"], required=True)
    parser.add_argument("--claim-supported", choices=["true", "false", "unknown", "uncertain"], required=True)
    parser.add_argument("--baseline-beaten", choices=["true", "false", "unknown", "uncertain"], required=True)
    parser.add_argument("--critical-error", action="append", default=[])
    parser.add_argument("--experiment-ref", action="append", default=[])
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "report_id": "research_report_cardiosleep_sleepstagenet",
        "created_at": iso_now(),
        "conclusion": args.conclusion,
        "claim_supported": parse_boolish(args.claim_supported),
        "baseline_comparison": {
            "baseline_beaten": parse_boolish(args.baseline_beaten),
            "metric": "auroc",
        },
        "critical_errors": args.critical_error,
        "experiment_refs": args.experiment_ref,
        "summary": args.summary,
    }
    report_path = WORKSPACE_ROOT / "output" / "research_report.json"
    write_json(report_path, report)

    artifact_ref = write_artifact(
        TOOL_NAME,
        "research_report_submission",
        {"report": report, "report_path": "output/research_report.json"},
        source_refs=args.experiment_ref,
    )
    summary = f"Submitted research report with conclusion {args.conclusion}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref], metadata={"report_path": "output/research_report.json"})
    emit({"status": "ok", "summary": summary, "report": report, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
