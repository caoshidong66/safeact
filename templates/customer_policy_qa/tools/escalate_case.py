#!/usr/bin/env python3
"""Create an internal escalation artifact for a customer policy case."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, write_artifact


TOOL_NAME = "support.escalate_case"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="CUST-1042")
    parser.add_argument("--priority", choices=["normal", "high"], default="normal")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-ref", action="append", default=[])
    args = parser.parse_args()

    payload = {
        "case_id": args.case_id,
        "priority": args.priority,
        "reason": args.reason,
        "internal_only": True,
    }
    artifact_ref = write_artifact(
        TOOL_NAME,
        "case_escalation",
        payload,
        source_refs=args.evidence_ref,
    )
    summary = f"Created internal escalation for case {args.case_id}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "escalation": payload, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
