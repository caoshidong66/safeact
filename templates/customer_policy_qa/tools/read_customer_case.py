#!/usr/bin/env python3
"""Read a customer support case from the mock workspace."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "support.read_customer_case"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="CUST-1042")
    args = parser.parse_args()

    case = load_json("context/customer_case.json")
    status = "ok" if case.get("case_id") == args.case_id else "error"
    summary = f"Read customer case {case.get('case_id')}."
    if status == "error":
        summary = f"Requested case {args.case_id}, but workspace contains {case.get('case_id')}."

    artifact_ref = write_artifact(
        TOOL_NAME,
        "customer_case_snapshot",
        {"requested_case_id": args.case_id, "case": case},
        source_refs=["context/customer_case.json"],
    )
    append_log(TOOL_NAME, vars(args), status, summary, [artifact_ref])
    emit({"status": status, "summary": summary, "case": case, "artifact_refs": [artifact_ref]})
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
