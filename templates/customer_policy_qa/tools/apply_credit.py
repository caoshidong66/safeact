#!/usr/bin/env python3
"""Submit a mock account-credit action inside the customer support workspace."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, write_artifact


TOOL_NAME = "support.apply_delay_credit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="CUST-1042")
    parser.add_argument("--amount-usd", type=int, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence-ref", action="append", default=[])
    args = parser.parse_args()

    payload = {
        "case_id": args.case_id,
        "amount_usd": args.amount_usd,
        "reason": args.reason,
        "submitted": True,
        "external_customer_message": False,
        "account_effect": "credit_adjustment_request",
    }
    artifact_ref = write_artifact(
        TOOL_NAME,
        "credit_adjustment_request",
        payload,
        source_refs=args.evidence_ref,
    )
    summary = f"Submitted mock credit adjustment request for case {args.case_id}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "credit_request": payload, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
