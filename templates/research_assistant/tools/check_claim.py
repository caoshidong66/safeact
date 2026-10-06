#!/usr/bin/env python3
"""Check a mock research claim against available evidence."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "research.check_claim"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-id", required=True)
    args = parser.parse_args()

    claims = load_json("resources/claim_checks.json")["claims"]
    claim = next((item for item in claims if item["claim_id"] == args.claim_id), None)
    if claim is None:
        append_log(TOOL_NAME, vars(args), "error", f"Claim not found: {args.claim_id}")
        emit({"status": "error", "summary": f"Claim not found: {args.claim_id}"})
        return 2

    artifact_ref = write_artifact(
        TOOL_NAME,
        "claim_check",
        {"claim_check": claim},
        source_refs=["resources/claim_checks.json"],
    )
    summary = f"Checked claim {args.claim_id}: supported={claim['supported']}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref], metadata={"claim_supported": claim["supported"]})
    emit({"status": "ok", "summary": summary, "claim_check": claim, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
