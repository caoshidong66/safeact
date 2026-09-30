#!/usr/bin/env python3
"""Read current and archived policy evidence from the mock workspace."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "support.read_policy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="refund_delay")
    parser.add_argument("--source", choices=["current", "macro", "both"], default="current")
    args = parser.parse_args()

    selected_paths: list[str] = []
    if args.source in {"current", "both"}:
        selected_paths.append("evidence/current_refund_policy.json")
    if args.source in {"macro", "both"}:
        selected_paths.append("evidence/stale_support_macro.json")

    evidence = [load_json(path) for path in selected_paths]
    artifact_ref = write_artifact(
        TOOL_NAME,
        "policy_evidence_snapshot",
        {
            "topic": args.topic,
            "requested_source": args.source,
            "evidence": evidence,
        },
        source_refs=selected_paths,
    )
    summary = f"Read {len(evidence)} policy evidence item(s) for topic {args.topic}."
    append_log(
        TOOL_NAME,
        vars(args),
        "ok",
        summary,
        [artifact_ref],
        metadata={"source_paths": selected_paths},
    )
    emit({"status": "ok", "summary": summary, "evidence": evidence, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
