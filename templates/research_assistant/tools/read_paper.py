#!/usr/bin/env python3
"""Read a paper from the mock corpus."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "research.read_paper"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    args = parser.parse_args()

    papers = load_json("resources/papers.json")["papers"]
    paper = next((item for item in papers if item["paper_id"] == args.paper_id), None)
    if paper is None:
        append_log(TOOL_NAME, vars(args), "error", f"Paper not found: {args.paper_id}")
        emit({"status": "error", "summary": f"Paper not found: {args.paper_id}"})
        return 2

    artifact_ref = write_artifact(
        TOOL_NAME,
        "paper_fulltext_snapshot",
        {"paper": paper},
        source_refs=["resources/papers.json"],
    )
    summary = f"Read paper {args.paper_id}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "paper": paper, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
