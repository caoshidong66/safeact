#!/usr/bin/env python3
"""Search the mock paper corpus."""

from __future__ import annotations

import argparse

from _tool_common import append_log, emit, load_json, write_artifact


TOOL_NAME = "research.paper_search"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    corpus = load_json("resources/papers.json")["papers"]
    terms = [term.lower() for term in args.query.split() if term.strip()]
    matches = []
    for paper in corpus:
        haystack = " ".join(
            [
                paper.get("paper_id", ""),
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(paper.get("keywords", [])),
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            matches.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "venue": paper["venue"],
                    "abstract": paper["abstract"],
                    "score": score,
                }
            )
    matches = sorted(matches, key=lambda item: (-item["score"], item["paper_id"]))[: args.limit]
    artifact_ref = write_artifact(
        TOOL_NAME,
        "paper_search_results",
        {"query": args.query, "matches": matches},
        source_refs=["resources/papers.json"],
    )
    summary = f"Found {len(matches)} paper(s) for query {args.query!r}."
    append_log(TOOL_NAME, vars(args), "ok", summary, [artifact_ref])
    emit({"status": "ok", "summary": summary, "matches": matches, "artifact_refs": [artifact_ref]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
