#!/usr/bin/env python3
"""Shared helpers for customer policy QA mock tools."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENV_ID = "customer_policy_qa"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE_ROOT / "output"
ARTIFACT_DIR = OUTPUT_DIR / "action_artifacts"
ACTION_LOG_PATH = OUTPUT_DIR / "action_log.jsonl"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(relative_path: str) -> Any:
    path = WORKSPACE_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def relative_to_workspace(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def write_artifact(
    tool_name: str,
    artifact_type: str,
    payload: dict,
    source_refs: list[str] | None = None,
) -> dict:
    artifact_id = new_id(artifact_type)
    artifact = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "created_at": iso_now(),
        "created_by_tool": tool_name,
        "env_id": ENV_ID,
        "source_refs": source_refs or [],
        "payload": payload,
    }
    artifact_path = ARTIFACT_DIR / f"{artifact_id}.json"
    write_json(artifact_path, artifact)
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "path": relative_to_workspace(artifact_path),
    }


def append_log(
    tool_name: str,
    args: dict,
    result_status: str,
    summary: str,
    artifact_refs: list[dict] | None = None,
    metadata: dict | None = None,
) -> dict:
    event = {
        "schema_version": 1,
        "event_id": new_id("evt"),
        "timestamp": iso_now(),
        "env_id": ENV_ID,
        "tool_name": tool_name,
        "args": args,
        "result_status": result_status,
        "summary": summary,
        "artifact_refs": artifact_refs or [],
        "metadata": metadata or {},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ACTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        json.dump(event, handle, sort_keys=True)
        handle.write("\n")
    return event


def emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
