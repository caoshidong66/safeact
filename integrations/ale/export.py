"""Export an existing trusted SafeAct rollout into ALE's trajectory schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import TraceError, fingerprint, strict_json
from .runtime import case_row


def export_messages(messages, row, *, agent_name, model=None, episode_id=None):
    """Preserve call/result order from the host runner's messages.json."""
    if not isinstance(messages, list):
        raise TraceError("messages.json must contain a list")
    steps = []
    for message in messages:
        if not isinstance(message, dict):
            raise TraceError("Messages must be objects")
        role = message.get("role")
        if role in {"system", "user", "assistant"}:
            step = {
                "step_id": len(steps) + 1,
                "source": "agent" if role == "assistant" else role,
                "message": message.get("content"),
            }
            calls = []
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = strict_json(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TraceError("Function arguments must decode to an object")
                calls.append(
                    {"id": call["id"], "name": function["name"], "arguments": arguments}
                )
            if calls:
                step["tool_calls"] = calls
        elif role == "tool":
            step = {
                "step_id": len(steps) + 1,
                "source": "environment",
                "observation": {
                    "results": [
                        {
                            "tool_call_id": message["tool_call_id"],
                            "is_error": False,
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    ]
                },
            }
        else:
            raise TraceError(f"Unsupported message role: {role}")
        steps.append(step)
    return {
        "schema_version": "ALE-v1.0",
        "episode_id": episode_id or row["case_id"],
        "agent": {"name": agent_name, "model": model},
        "task_path": f"safeact/{row['case_id']}",
        "variant_index": 0,
        "instruction": row["case_spec"].get("user_goal", ""),
        "steps": steps,
        "final_metrics": {"reward": None, "status": "completed"},
        "extra": {
            "safeact_case_id": row["case_id"],
            "safeact_case_spec_sha256": fingerprint(row["case_spec"]),
            "exported_from_safeact": True,
            "ale_vm_run": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_directory", type=Path)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = strict_json(
            (args.rollout_directory / "case_metadata.json").read_text(encoding="utf-8")
        )
        row = case_row(metadata["case_id"])
        if metadata.get("case_spec_sha256") != fingerprint(row["case_spec"]):
            raise TraceError(
                "Rollout was recorded against a different case specification"
            )
        messages = strict_json(
            (args.rollout_directory / "messages.json").read_text(encoding="utf-8")
        )
        trace = export_messages(
            messages,
            row,
            agent_name=args.agent_name,
            model=args.model,
            episode_id=args.rollout_directory.name,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Rollout export error: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {row['case_id']} to {args.output}")


if __name__ == "__main__":
    main()
