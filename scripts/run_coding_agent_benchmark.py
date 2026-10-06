#!/usr/bin/env python3
"""Run SafeActBench with Codex CLI or Claude Code."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["codex", "claude"], required=True)
    parser.add_argument(
        "--strategy",
        choices=["baseline", "scgr-eg"],
        default=os.environ.get("SAFEACT_AGENT_STRATEGY", "baseline"),
        help="Optional inference strategy; baseline remains the default.",
    )
    parser.add_argument("--model", default=None, help="Optional model/provider alias; omit to use the CLI default.")
    parser.add_argument("--profile", default=None, help="Codex configuration profile.")
    parser.add_argument(
        "--cfuse-config",
        default=None,
        help="Optional cfuse config path; placed before the exec subcommand.",
    )
    parser.add_argument("--cli-bin", default=None, help="Override the codex or claude executable.")
    parser.add_argument(
        "--model-catalog",
        default=None,
        help="Trusted Codex model catalog mounted read-only by the adapter.",
    )
    parser.add_argument("--agent-python", default=sys.executable, help="Python used by the Adapter.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python used by the benchmark runner.")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per case (hard cap: 600).")
    parser.add_argument("--max-turns", type=int, default=30, help="Claude Code turn limit per case.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional backend CLI argument; repeat as needed.")
    parser.add_argument(
        "--protocol",
        action="append",
        choices=["legacy", "v0", "v1", "v2", "v3"],
        help="Run only this protocol. May be repeated; default is all cases.",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--keep-sandbox", action="store_true", help="Keep sanitized per-case temp directories for debugging.")
    parser.add_argument("--print-command", action="store_true", help="Print the expanded batch command without running it.")
    args = parser.parse_args()

    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds.")

    root = Path(args.repo_root).resolve()
    adapter = [
        args.agent_python,
        "../../../agents/coding_cli_safeact_agent.py",
        "--backend",
        args.backend,
        "--strategy",
        args.strategy,
        "--timeout",
        str(args.timeout),
        "--max-turns",
        str(args.max_turns),
    ]
    if args.model:
        adapter.extend(["--model", args.model])
    if args.profile:
        adapter.extend(["--profile", args.profile])
    if args.cfuse_config:
        adapter.extend(["--cfuse-config", args.cfuse_config])
    if args.cli_bin:
        adapter.extend(["--cli-bin", args.cli_bin])
    if args.model_catalog:
        adapter.extend(["--model-catalog", args.model_catalog])
    if args.keep_sandbox:
        adapter.append("--keep-sandbox")
    adapter.extend(f"--extra-arg={value}" for value in args.extra_arg)

    command = [
        args.python_bin,
        "scripts/run_safeact_agent_batch.py",
        "--repo-root",
        str(root),
        "--python-bin",
        args.python_bin,
        "--agent-cmd",
        shlex.join(adapter),
    ]
    for protocol in args.protocol or []:
        command.extend(["--protocol", protocol])
    for case_id in args.case_id:
        command.extend(["--case-id", case_id])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])

    print(shlex.join(command), flush=True)
    if args.print_command:
        return 0
    return subprocess.run(command, cwd=root, env=os.environ.copy(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
