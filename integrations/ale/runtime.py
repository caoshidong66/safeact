"""Load the unchanged SafeAct V0–V3 runtimes from this checkout."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_PROTOCOLS = frozenset({"v0", "v1", "v2", "v3"})


class PlatformUnavailable(RuntimeError):
    """The original runtime requires unavailable secure filesystem facilities."""


def case_row(case_id):
    dataset = json.loads((ROOT / "data/safeact/cases.json").read_text(encoding="utf-8"))
    row = next((row for row in dataset["cases"] if row["case_id"] == case_id), None)
    if row is None:
        raise ValueError(f"Unknown SafeAct case: {case_id}")
    if row["protocol"] not in SUPPORTED_PROTOCOLS:
        raise ValueError("Only V0–V3 execution protocols are supported")
    return row


@contextmanager
def runtime(row, turn, *, seed):
    """Yield a private gateway; its temporary workspace never reaches the agent."""
    old_path = sys.path[:]
    try:
        for relative in ("scripts", "agents", "env/tools"):
            sys.path.insert(0, str(ROOT / relative))
        names = (
            "run_agent",
            "evaluate_workspace",
            "coding_cli_safeact_agent",
            "english_interactive_gateway",
        )
        run, evaluator, core, live = [importlib.import_module(name) for name in names]
        for module in (run, evaluator, core, live):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise RuntimeError(
                    f"Runtime import escaped this checkout: {module.__name__}"
                )
        episode = copy.deepcopy(row["case_spec"])
        protocol = row["protocol"]
        expose = {
            "v0": run.exposed_no_action_episode,
            "v1": run.exposed_state_action_episode,
            "v2": run.exposed_linear_workflow_episode,
            "v3": run.exposed_multi_step_episode,
        }[protocol]
        try:
            scenario = expose(episode)
        except ValueError as exc:
            if "cannot be opened safely on this platform" in str(exc):
                raise PlatformUnavailable(str(exc)) from exc
            raise
        with tempfile.TemporaryDirectory(prefix=".ale-replay-", dir=ROOT) as directory:
            try:
                base = core.ToolGateway(protocol, scenario, episode, Path(directory))
            except ValueError as exc:
                if "cannot be opened safely on this platform" in str(exc):
                    raise PlatformUnavailable(str(exc)) from exc
                raise
            if protocol in {"v2", "v3"}:
                validator = (
                    evaluator.validate_linear_workflow_execution_record
                    if protocol == "v2"
                    else evaluator.validate_multi_step_execution_record
                )
                gateway = live.EnglishInteractiveGateway(
                    base,
                    validator,
                    turn_provider=lambda: turn[0],
                    seed=seed,
                )
            else:
                from .single import SingleProtocolGateway

                validator = (
                    evaluator.validate_no_action_execution_record
                    if protocol == "v0"
                    else evaluator.validate_single_commit_execution_record
                )
                gateway = SingleProtocolGateway(
                    base, validator, turn_provider=lambda: turn[0]
                )
            yield gateway, run, live
    finally:
        sys.path[:] = old_path
