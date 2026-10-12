"""Load the unchanged SafeAct V2 runtime from this checkout."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def case_row(case_id):
    dataset = json.loads((ROOT / "data/safeact/cases.json").read_text(encoding="utf-8"))
    row = next((row for row in dataset["cases"] if row["case_id"] == case_id), None)
    if row is None:
        raise ValueError(f"Unknown SafeAct case: {case_id}")
    if row["protocol"] != "v2":
        raise ValueError("This pilot supports V2 only; other protocols are not scored")
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
        scenario = run.exposed_linear_workflow_episode(episode)
        with tempfile.TemporaryDirectory(prefix=".ale-replay-", dir=ROOT) as directory:
            gateway = live.EnglishInteractiveGateway(
                core.ToolGateway("v2", scenario, episode, Path(directory)),
                evaluator.validate_linear_workflow_execution_record,
                turn_provider=lambda: turn[0],
                seed=seed,
            )
            yield gateway, run, live
    finally:
        sys.path[:] = old_path
