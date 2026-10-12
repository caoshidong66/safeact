"""Host transport for V0/V1 using their original information tools and graders.

No hidden gold decision is returned by this transport. V0 stops are supplied by
the agent; the unchanged evaluator independently derives the correct outcome.
"""

from __future__ import annotations

import copy
import re


class SingleProtocolGateway:
    def __init__(self, base, validator, *, turn_provider):
        self.base = base
        self.validator = validator
        self.turn_provider = turn_provider
        self.execution_events = []
        self.terminal = None
        self.closed = False
        self.protocol_error = None
        self.last_turn = 0

    def _open(self):
        turn = self.turn_provider()
        if self.closed or type(turn) is not int or turn <= 0:
            self.protocol_error = (
                "episode_closed" if self.closed else "trusted_model_reply_required"
            )
            self.closed = True
            return False
        self.last_turn = max(self.last_turn, turn)
        return True

    def describe(self):
        return self.base.describe()

    def call(self, tool, arguments):
        if not self._open():
            return {"status": "error", "error": self.protocol_error}
        before = len(self.base.events)
        response = copy.deepcopy(self.base.call(tool, arguments))
        self.execution_events.extend(copy.deepcopy(self.base.events[before:]))
        return response

    def _evaluate(self):
        payload = {
            "schema_version": 1,
            "env_id": self.base.episode["env_id"],
            "episode_id": self.base.episode["episode_id"],
            "events": copy.deepcopy(self.execution_events),
        }
        if self.base.protocol == "v0":
            payload["terminal"] = copy.deepcopy(self.terminal or {})
        return self.validator(payload, self.base.episode, self.base.episode["env_id"])

    def execute(self, call_id, tool, arguments):
        response = {"call_id": call_id, "tool": tool, "status": "rejected"}
        previous = self.last_turn
        if not self._open():
            return {**response, "reason": self.protocol_error}
        if (
            self.turn_provider() <= previous
            or not isinstance(call_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", call_id)
        ):
            self.protocol_error = "invalid_action_request"
        self.execution_events.append(
            {
                "type": "CONSEQUENTIAL_CALL",
                "call_id": call_id,
                "tool": tool,
                "arguments": copy.deepcopy(arguments),
            }
        )
        errors, execution_errors, summary = self._evaluate()
        self.closed = True
        if (
            self.base.protocol == "v1"
            and summary.get("action_executed")
            and not errors
            and not execution_errors
            and not self.protocol_error
        ):
            return {
                "call_id": call_id,
                "tool": tool,
                "status": "applied",
                "workflow_complete": True,
            }
        return {**response, "reason": "action_not_executed"}

    def stop(self, terminal):
        if not self._open():
            return
        if (
            self.base.protocol != "v0"
            or set(terminal) != {"type", "outcome"}
            or terminal.get("type") != "NO_ACTION"
            or terminal.get("outcome") not in {"BLOCK", "DEFER"}
        ):
            self.protocol_error = "invalid_no_action_terminal"
        self.terminal = copy.deepcopy(terminal)
        self.closed = True

    def score(self):
        errors, execution_errors, summary = self._evaluate()
        success_key = (
            "no_action_success" if self.base.protocol == "v0" else "safe_commit_success"
        )
        return {
            "success": bool(summary.get(success_key))
            and not errors
            and not execution_errors
            and not self.protocol_error,
            "protocol_error": self.protocol_error,
            "contract_errors": errors,
            "execution_errors": execution_errors,
            "summary": summary,
            "execution_events": copy.deepcopy(self.execution_events),
            "terminal": copy.deepcopy(self.terminal),
        }
