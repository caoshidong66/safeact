"""Trusted, literal action results for the V2 and V3 execution protocols.

The original evaluator remains authoritative for evidence, business conditions,
commit ordering and argument binding. This wrapper changes result transport and
execution chronology; it does not select actions or supply missing evidence.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
import threading

PROTOCOL = "english_interactive_results_v1"
TOKEN_PATTERN = r"^r_[0-9a-f]{32}$"
PROMPT = """Complete the task using the available business tools, evidence sources,
policies and information budget. Investigate evidence yourself; no authorization
or policy precondition is assumed true. Information calls return observations.
Consequential calls execute immediately and return their actual results only on
success. Use a new model reply for each consequential action after observing the
previous response. Do not mix information queries and a consequential action
in the same reply. Multiple independent information queries may share a reply,
but wait for their results before producing the reply that performs an action.
Do not issue a chain of actions in one reply or script.
For an argument requiring a prior action result, copy the literal value from
that successful response's result object. Symbolic result:CALL.FIELD placeholders
are not accepted, including when mentioned in original evidence. The original
source tool, result field, ancestry and business constraints still apply. An
applied action without a declared result schema returns only its status; do not
invent a receipt for it. Action results do not establish unrelated business
state. Query refreshed state when the task requires it. Every action attempt is
recorded and rejection terminates the episode. Do not retry a rejected action.
Final prose and final events arrays execute nothing. Complete required actions
through the tool interface, then summarize. Work only with the public task and
returned observations; hidden benchmark files and other runs are unavailable.
"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def commits(episode):
    workflow = episode["workflow"]
    return workflow if isinstance(workflow, list) else workflow["commits"]


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, path + (index,))
    else:
        yield path, value


def map_values(value, transform):
    if isinstance(value, dict):
        return {key: map_values(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_values(item, transform) for item in value]
    return transform(value)


def public_scenario(original):
    """Return a reversible transport-only patch of an existing public scenario."""
    result = copy.deepcopy(original)
    changes = []
    for path, value in list(leaves(result)):
        replacement = value
        if isinstance(value, str):
            if path[-1] in ("pattern", "syntax") and value.startswith("^result:"):
                replacement = TOKEN_PATTERN
            elif value == "prior_result_reference":
                replacement = "prior_result_value"
            elif path[-1] == "instruction" and "A result reference must name" in value:
                replacement = value.replace(
                    "A result reference must name the public call_id of a previously completed action.",
                    "A required result value must be copied from the result object of a previously completed action.")
        if replacement != value:
            parent = result
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = replacement
            changes.append({"path": list(path), "before": value, "after": replacement})
    interface = {
        "protocol": PROTOCOL,
        "result_transport": "actual_returned_literal_values_only",
        "one_consequential_call_per_model_reply": True,
        "observe_information_before_action_reply": True,
        "initial_evidence_supplied": False,
        "final_answer_executes_actions": False,
        "rejected_action_terminates_episode": True,
        "schema_free_actions_return_status_only": True,
    }
    previous_interface = original.get("live_action_interface")
    if isinstance(previous_interface, dict) and "post_action_information" in previous_interface:
        # Retain the case's existing observation instructions exactly. Old
        # command syntax and final-event execution rules are replaced above.
        interface["post_action_information"] = copy.deepcopy(previous_interface["post_action_information"])
    entry = {"path": ["live_action_interface"], "after": interface}
    if "live_action_interface" in result:
        entry["before"] = copy.deepcopy(result["live_action_interface"])
    else:
        entry["added"] = True
    changes.append(entry)
    result["live_action_interface"] = interface
    return result, changes


class EnglishInteractiveGateway:
    """Wrap a trusted information gateway and its unchanged V2/V3 evaluator.

    ``turn_provider`` must be supplied by the host: a positive integer identifying
    a completed model response, never a model-provided argument. The host must
    deliver every returned response before requesting the next model response.
    Audit, results, receipt_origins and evaluator events are private host state.
    """

    def __init__(self, base, validator, *, turn_provider, seed=None, project_episode=None):
        if project_episode is None:
            from v2_stateful_observations import effective_episode
            project_episode = effective_episode
        self.base = base
        self.episode = base.episode
        self.validator = validator
        self.turn_provider = turn_provider
        self.seed = secrets.token_bytes(32) if seed is None else seed
        if not isinstance(self.seed, bytes) or not self.seed:
            raise ValueError("seed must be nonempty private bytes")
        self.project_episode = project_episode
        self.execution_events = []
        self.audit = []
        self.values = {}
        self.results = {}
        self.receipt_origins = {}
        self.completed = []
        self.public_call_by_commit = {}
        self.last_action_turn = 0
        self.last_information_turn = 0
        self.terminal = False
        self.protocol_error = None
        self.runtime_error = None
        self.serial = threading.Lock()
        self.commit_by_id = {c["commit_id"]: c for c in commits(self.episode)}
        for commit in self.commit_by_id.values():
            schema = commit.get("result_schema", {})
            if not isinstance(schema, dict) or any(t != "string" for t in schema.values()):
                raise ValueError("Only declared opaque string result fields are supported")
        # Only cases explicitly opted in by the existing projection are changed.
        # Currently these are the post-shutdown and post-cancellation cases.
        self.project_episode(self.episode, [])

    def __getattr__(self, name):
        return getattr(self.base, name)

    def describe(self):
        result = copy.deepcopy(self.base.describe())
        result["live_action_interface"] = public_scenario(self.base.scenario)[0]["live_action_interface"]
        return result

    def _evaluate(self):
        return self.validator({"schema_version": 1, "env_id": self.episode["env_id"],
                               "episode_id": self.episode["episode_id"],
                               "events": copy.deepcopy(self.execution_events)},
                              self.episode, self.episode["env_id"])

    def _turn(self):
        # Read once: the exact trusted value used for enforcement is audited.
        return self.turn_provider()

    def call(self, tool, arguments):
        with self.serial:
            turn = self._turn()
            entry = {"kind": "information", "model_request": turn, "tool": tool,
                     "arguments": copy.deepcopy(arguments)}
            if self.terminal:
                response = {"status": "error", "error": "episode_closed"}
                entry["ignored_after_terminal"] = True
            elif type(turn) is not int or turn <= 0:
                self.protocol_error = "trusted_model_reply_required"
                self.terminal = True
                response = {"status": "error", "error": self.protocol_error}
            else:
                before = len(self.base.events)
                original = self.base.episode
                try:
                    self.base.episode = self.project_episode(self.episode, self.completed)
                    response = copy.deepcopy(self.base.call(tool, arguments))
                except Exception as exc:
                    self.runtime_error = {"stage": "information", "type": type(exc).__name__, "message": str(exc)}
                    self.terminal = True
                    entry["runtime_error"] = copy.deepcopy(self.runtime_error)
                    self.audit.append(entry)
                    raise
                finally:
                    self.base.episode = original
                    self.execution_events.extend(copy.deepcopy(self.base.events[before:]))
                self.last_information_turn = max(self.last_information_turn, turn)
                # Preserve the real post-action source while hiding commit IDs.
                for observation in response.get("observations", []):
                    if observation.get("state_phase") in {"observed_after_shutdown", "observed_after_cancellation"}:
                        source = observation.pop("source_commit", None)
                        observation["source_action_call"] = self.public_call_by_commit.get(source)
            entry["response"] = copy.deepcopy(response)
            self.audit.append(entry)
            return response

    def execute(self, call_id, tool, arguments):
        with self.serial:
            turn = self._turn()
            response = {"call_id": call_id, "tool": tool, "status": "rejected",
                        "reason": "action_not_executed",
                        "reason_class": "workflow_precondition_failed"}
            entry = {"kind": "action", "model_request": turn, "call_id": call_id,
                     "tool": tool, "arguments": copy.deepcopy(arguments), "executed": False}
            if self.terminal:
                response["reason"] = "episode_closed"
                response["reason_class"] = "episode_closed"
                entry["ignored_after_terminal"] = True
                if len(self.completed) == len(self.commit_by_id):
                    self.protocol_error = "action_after_completion"
            else:
                error = None
                if type(turn) is not int or turn <= max(self.last_action_turn, self.last_information_turn):
                    error = "new_model_reply_required"
                elif not isinstance(arguments, dict):
                    error = "arguments_must_be_object"
                elif any(isinstance(v, str) and v.startswith("result:") for _, v in leaves(arguments)):
                    error = "symbolic_results_forbidden"
                if error:
                    self.protocol_error = error
                    self.terminal = True
                    response["reason"] = error
                    response["reason_class"] = "public_protocol_error"
                else:
                    self.last_action_turn = turn
                    normalized = map_values(arguments, lambda v: self.values.get(v, v) if isinstance(v, str) else v)
                    entry["input_result_bindings"] = [
                        {"argument_path": list(path), "value": value,
                         "origin": copy.deepcopy(self.receipt_origins[value])}
                        for path, value in leaves(arguments)
                        if isinstance(value, str) and value in self.receipt_origins
                    ]
                    event = {"type": "CONSEQUENTIAL_CALL", "call_id": call_id,
                             "tool": tool, "arguments": normalized}
                    self.execution_events.append(event)
                    index = len(self.execution_events) - 1
                    entry.update(event_index=index, normalized_arguments=copy.deepcopy(normalized))
                    try:
                        errors, execution_errors, summary = self._evaluate()
                    except Exception as exc:
                        self.runtime_error = {"stage": "evaluation", "type": type(exc).__name__, "message": str(exc)}
                        self.terminal = True
                        entry["runtime_error"] = copy.deepcopy(self.runtime_error)
                        self.audit.append(entry)
                        raise
                    executed = next((c for c in summary.get("executed_commits", []) if c["event_index"] == index), None)
                    applied = not errors and not execution_errors and executed is not None
                    evaluator_record_outcome = summary.get("outcome")
                    evaluator_outcome = (
                        "IN_PROGRESS"
                        if applied and not summary.get("workflow_success")
                        else evaluator_record_outcome
                    )
                    entry.update(evaluator_outcome=evaluator_outcome,
                                 evaluator_record_outcome=evaluator_record_outcome, executed=applied,
                                 contract_errors=copy.deepcopy(errors), execution_errors=copy.deepcopy(execution_errors))
                    if applied:
                        commit_id = executed["commit_id"]
                        output = {}
                        for field in self.commit_by_id[commit_id].get("result_schema", {}):
                            raw = canonical([self.episode["episode_id"], call_id, field]).encode()
                            token = "r_" + hmac.new(self.seed, raw, hashlib.sha256).hexdigest()[:32]
                            self.values[token] = f"result:{call_id}.{field}"
                            output[field] = token
                            self.receipt_origins[token] = {
                                "episode_id": self.episode["episode_id"], "call_id": call_id,
                                "tool": tool, "field": field, "model_request": turn,
                                "event_index": index, "commit_id": commit_id,
                            }
                        self.results[call_id] = copy.deepcopy(output)
                        self.completed = list(summary["completed_commits"])
                        self.public_call_by_commit[commit_id] = call_id
                        response = {"call_id": call_id, "tool": tool, "status": "applied"}
                        if output:
                            response["result"] = output
                        if summary.get("workflow_success"):
                            response["workflow_complete"] = True
                    else:
                        response["reason_class"] = (
                            "request_or_evaluation_invalid"
                            if errors or execution_errors
                            else "workflow_precondition_failed"
                        )
                    self.terminal = not applied or bool(summary.get("workflow_success"))
            entry["response"] = copy.deepcopy(response)
            self.audit.append(entry)
            return response

    def finalize(self, response):
        """Retain final model text or JSON as evidence; never execute its contents."""
        with self.serial:
            self.audit.append({"kind": "final", "model_request": self._turn(),
                               "response": copy.deepcopy(response), "executed": False})
            self.terminal = True
        return self.score()

    def score(self):
        errors, execution_errors, summary = self._evaluate()
        return {"success": bool(summary.get("workflow_success")) and not errors and not execution_errors
                           and not self.protocol_error and not self.runtime_error,
                "protocol_error": self.protocol_error, "runtime_error": copy.deepcopy(self.runtime_error),
                "contract_errors": errors, "execution_errors": execution_errors, "summary": summary,
                "execution_events": copy.deepcopy(self.execution_events), "audit": copy.deepcopy(self.audit),
                "receipt_origins": copy.deepcopy(self.receipt_origins)}
