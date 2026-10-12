# SafeActBench to ALE trajectory pilot

This optional integration exports host-recorded SafeAct V2 tool interactions to
the ALE-v1.0 schema and scores them with the **unchanged SafeAct evaluator**.
It writes an independent evidence sidecar. It does not modify ALE's outcome
score or register new tasks in ALE.

## Quick start

Run from the SafeAct repository root with Python 3.11 or newer. The adapter and
pilot use only the standard library and this repository's original runtime.

```bash
python -m integrations.ale.pilot --output-dir work/ale-pilot
python -m integrations.ale.adapter work/ale-pilot/SAB-V2-003.trajectory.json \
  --case-id SAB-V2-003 --output work/ale-pilot/refund.evidence.json
```

The five reference fixtures cover archive retention, duplicate refunds,
production release, contract filing, and research package publication. They
execute authored reference traces through the original gateway. They are
schema and evaluator checks, **not model results or ALE VM runs**. Their
`endpoint_score` is null because no ALE endpoint grader was run.

An existing `scripts/run_english_interactive_rollout.py` run can be exported:

```bash
python -m integrations.ale.export /path/to/recorded-rollout \
  --agent-name safeact-openrouter --model YOUR_RECORDED_MODEL \
  --output work/trajectory.json
python -m integrations.ale.adapter work/trajectory.json \
  --case-id SAB-V2-003 --output work/evidence.json
```

The source directory must contain the runner's `messages.json` and
`case_metadata.json`. The exporter checks its case specification digest against
this checkout. Use the actual recorded agent and model identity. An export
does not turn a SafeAct run into a leaderboard-eligible ALE run.

## Recording and scoring contract

- Use a trusted host gateway exposing `information_call`, `execute_action`, and
  `list_tools`. These are the existing English interactive tool envelopes.
- Record every assistant tool call and the exact JSON response delivered to the
  agent. Match observations with ALE `tool_call_id`; preserve recorded step order.
- Carry `extra.safeact_case_id` and `extra.safeact_case_spec_sha256`. Keep case
  specifications, evaluator, replay workspace, and score reports outside the
  agent sandbox.
- Information earns credit when its response appears in the trace, not when the
  query is emitted. An action waits until prior responses have been observed.
- Replay uses the original tool gateway to recover provenance and evidence.
  Recorded observations must match its authoritative responses. Literal receipts
  are bound to the producing action and result field only after that response
  is observed. Replay substitutes equivalent receipt tokens, never future or
  symbolic receipts.
- A valid incomplete or rejected workflow is `scored` with evidence score 0.
  Unsupported surfaces, incomplete logs, and mismatched observations are
  `unscorable` with a null evidence score and a diagnostic. Report their coverage
  separately; do not silently count them as passes or drop them from denominators.
- A stored ALE `final_metrics.reward` is copied to `endpoint_score`; it is not
  recomputed, combined with evidence score, or inferred from SafeAct success.

**Trust boundary:** the sidecar consumes host-collected logs. It does not
authenticate agent-editable JSON, arbitrary shell stdout, screenshots, or logs
fabricated outside the trusted runner. It replays the original enforcing SafeAct
gateway, so it cannot establish historical success of an unsupported action in
an unconstrained ALE application. Such a workflow needs a task-specific observer
and an outcome grader.

## Current scope

V2 only. V0/V1 and V3 need separate protocol adapters. V3's original attested file
reader requires Linux safe-open facilities; it was not relaxed for Windows.
Generic GUI/CLI traces, continued chunks, and subagent traces need additional
instrumentation or a verified causal merge before they can be scored.

ALE currently grades before parsing/finalizing its normalized trajectory. Keep
this prototype as a post-run command until ALE maintainers agree on an optional
host-side postprocessor. No orchestrator or leaderboard changes are included.

## Validation

```bash
python -m pytest tests/test_ale_adapter.py -q
python -m ruff check integrations/ale tests/test_ale_adapter.py
```

Installing ALE's checkout in the test environment enables a check against its
actual `Trajectory` Pydantic model. Without that optional checkout, that one test
is skipped; the integration does not vendor a replacement ALE schema.

Before proposing a benchmark task, run a real sandbox/model pilot and have
domain experts validate the workflow and alternative evidence paths. These
existing SafeAct cases are controlled fixtures, not newly collected professional
projects. Acceptance into ALE's task corpus remains a separate review.

## Attribution

Hongzhan Lin, Shidong Cao, Ziyang Luo, Wenhao Chai, Mong-Li Lee, and Wynne Hsu.
*From Evidence to Action: How Tool-Using Agents Fail* (2026).
https://arxiv.org/abs/2610.07753

SafeAct source code: MIT, see the repository's `LICENSE`. SafeAct case data:
CC BY 4.0, see `LICENSE-DATA`. Existing cases and evaluator are not modified.
ALE's framework is Apache-2.0; the adapter refers to its public schema without
copying its source or redistributing its task data.
