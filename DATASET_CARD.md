# SafeActBench dataset card

## Summary

SafeActBench is a benchmark for evidence-grounded action by tool-using agents.
Cases test whether an agent can locate the relevant records, reason about
authorization and prerequisites, and either stop safely or propose the correct
consequential action or workflow.

## Dataset identity

- Dataset ID: `safeact`
- Dataset version: `2026-08-31-v3`
- Public repository snapshot: September 30, 2026
- Language: English
- Cases: 656 unique canonical IDs
- Protocols: Legacy, V0, V1, V2, V3
- Environments: 6

Exact identity should be established from the Git commit and the per-case
hashes in `env/case_manifest.json`, not from a directory name alone.

## Composition

| Environment | Cases |
| --- | ---: |
| Customer Support and Policy | 112 |
| Healthcare Operations | 98 |
| Legal and Finance | 144 |
| Operations and Engineering | 109 |
| Research | 97 |
| Smart Home | 96 |

The repository contains complete case specifications, environment records,
materialized evidence, schemas, evaluation contracts, and pinned V3 resources.

## Intended uses

- Reproducible evaluation of evidence gathering and action selection.
- Analysis of safe stopping, prerequisite handling, and multi-step workflows.
- Benchmark auditing, error analysis, and evaluation-tool development.
- Research on tool-using agents in controlled synthetic business scenarios.

## Out-of-scope uses

- Deployment decisions in healthcare, legal, financial, home-control, or other
  real operational settings.
- Training or evaluating an agent while exposing reference annotations or
  expected outcomes in its prompt or retrieval corpus.
- Claiming blind generalization from results on this public, answer-bearing
  repository.
- Treating simulated actions or records as evidence about real people or
  organizations.

## Evaluation guidance

An agent should receive only a sanitized task plus observations returned by the
allowed information tools. Reference annotations, hidden evidence, evaluator
rules, and expected outcomes are evaluator-side data.

Report at least the dataset version, Git commit, protocol subset, model and
harness versions, prompting or scaffolding method, number of attempts, and
whether any reference-bearing files were accessible to the agent.

## Integrity

The repository includes:

- `env/case_manifest.json` for case discovery and per-case identity;
- `env/evaluation_contract.json` for the evaluation contract;
- `env/case_validation_report.json` as the retained validation record;
- deterministic validation and test entry points under `scripts/`.

The large pinned V3 observation catalog is stored as
`scripts/v3_public_observation_template_catalog.json.gz`. Runtime integrity
checks hash the original bytes while reading the compressed resource, so the
benchmark contract remains bound to the canonical catalog without requiring
Git LFS.

## Limitations

- Scenarios are controlled benchmark fixtures and do not reproduce the full
  ambiguity or organizational context of real operations.
- Public reference annotations make this repository unsuitable as a secret
  test set.
- Aggregate accuracy can hide protocol-specific failures; results should be
  broken down by protocol and environment.
- Passing deterministic integrity checks is not equivalent to independently
  validating every scenario's semantics.
