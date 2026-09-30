# Contributing

SafeActBench is hash-addressed: changing a case, evidence record, schema, or
evaluation resource can change the meaning or identity of benchmark results.

## Reporting an issue

Please include:

- the case ID or repository path;
- the dataset version and Git commit;
- the observed problem and expected interpretation;
- the smallest relevant evidence or evaluator excerpt;
- whether the proposal changes semantics or only presentation.

Do not include credentials, private model transcripts, or unrelated production
data in an issue.

## Change classes

Presentation-only changes fix documentation without changing case or evidence
semantics. Semantic changes affect tasks, tools, evidence, expected outcomes,
schemas, or scoring and require a new dataset version.

For a semantic change, regenerate the case manifest and validation records, run
the deterministic validators and tests, and document compatibility in the pull
request. Never retag an existing release to point at changed benchmark data.

## Validation expectations

Before opening a pull request:

```bash
python3 verify_package.py
python3 scripts/check_runtime.py \
  --repo-root . \
  --output output/runtime_check.json \
  --cli-matrix --tool-backends
```

Contributions to code are accepted under Apache-2.0. Contributions to dataset
and documentation files are accepted under CC BY 4.0, as described in
`LICENSE`.
