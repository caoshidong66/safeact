# SafeActBench

SafeActBench evaluates whether tool-using agents gather sufficient evidence
before taking consequential actions. The repository contains **656 canonical
cases** across five protocols and six business environments, together with the
runtime, deterministic evaluator, schemas, and offline validation tools needed
to inspect and evaluate them.

## Clone and validate

```bash
git clone https://github.com/caoshidong66/safeact.git
cd safeact
python3 verify_package.py
python3 run_benchmark.py --list-only
python3 run_benchmark.py --simulate --case-id SAB-V1-001 --output-dir output/smoke
```

The core project uses the Python standard library and supports Python 3.11 or
newer. Individual model backends may require their own CLI or API setup.

## Dataset at a glance

| Protocol | Cases | Task shape |
| --- | ---: | --- |
| Legacy | 86 | Decide whether a fixed candidate action is supported |
| V0 | 175 | Gather evidence and explicitly decline an unsupported action |
| V1 | 131 | Gather evidence and complete one consequential action |
| V2 | 132 | Complete a linear multi-action workflow |
| V3 | 132 | Complete a dependency-structured workflow |

The six environments are Customer Support and Policy, Healthcare Operations,
Legal and Finance, Operations and Engineering, Research, and Smart Home. See
[DATASET_CARD.md](DATASET_CARD.md) for intended use and limitations.

## Repository layout

```text
agents/          Agent adapters and benchmark runtime components
data/safeact/    Unified canonical dataset export
env/             Cases, evidence, schemas, worlds, and evaluation contract
scripts/         Runtime checks, transports, and pinned V3 resources
templates/       Immutable workspace templates
cases/           Offline HTML review pages
docs/            Running and project documentation
```

The primary machine-readable dataset is `data/safeact/cases.json`. Case
discovery and identity are defined by `env/case_manifest.json`. See
[docs/RUNNING.md](docs/RUNNING.md) for model integration and runtime details.

One large pinned V3 catalog is stored as a gzip-compressed JSON resource. The
runtime validates its original uncompressed bytes directly; no setup step is
required. To materialize a plain JSON copy for inspection, run
`python3 scripts/materialize_v3_catalog.py`.

## Important evaluation boundary

This repository contains supporting evidence, reference annotations, and
expected outcomes. Do not place the repository or complete dataset in an
evaluated agent's context. The runner must expose only the sanitized task and
the observations returned by permitted tools.

Because reference answers are public, scores on this repository should be
reported as reproducible public-set results, not blind leaderboard results. A
leakage-resistant leaderboard requires a separately maintained private test set
or newly generated held-out cases.

## Licensing

- Dataset files and documentation are licensed under
  [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
- Source code is licensed under
  [Apache License 2.0](LICENSES/Apache-2.0.txt).

See [LICENSE](LICENSE) for the exact file-scope rules.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).
