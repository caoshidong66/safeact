# Project structure

```text
safeact/
├── README.md
├── DATASET_CARD.md
├── LICENSE
├── LICENSES/
├── CITATION.cff
├── agents/
├── data/
│   └── safeact/
│       └── cases.json
├── env/
│   ├── case_manifest.json
│   ├── evaluation_contract.json
│   ├── schemas/
│   ├── <environment>/
│   │   ├── materialized_evidence/
│   │   └── world/
│   └── v3_episode_query_shards/
├── scripts/
│   ├── run_safeact_agent_batch.py
│   ├── check_runtime.py
│   └── *.json
├── templates/
├── cases/
├── run_benchmark.py
└── verify_package.py
```

## Stable machine-readable entry points

- `data/safeact/cases.json` is the unified canonical data export.
- `env/case_manifest.json` is the case discovery and identity index.
- `env/evaluation_contract.json` describes the evaluation contract.
- `env/schemas/` contains machine-readable schemas.

## Large pinned resource

`scripts/v3_public_observation_template_catalog.json.gz` is a pinned benchmark
resource whose uncompressed form exceeds GitHub's ordinary single-blob limit.
The runtime transparently hashes its original bytes from gzip, preserving the
canonical contract digest. Run `python3 scripts/materialize_v3_catalog.py` only
if a plain JSON inspection copy is needed.

## Generated and private outputs

`output/`, caches, local virtual environments, credentials, and historical
experiment artifacts are intentionally excluded from the public source
repository.
