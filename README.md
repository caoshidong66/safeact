# SafeActBench

SafeActBench evaluates evidence-grounded consequential actions by tool-using
agents. The release contains **656 cases**, **five protocols**, and **six
environments**.

## Dataset

| Protocol | Cases | Protocol | Cases |
|:--|--:|:--|--:|
| Legacy | 86 | V0 | 175 |
| V1 | 131 | V2 | 132 |
| V3 | 132 | **Total** | **656** |

| Environment | Cases |
|:--|--:|
| Customer policy QA | 112 |
| Healthcare operations | 98 |
| Legal and financial advice | 144 |
| Operations and code | 109 |
| Research assistance | 97 |
| Smart home | 96 |

## Repository layout

| Path | Contents |
|:--|:--|
| [`data/safeact/cases.json`](data/safeact/cases.json) | Unified benchmark cases |
| [`env/`](env/) | Case catalogs, evidence, tools, and executable worlds |
| [`templates/`](templates/) | Reusable environment templates |
| [`agents/`](agents/) | Agent and model-broker adapters |
| [`scripts/`](scripts/) | Batch runners and runtime support |

The large V3 observation template catalog is stored as gzip at
[`scripts/v3_public_observation_template_catalog.json.gz`](scripts/v3_public_observation_template_catalog.json.gz).
The runtime reads the compressed catalog directly.

## Quick start

Python 3.11 or newer is required.

```bash
python3 run_benchmark.py --list-only
python3 run_benchmark.py --simulate --case-id SAB-V2-001 --output-dir output/smoke
```

Run an interactive V2 or V3 episode through OpenRouter:

```bash
export OPENROUTER_API_KEY=your_key
python3 run_interactive.py --case-id SAB-V2-001 \
  --model deepseek/deepseek-v4-flash-0731 \
  --output-dir output/example
```

Supported model IDs are `deepseek/deepseek-v4-flash-0731`, `z-ai/glm-5.2`,
and `qwen/qwen3.8-flash`.
