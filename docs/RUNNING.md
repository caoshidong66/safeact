# Running SafeActBench

## Requirements

Use Python 3.11 or newer. The reference simulator, evaluator, and package checks
require no third-party Python packages. Linux is required for the bundled model
CLI isolation route. It additionally uses `bubblewrap` (`bwrap`), `timeout`,
`prlimit`, and a compatible Codex CLI. Bubblewrap must be installed as a trusted
system binary, and the host must permit its namespaces. The isolated environment
also invokes `/usr/bin/python3`.

The model CLI route expects support for the isolation/configuration switches
used by the adapter, including `--ignore-user-config` and `--ignore-rules`.
Provider credentials and CLI installations are supplied by the operator.

## Select cases

```bash
python3 run_benchmark.py --list-only --protocol v2
python3 run_benchmark.py --simulate --case-id SAB-V2-001 --output-dir output/one
python3 run_benchmark.py --simulate --protocol v0 --limit 10 --output-dir output/v0
```

`--protocol` and `--case-id` can be repeated. Filters select from the same
656-case manifest. The built-in batch runner is sequential. Each environment
has a mutable runtime workspace, so parallel workers need separate extracted
copies of the package.

Resume a run using matching case, runtime, and artifact fingerprints:

```bash
python3 run_benchmark.py --simulate --output-dir output/reference_replay --resume
```

## Model execution through the included transport

The DeepSeek entrypoint starts the trusted model proxy and configures the Codex
adapter. Inspect the command without credentials or a model request:

```bash
python3 scripts/run_deepseek_safeact_benchmark.py \
  --repo-root . \
  --case-id SAB-V0-001 \
  --user-id company-demo \
  --model deepseek-v4-flash \
  --output-dir output/model_demo \
  --print-command
```

To execute, set `DEEPSEEK_API_KEY`, provide the compatible CLI path, and omit
`--print-command`:

```bash
python3 scripts/run_deepseek_safeact_benchmark.py \
  --repo-root . \
  --case-id SAB-V0-001 \
  --user-id company-demo \
  --model deepseek-v4-flash \
  --codex-bin /path/to/codex \
  --output-dir output/model_demo
```

The model identifier above is supported by this runner's bundled configuration.
Select a model and endpoint enabled for your provider account; `--help` lists
the implemented options. `--base-url`, `--api-key-env`, and `--upstream-model`
configure the transport. The default inference strategy is `baseline`.

The lower-level `scripts/run_coding_agent_benchmark.py` exposes Codex and Claude
adapter entrypoints. These require a compatible, separately configured trusted
Unix-socket model broker. CLI authentication alone does not configure that
broker. The DeepSeek entrypoint above performs the broker setup for its own
transport route.

## Integrate another agent

The generic runner accepts an adapter command:

```bash
python3 run_benchmark.py \
  --agent-cmd 'python3 ../../../my_agent.py' \
  --case-id SAB-V1-001 \
  --output-dir output/custom_agent
```

Place `my_agent.py` in the package root. The command runs in
`env/<environment>/workspace/`, where `scenario.json` is the sanitized task,
`tools/` contains the permitted tool entrypoints, and `output/` receives the
protocol submission. The process receives `SAFEACT_PROTOCOL`,
`SAFEACT_CASE_ID`, `SAFEACT_CATALOG`, and `SAFEACT_ENV_ID`.

Implement the submission contract for the selected protocol, as used by
`env/tools/run_agent.py` and `env/tools/evaluate_workspace.py`; the supplied
model adapter provides an integration example. An external adapter is trusted
harness code and is responsible for exposing only the permitted task and tools
to its model.

## Results

The batch runner writes `summary.json` and case records under the selected
output directory. Inspect protocol-level metrics and infrastructure failures,
then the case evaluation records and traces. Per-case files include:

```text
logs/
raw_outputs/
normalized_results/
records/
traces/
scenarios/
```

Keep protocol labels and package/case hashes with published results. Reference
simulation success is an environment check, not a model accuracy result.

## Offline validation

```bash
python3 verify_package.py
python3 scripts/check_runtime.py \
  --repo-root . \
  --output output/runtime_check.json \
  --cli-matrix --tool-backends
```

The runtime check covers all reference cases, sanitized task visibility,
negative controls, tool access, and representative command-line paths. It does
not request model inference. `VALIDATION.json` records the checks completed for
this packaged distribution, including any checks that were not performed.
