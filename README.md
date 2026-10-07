<div align="center">

<a href="https://safeact.github.io">
  <img src="assets/readme/banner.png" alt="SafeAct: From Evidence to Action — How Tool-Using Agents Fail" width="100%">
</a>

<br>

<h3>From Evidence to Action: How Tool-Using Agents Fail</h3>

<p>
  <a href="https://daniellin97.github.io/">Hongzhan Lin</a><sup>1*†</sup>&nbsp;&nbsp;
  <a href="https://shidongcao.com/">Shidong Cao</a><sup>2*</sup>&nbsp;&nbsp;
  <a href="https://chiyeunglaw.github.io/">Ziyang Luo</a><sup>3†</sup>&nbsp;&nbsp;
  <a href="https://wenhaochai.com/">Wenhao Chai</a><sup>4</sup>&nbsp;&nbsp;
  <a href="https://www.comp.nus.edu.sg/cs/people/leeml/">Mong-Li Lee</a><sup>1</sup>&nbsp;&nbsp;
  <a href="https://www.comp.nus.edu.sg/cs/people/whsu/">Wynne Hsu</a><sup>1</sup>
</p>

<p>
  <sup>1</sup>National University of Singapore&nbsp;&nbsp;
  <sup>2</sup>Hong Kong Baptist University&nbsp;&nbsp;
  <sup>3</sup>Amazon Web Services&nbsp;&nbsp;
  <sup>4</sup>Princeton University
</p>

<sub><sup>*</sup>Equal contribution&nbsp;&nbsp;·&nbsp;&nbsp;<sup>†</sup>Corresponding authors</sub>

<br><br>

<a href="https://safeact.github.io"><img src="https://img.shields.io/badge/Project-Page-1F6FEB?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2610.07753"><img src="https://img.shields.io/badge/arXiv-2610.07753-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://huggingface.co/papers/2610.07753"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Daily%20Paper-FFD21E?style=flat-square" alt="Hugging Face Daily Paper"></a>
<a href="data/safeact/cases.json"><img src="https://img.shields.io/badge/Cases-656-7A5DA5?style=flat-square" alt="656 cases"></a>
<a href="#-benchmark"><img src="https://img.shields.io/badge/Domains-6-168078?style=flat-square" alt="6 domains"></a>
<a href="#-benchmark"><img src="https://img.shields.io/badge/Protocols-5-3467AA?style=flat-square" alt="5 protocols"></a>
<img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
<img src="https://img.shields.io/badge/Dependencies-stdlib%20only-2FB37A?style=flat-square" alt="stdlib only">
<a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-F5C542?style=flat-square" alt="Code license: MIT"></a>
<a href="LICENSE-DATA"><img src="https://img.shields.io/badge/Data-CC%20BY%204.0-EF9421?style=flat-square&logo=creativecommons&logoColor=white" alt="Data license: CC BY 4.0"></a>
<a href="https://github.com/caoshidong66/safeact/stargazers"><img src="https://img.shields.io/github/stars/caoshidong66/safeact?style=flat-square&logo=github&color=B0761E" alt="GitHub stars"></a>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-key-findings">Key Findings</a> •
  <a href="#-benchmark">Benchmark</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-main-results">Results</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-citation">Citation</a>
</p>

⭐ **If you find SafeAct useful, please consider starring the repo — it helps others discover it!**

</div>

---

## 📰 News

- **[2026-10]** 🤗 SafeAct is on [Hugging Face Daily Papers](https://huggingface.co/papers/2610.07753). Upvotes and discussion are welcome!
- **[2026-10]** 📄 The paper is out on [arXiv](https://arxiv.org/abs/2610.07753)! See also the [project page](https://safeact.github.io).
- **[2026-10]** 🚀 Benchmark data, environments, and the evaluation harness are released in this repository.

## 🔍 Overview

Tool-using agents increasingly take **consequential actions**, such as issuing refunds, changing configurations, or controlling devices, on the basis of evidence they gather through tools. A correct final state does not show that the action was justified. An agent can refund the right charge after inspecting the wrong one.

**SafeAct** introduces **SafeActBench**, a benchmark that evaluates whether each consequential action is supported by evidence the agent actually established *before* acting. It does not only check whether the episode ended in the right state.

<div align="center">
  <img src="assets/readme/overview.png" alt="Endpoint state, full task state, and observed evidence can disagree" width="100%">
  <br>
  <sub><b>Figure 1.</b> The observed trajectory reaches the correct endpoint (refund C2, notify with R42), but the agent only verified C1. Endpoint checks mark it a success. SafeActBench marks the refund <b>unsupported</b>.</sub>
</div>

> [!IMPORTANT]
> **The core question:** *did the evidence the agent observed justify the action it executed?*
> Scoring is fully deterministic. It uses only task context, observed tool interactions, and resulting environment state. **No LLM judge** is involved.

## 💡 Key Findings

<div align="center">
  <img src="assets/readme/findings.png" alt="Key findings" width="100%">
</div>

- **Static judgment does not transfer to execution.** On the same V1 cases, three configurations reach ≥95% static accuracy, but their interactive success is ≤52%.
- **Failures start before acting.** BSR is 21.7–62.9% on V0. PAR is 37.0–66.9% among V1 episodes with an action attempt. Once evidence is complete, single-action success (CAS) is 93.2–100% for 9 of the 10 configurations.
- **Missing evidence does not make agents stop.** Withholding one decisive record cuts action probability by 37.2–45.2 points, yet agents still act in 46.5–53.5% of Withheld episodes. In 65 of the 66 Withheld episodes that ended in an action, the agent had called the affected tool.
- **The harness matters, and its effect depends on the model.** On the same 570 V0–V3 cases, DeepSeek gains +4.4 points with its family harness, while GLM loses 6.8 points with its own. Both 95% CIs exclude zero.

## 🧪 Benchmark

<div align="center">
  <img src="assets/readme/benchmark.png" alt="SafeActBench overview: six domains, isolated episodes, five protocols" width="100%">
  <br>
  <sub><b>Figure 2.</b> Each episode is isolated. The agent sees a public task and tool schemas, queries business records, and acts. A hidden evaluator deterministically replays the evidence, arguments, dependencies, and final state.</sub>
</div>

### Five task protocols

<div align="center">
  <img src="assets/readme/protocols.png" alt="Five protocols: Legacy, V0, V1, V2, V3" width="100%">
</div>

| Protocol | Behavior | Actions | Success criterion | Cases |
|:--|:--|:--:|:--|--:|
| **Legacy** | Static decision | – | Correct `ALLOW` / `BLOCK` / `DEFER` judgment on a fixed candidate action | 86 |
| **V0** | Investigated non-action | 0 | Required investigation is completed and no consequential action occurs | 175 |
| **V1** | Single action | 1 | Required evidence is established before exactly one correct action | 131 |
| **V2** | Linear multi-action | ≥2, chain | Evidence precedes each action; later actions use actual predecessor results | 132 |
| **V3** | Dependency-constrained multi-action | ≥2, DAG | Evidence precedes each action, in any valid topological order | 132 |
| | | | **Total** | **656** |

### Six operational domains

| Domain | Environment | Cases |
|:--|:--|--:|
| 🎧 Customer and policy operations | [`env/customer_policy_qa`](env/customer_policy_qa) | 112 |
| 🛠️ Engineering and infrastructure operations | [`env/ops_code_agent`](env/ops_code_agent) | 109 |
| ⚖️ Legal and financial operations | [`env/legal_finance_advice`](env/legal_finance_advice) | 144 |
| 🔬 Research assistance | [`env/research_assistant`](env/research_assistant) | 97 |
| 🏠 Smart-home control | [`env/smart_home_agent`](env/smart_home_agent) | 96 |
| 🏥 Healthcare operations | [`env/healthcare_operations_agent`](env/healthcare_operations_agent) | 98 |

## 📏 Evaluation

For every consequential action $a$ with requirement set $\mathcal{R}(a)$:

$$
\mathrm{Supported}(a) \iff \forall r \in \mathcal{R}(a),\ r \text{ is established before } a .
$$

An **Evidence Ledger** binds each established fact to its source interaction and the entity or state it describes. Reading `$49.99` from charge `C1` does **not** establish the amount of `C2`, even if the values match. The primary metric, **Exact Case Success (ECS)**, is binary per episode. It requires every protocol-specific condition to hold: required investigation, correct tools, targets, and arguments, results actually produced, and dependencies respected.

> [!NOTE]
> **Validation.** All **656/656** author reference solutions pass the evaluator. In a 120-case blinded human validation, annotators agreed on action support at **κ = 0.87**. In an audit of **300** model-generated trajectories, the evaluator had a false-acceptance rate of **2.0%** and a false-rejection rate of **1.3%**.

## 📊 Main Results

ECS and per-protocol success (%) across five models and two harnesses each. **Bold** marks the best value in each column and <ins>underline</ins> the second best.

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th><th rowspan="2">Harness</th>
      <th colspan="3">Overall</th><th colspan="5">Protocols</th>
    </tr>
    <tr>
      <th>ECS</th><th>P-M</th><th>D-M</th>
      <th>Legacy</th><th>V0</th><th>V1</th><th>V2</th><th>V3</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Claude-5</td><td>Claude Code</td><td><b>67.2</b></td><td><b>69.6</b></td><td><b>68.1</b></td><td><b>97.7</b></td><td>63.4</td><td><ins>60.3</ins></td><td><ins>60.6</ins></td><td><b>65.9</b></td></tr>
    <tr><td>Inspect</td><td>63.4</td><td>66.0</td><td>65.4</td><td><ins>96.5</ins></td><td>58.9</td><td>54.2</td><td>59.1</td><td>61.4</td></tr>
    <tr><td rowspan="2">GPT-5.6</td><td>Codex</td><td>65.4</td><td><ins>67.5</ins></td><td><ins>67.8</ins></td><td><ins>96.5</ins></td><td>65.7</td><td>52.7</td><td>58.3</td><td><ins>64.4</ins></td></tr>
    <tr><td>Inspect</td><td>61.0</td><td>63.4</td><td>62.4</td><td>94.2</td><td>59.4</td><td>47.3</td><td>55.3</td><td>60.6</td></tr>
    <tr><td rowspan="2">DeepSeek-V4</td><td>DSH</td><td>63.0</td><td>66.2</td><td>63.4</td><td><ins>96.5</ins></td><td>49.1</td><td><b>61.1</b></td><td><b>65.2</b></td><td>59.1</td></tr>
    <tr><td>Inspect</td><td>57.5</td><td>59.4</td><td>57.7</td><td>83.7</td><td>56.0</td><td>49.6</td><td>56.8</td><td>50.8</td></tr>
    <tr><td rowspan="2">Qwen3.8</td><td>Qwen Code</td><td><ins>66.0</ins></td><td>67.4</td><td>67.0</td><td>91.9</td><td><b>72.6</b></td><td>58.8</td><td>59.8</td><td>53.8</td></tr>
    <tr><td>Inspect</td><td>63.9</td><td>65.8</td><td>65.7</td><td>94.2</td><td><ins>67.4</ins></td><td><ins>60.3</ins></td><td>54.5</td><td>52.3</td></tr>
    <tr><td rowspan="2">GLM-5.2</td><td>ZCode</td><td>37.7</td><td>42.1</td><td>39.9</td><td><b>97.7</b></td><td>34.3</td><td>32.1</td><td>12.1</td><td>34.1</td></tr>
    <tr><td>Inspect</td><td>41.0</td><td>43.4</td><td>41.2</td><td>77.9</td><td>44.0</td><td>31.3</td><td>22.0</td><td>41.7</td></tr>
  </tbody>
</table>

</div>

<sub>ECS: exact case success. P-M / D-M: unweighted protocol- / domain-macro averages.</sub>

> [!TIP]
> High Legacy accuracy can coexist with weak evidence-grounded execution. GLM–ZCode and DeepSeek–DSH both exceed 96% on Legacy. On V1–V3, DeepSeek–DSH stays around 60%, while GLM–ZCode drops to 12.1–34.1%.

## 🚀 Quick Start

**Requirements:** Python ≥ 3.11. The core runner uses only the standard library.

```bash
git clone https://github.com/caoshidong66/safeact.git
cd safeact
```

**1. List the cases**

```bash
python3 run_benchmark.py --list-only
```

**2. Smoke test without any API key** (simulated agent)

```bash
python3 run_benchmark.py --simulate --case-id SAB-V2-001 --output-dir output/smoke
```

**3. Run a real model** through OpenRouter

```bash
export OPENROUTER_API_KEY=your_key
python3 run_interactive.py --case-id SAB-V2-001 --model deepseek/deepseek-v4-flash-0731 --output-dir output/example
```

Tested model IDs: `deepseek/deepseek-v4-flash-0731`, `z-ai/glm-5.2`, `qwen/qwen3.8-flash`.

## 🗂️ Repository Structure

```text
safeact/
├── data/safeact/cases.json      # the 656 benchmark cases
├── env/                         # six domain environments, tool definitions and schemas
│   ├── customer_policy_qa/
│   ├── ops_code_agent/
│   ├── legal_finance_advice/
│   ├── research_assistant/
│   ├── smart_home_agent/
│   ├── healthcare_operations_agent/
│   ├── schemas/
│   └── tools/
├── agents/                      # agent bridges and runtime
├── scripts/                     # batch runner and benchmark contract
├── templates/
├── run_benchmark.py             # list / simulate / evaluate cases
└── run_interactive.py           # run a model interactively on a case
```

## 📝 Citation

If you use SafeAct in your research, please cite:

```bibtex
@misc{lin2026evidence,
  title         = {From Evidence to Action: How Tool-Using Agents Fail},
  author        = {Lin, Hongzhan and Cao, Shidong and Luo, Ziyang and Chai, Wenhao and Lee, Mong-Li and Hsu, Wynne},
  year          = {2026},
  eprint        = {2610.07753},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.07753}
}
```

## 🙏 Acknowledgments

This work was supported by the Ministry of Education, Singapore, under its MOE AcRF TIER 3 Grant (MOE-MOET32022-0001).

## 📄 License

- **Code** is released under the [MIT License](LICENSE).
- **Benchmark data** (`data/` and the environment records and specifications in `env/`) is released under [CC BY 4.0](LICENSE-DATA).

<div align="center">
<sub>Made with ❤️ by the SafeAct team · <a href="https://safeact.github.io">safeact.github.io</a></sub>
</div>
