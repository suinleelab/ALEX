# [ALEX: Automatic Language Explanations for Interpreting Treatment Effects via Multi-Agents](https://www.medrxiv.org/content/10.64898/2026.04.23.26351510v1.full)

ALEX is a multi-agent system that turns SHAP attributions of CATE (Conditional Average Treatment Effect) models on clinical-trial cohorts into structured natural-language explanations, then scores and validates those explanations against the published literature.

---

## System requirements

### Software

| Package | Version tested | Notes |
|---------|----------------|-------|
| Python | 3.9 / 3.10 | |
| `numpy` | 1.24.x | |
| `torch` | 2.0.0 | |
| `captum` | 0.6.0 | |
| `pydantic` | 2.10.6 | |
| `openai` | ≥ 1.40 | Structured Outputs API |
| `pandas` | 1.5.3 | bundled `src/` + baselines |
| `scipy` | 1.10.1 | bundled `src/` + baselines |
| `matplotlib` | 3.7.1 | bundled `src/` + baselines |
| `scikit-uplift` | 0.5.1 | bundled `src/` + baselines |
| `jax` / `jaxlib` | 0.4.12 | optional — JAX CATENets backend (Torch is default) |
| `transformers` | ≥ 4.40 | optional — local MedGemma inference |

The first six are required to run the four pipeline stages; the rest support the bundled
library, baselines, and optional backends. All versions are pinned in `requirements.txt`.

### Operating systems

Developed and tested on **Linux** (x86-64; kernel 6.12, Rocky/RHEL-family). The code is
pure Python and expected to run on macOS and Windows, but only Linux has been tested.

### Hardware

- A **CUDA-capable GPU** is recommended for the SHAP/CATE stage
  (`ensemble_shap.py` trains a bootstrapped ensemble; set `DEVICE=cuda:0`).
  It runs on CPU but is substantially slower.
- The literature-validation and judging stages are network/LLM-bound and need no GPU.
- An API key for an LLM provider (OpenAI or OpenRouter) is required for the
  explanation-generation, literature-validation, and judging stages (not needed when
  running explanation generation with a local MedGemma model).
- No non-standard hardware is required.

---

## Installation guide

```bash
# 1. Clone / unpack this repository, then from its top level:
conda create -n alex python=3.10 -y
conda activate alex
pip install -r requirements.txt

# 2. Provide an LLM API key (src/constants.py is a placeholder that reads the env var):
export OPENAI_API_KEY=sk-...          # or configure OpenRouter, see clinical_agent.py
```

`src/` is kept as a sibling package so that `from src...` imports resolve; run all
scripts from this top-level directory.

**Typical install time:** ~5–10 minutes on a normal desktop (dominated by the `torch`
download). No compilation step.

---

## Demo

No demo data ships (the cohorts are access-restricted). Once a cohort is registered in
`src/dataset.py`, the full pipeline runs end-to-end:

```bash
# Run all four stages for one cohort + seed:
COHORT=crash_2 SEED=0 DEVICE=cuda:0 ./run_pipeline.sh
```

`run_pipeline.sh` runs the stages in order, skips any whose output already exists
(`FORCE=1` to recompute), and fails fast. Every parameter is an overridable environment
variable documented at the top of the script.

**Expected output (under `results/<cohort>/`):** a SHAP summary JSON in `shapley/`, an
explanations JSON, a PubMed-validation JSON, and a judge/logic-score JSON.

**Expected run time:** the SHAP stage is ~5–20 min on a single GPU for 20 bootstrap
trials (cohort-size dependent); each LLM stage is seconds-to-minutes depending on the
provider.

---

## Instructions for use

The pipeline has four stages, orchestrated by `run_pipeline.sh`. Each stage is also a
standalone script you can run directly (`--help` documents its arguments).

| Stage | Script | What it does |
|-------|--------|--------------|
| 1. SHAP computation | `ensemble_shap.py` | Trains a bootstrapped CATE ensemble (`src.CATENets` learners) and computes ensemble Shapley values via `captum`. Writes a SHAP-summary JSON to `results/<cohort>/shapley/`. |
| 2. Explanation generation | `clinical_agent.py` | Generates mechanistic explanations for the top SHAP features via an LLM (OpenAI / OpenRouter / local MedGemma), with an optional self-verification loop. |
| 3. Literature validation | `pubmed_mechanism_validator.py` | Validates each proposed mechanism explanation against PubMed abstracts using a tiered keyword search strategy. |
| 4. Logic scoring | `judge_evaluation.py` | Independent gate-based logic scoring of the generated explanations. |

**Run on a new cohort** by registering it in `src/dataset.py` and setting `COHORT=<name>`.
**Reproduce the manuscript** by running each reported cohort/seed via `COHORT`/`SEED`.
Outputs are written under `results/`.

### Baselines (`baselines/`)

- `cot_baseline.py` — chain-of-thought explanation-generation baseline.
- `hypogenic.py` — HypoGeniC-style baseline.
- `ResearchAgent/` — multi-agent research baseline (problem / method / experiment agents).

---

## Repository layout

```
run_pipeline.sh                Main entry point — runs all four stages for a cohort + seed
ensemble_shap.py               Stage 1: SHAP computation
clinical_agent.py              Stage 2: explanation generation
pubmed_mechanism_validator.py  Stage 3: PubMed validation
judge_evaluation.py            Stage 4: gate-based scoring
baselines/                     Baseline methods
src/                           Bundled library: dataset loader, CATE estimators (CATENets),
                               agent utilities and Pydantic schemas (code only — no data)
requirements.txt
LICENSE.md                     CC BY-NC-ND 4.0 license
```

---

## Data availability

Datasets are obtained from their original repositories under their respective access
agreements — they are **not** bundled here; only the loader *code* (`src/dataset.py`,
`src/CATENets/`) is included.

| Dataset | Access |
|---------|--------|
| Synthetic (generator) | [CATENets](https://github.com/AliciaCurth/CATENets) |
| IST-3 | [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/1931) |
| CRASH-2 | [FreeBIRD portal](https://freebird.lshtm.ac.uk/index.php/available-trials/) (treatment-allocation data on request) |
| ACCORD, SPRINT | [NHLBI BioLINCC](https://biolincc.nhlbi.nih.gov/home/) (on application) |

This review snapshot also omits experiment outputs, caches, human-study annotations, the
bundled CATENets benchmark datasets, and secrets — none are needed to review the method.

---

## License

The ALEX code is released under the **Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license;
see [`LICENSE.md`](LICENSE.md).

> Note: CC BY-NC-ND is a content license rather than an OSI-approved software license,
> and its NonCommercial / NoDerivatives terms are more restrictive than typical code
> licenses. Some journals' code policies expect an OSI-approved license — confirm
> CC BY-NC-ND is acceptable to the target journal before submission.

The bundled CATENets code (`src/CATENets/`) is third-party and remains under its own
**BSD 3-Clause License** (© 2021 Alicia Curth); see `src/CATENets/LICENSE`. The
CC BY-NC-ND terms above apply to the ALEX code, not to this third-party component.