#!/usr/bin/env bash
#
# run_pipeline.sh — end-to-end ALEX pipeline for a single cohort + seed.
#
# Stages:
#   1. SHAP computation   (ensemble_shap.py)
#   2. Explanation gen.   (clinical_agent.py)
#   3. PubMed validation  (pubmed_mechanism_validator.py)
#   4. Logic scoring      (judge_evaluation.py)
#
# Each stage skips itself if its output already exists (set FORCE=1 to recompute).
# The script runs from its own directory so that `from src...` imports resolve.
#
# Usage:
#   ./run_pipeline.sh                       # run with the defaults below
#   COHORT=ist3 SEED=0 ./run_pipeline.sh    # override any config var inline
#   FORCE=1 ./run_pipeline.sh               # ignore existing outputs and recompute
#
# Requires: an LLM API key in the environment (e.g. OPENAI_API_KEY); see README.md.

set -euo pipefail

# Always operate from the directory containing this script (repo top level).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Configuration (override any of these via environment variables)
# ---------------------------------------------------------------------------
COHORT="${COHORT:-crash_2}"          # cohort name registered in src/dataset.py
SEED="${SEED:-0}"                    # seed for explanation generation
DEVICE="${DEVICE:-cuda:0}"           # CUDA device for the SHAP/CATE stage
LEARNER="${LEARNER:-DRLearner}"      # CATE learner class

# Stage-1 (SHAP) parameters
NUM_TRIALS="${NUM_TRIALS:-20}"
TOP_N_FEATURES="${TOP_N_FEATURES:-10}"
RELATIVE_CHANGE_THRESHOLD="${RELATIVE_CHANGE_THRESHOLD:-0.05}"
BASELINE="${BASELINE:-false}"        # true → random-sample baseline; false → median

# Stage-2 (explanation generation) parameters
N_EXPLANATIONS="${N_EXPLANATIONS:-1}"
GEN_MODEL="${GEN_MODEL:-gpt-5-mini}"
GEN_PROVIDER="${GEN_PROVIDER:-openai}"
ENABLE_VERIFIER="${ENABLE_VERIFIER:-true}"

# Stage-3/4 (validation + judging) parameters
EVAL_MODEL="${EVAL_MODEL:-gpt-5-mini}"
EVAL_PROVIDER="${EVAL_PROVIDER:-openai}"
MAX_ABSTRACTS="${MAX_ABSTRACTS:-20}"
EMAIL="${EMAIL:-research@example.com}"   # contact email for the PubMed/NCBI API

FORCE="${FORCE:-0}"                  # 1 → recompute even if outputs exist

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
# Stage-1 writes results/<cohort>/shapley/<cohort>_shap_summary_<True|False>.json,
# where the suffix is the Python str() of the --baseline flag.
if [ "${BASELINE}" = "true" ]; then BASELINE_SUFFIX="True"; else BASELINE_SUFFIX="False"; fi
SHAP_JSON="results/${COHORT}/shapley/${COHORT}_shap_summary_${BASELINE_SUFFIX}.json"

OUT_DIR="results/${COHORT}/${GEN_MODEL}/with_shap_${LEARNER,,}/seed_${SEED}"
EXPLANATIONS_JSON="${OUT_DIR}/explanations.json"
mkdir -p "${OUT_DIR}" logs

log() { echo "[$(printf '%(%H:%M:%S)T')] [${COHORT}] $*"; }
exists() { [ "${FORCE}" != "1" ] && [ -f "$1" ]; }

# ---------------------------------------------------------------------------
# Stage 1 — SHAP computation
# ---------------------------------------------------------------------------
log "Stage 1/4: SHAP computation -> ${SHAP_JSON}"
if exists "${SHAP_JSON}"; then
    log "  ↳ skip (exists; FORCE=1 to recompute)"
else
    baseline_flag=()
    [ "${BASELINE}" = "true" ] && baseline_flag=(--baseline)
    python ensemble_shap.py \
        --num_trials "${NUM_TRIALS}" \
        --cohort_name "${COHORT}" \
        --device "${DEVICE}" \
        --learner "${LEARNER}" \
        --relative_change_threshold "${RELATIVE_CHANGE_THRESHOLD}" \
        --top_n_features "${TOP_N_FEATURES}" \
        "${baseline_flag[@]}"
fi
[ -f "${SHAP_JSON}" ] || { log "✗ SHAP summary not found at ${SHAP_JSON}"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 2 — Explanation generation
# ---------------------------------------------------------------------------
log "Stage 2/4: Explanation generation -> ${EXPLANATIONS_JSON}"
if exists "${EXPLANATIONS_JSON}"; then
    log "  ↳ skip (exists; FORCE=1 to recompute)"
else
    verifier_flag=()
    [ "${ENABLE_VERIFIER}" = "true" ] && verifier_flag=(--enable_verifier)
    python clinical_agent.py \
        --shap_json "${SHAP_JSON}" \
        --out_json "${EXPLANATIONS_JSON}" \
        --seed "${SEED}" \
        --trial_name "${COHORT}" \
        --n_features "${TOP_N_FEATURES}" \
        --n_explanations "${N_EXPLANATIONS}" \
        --model "${GEN_MODEL}" \
        --api_provider "${GEN_PROVIDER}" \
        "${verifier_flag[@]}"
fi
# Prefer a revised file if the verifier produced one, else the original.
PUBMED_INPUT="${EXPLANATIONS_JSON}"
[ -f "${OUT_DIR}/explanations_revised.json" ] && PUBMED_INPUT="${OUT_DIR}/explanations_revised.json"
[ -f "${PUBMED_INPUT}" ] || { log "✗ explanations not found at ${PUBMED_INPUT}"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 3 — PubMed validation
# ---------------------------------------------------------------------------
log "Stage 3/4: PubMed validation (input: ${PUBMED_INPUT})"
python pubmed_mechanism_validator.py \
    --input "${PUBMED_INPUT}" \
    --dataset "${COHORT}" \
    --max-abstracts "${MAX_ABSTRACTS}" \
    --email "${EMAIL}" \
    --model "${EVAL_MODEL}" \
    --api-provider "${EVAL_PROVIDER}"

# ---------------------------------------------------------------------------
# Stage 4 — Logic scoring (gate-based judge)
# ---------------------------------------------------------------------------
log "Stage 4/4: Logic scoring"
python judge_evaluation.py \
    --explanations_json "${PUBMED_INPUT}" \
    --shap_json "${SHAP_JSON}" \
    --trial_name "${COHORT}" \
    --model "${EVAL_MODEL}" \
    --api_provider "${EVAL_PROVIDER}"

log "✓ Pipeline complete. Outputs under results/${COHORT}/"
