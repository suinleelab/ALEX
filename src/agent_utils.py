import os
import re
import json
from typing import Dict, List, Optional

from src.agent_schemas import ArticleExtraction


COMMON_FEATURE_LABEL_MAP = {
    "age": "Age",
    "sbp": "Systolic blood pressure",
    "dbp": "Diastolic blood pressure",
    "egfr": "Estimated glomerular filtration rate",
    "screat": "Serum creatinine",
    "creat": "Serum creatinine",
    "bmi": "Body mass index",
    "hr": "Heart rate",
    "sex": "Sex",
    "male": "Male sex",
    "female": "Female sex",
    "uacr": "Urine albumin-to-creatinine ratio",
    "hba1c": "Hemoglobin A1c",
    "ldl": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "tg": "Triglycerides",
    "chr": "Total cholesterol / HDL ratio",
}


DATASET_FEATURE_LABEL_MAP = {
    "crash_2": {
        "ninjurytime": "Time from injury to treatment",
        "injurytime": "Time from injury to treatment",
        "icc": "Injury classification code",
    },
    "ist3": {
        "dbprand": "Randomization/baseline diastolic BP variable",
    },
}


# Canonical list of features (raw column names) that were actually measured and
# recorded in each trial dataset.  Used by the judge for Gate 1 verification.
DATASET_KNOWN_FEATURES: Dict[str, List[str]] = {
    "ist3": [
        "nihss",
        "antiplat_rand",
        "Stroke Type: TACI (Total Anterior Circulation Infarct)",
        "age",
        "dbprand",
        "gcs_score_rand",
        "weight",
        "sbprand",
        "Stroke Type: PACI (Partial Anterior Circulation Infarct)",
        "atrialfib_rand",
        "glucose",
        "gender",
        "infarct",
        "Stroke Type: POCI (Posterior Circulation Infarct)",
        "Stroke Type: LACI (Lacunar Infarct)",
    ],
    "crash_2": [
        "iinjurytype",
        "isbp",
        "icc",
        "ninjurytime",
        "ihr",
        "igcs",
        "irr",
        "iage",
        "isex",
    ],
    "sprint": [
        "sub_cvd",
        "sub_ckd",
        "race_black",
        "sbp",
        "age",
        "dbp",
        "chr",
        "hdl",
        "bmi",
        "smoke_3cat",
        "glur",
        "female",
        "aspirin",
        "statin",
        "trr",
        "umalcr",
    ],
    "accord": [
        "anti_coag",
        "bmi",
        "baseline_age",
        "ldl",
        "potassium",
        "hdl",
        "hr",
        "raceclass",
        "dbp",
        "sbp",
        "fpg",
        "antiarrhythmic",
        "aspirin",
        "bp_med",
        "x4smoke",
        "female",
        "gfr",
        "statin",
        "cvd_hx_baseline",
        "alt",
        "trig",
        "cpk",
        "uacr",
    ],
    "accord_glycemia": [
        "baseline_age",
        "bmi",
        "hba1c",
        "yrsdiab",
        "sbp",
        "dbp",
        "hr",
        "fpg",
        "alt",
        "cpk",
        "potassium",
        "gfr",
        "uacr",
        "trig",
        "ldl",
        "hdl",
        "bp_med",
        "dm_med",
        "female",
        "raceclass",
        "cvd_hx_baseline",
        "insulin",
        "statin",
        "aspirin",
        "antiarrhythmic",
        "anti_coag",
        "x4smoke",
    ],
}


def get_dataset_features(dataset: Optional[str]) -> List[str]:
    """Return the canonical measured-feature list for a known dataset, or [] if unknown."""
    if not dataset:
        return []
    return DATASET_KNOWN_FEATURES.get(dataset.lower(), [])


def map_feature_label(raw_feature: str, dataset: Optional[str] = None) -> str:
    """Map raw feature names to clinician-readable labels while preserving raw codes."""
    if not raw_feature:
        return raw_feature

    raw = str(raw_feature).strip()
    key = raw.lower()
    dataset_key = (dataset or "").lower()

    mapped = None
    if dataset_key in DATASET_FEATURE_LABEL_MAP:
        mapped = DATASET_FEATURE_LABEL_MAP[dataset_key].get(key)
    if not mapped:
        mapped = COMMON_FEATURE_LABEL_MAP.get(key)

    if not mapped:
        if "_" in raw:
            humanized = raw.replace("_", " ").strip()
            humanized = re.sub(r"\s+", " ", humanized)
            if humanized and humanized.lower() != key:
                mapped = humanized[:1].upper() + humanized[1:]

    if mapped and mapped.lower() != key:
        return f"{mapped} ({raw})"
    return raw


def load_local_env(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ.

    Existing environment variables are not overwritten.
    """
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def ensure_out_dir(path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def load_json_file(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json_file(path: str, payload: dict) -> None:
    ensure_out_dir(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def infer_revised_path(original_path: str) -> Optional[str]:
    base, ext = os.path.splitext(original_path)
    candidate = f"{base}_revised{ext}"
    return candidate if os.path.exists(candidate) else None


def resolve_seeded_output_path(path: str, seed: int) -> str:
    """Insert seed subfolder into output path unless one already exists."""
    normalized = path.replace("\\", "/")
    directory, filename = os.path.split(normalized)

    directory_parts = [part for part in directory.split("/") if part]
    if any(part.startswith("seed_") for part in directory_parts):
        return path

    seeded_directory = os.path.join(directory, f"seed_{seed}") if directory else f"seed_{seed}"
    return os.path.join(seeded_directory, filename)


def get_trial_metadata(trial_name: str) -> dict:
    """Return treatment/outcome/population metadata for known clinical trials."""
    trial_map = {
        "ist3": {
            "treatment": "IV alteplase (recombinant tissue plasminogen activator)",
            "outcome": "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
            "population": "Acute ischemic stroke patients within 6 hours of symptom onset",
            "article_query": "IST-3 trial alteplase stroke Sandercock 2012",
        },
        "crash_2": {
            "treatment": "Tranexamic acid (TXA)",
            "outcome": "All-cause mortality at 28 days or in-hospital death",
            "population": "Trauma patients with significant bleeding or at risk of significant hemorrhage",
            "article_query": "CRASH-2 trial tranexamic acid trauma 2010",
        },
        "sprint": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
            "population": "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
            "article_query": "SPRINT trial intensive blood pressure control 2015",
        },
        "accord": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
            "article_query": "ACCORD BP trial intensive blood pressure control diabetes 2010",
        },
        "accord_glycemia": {
            "treatment": "Intensive glycemic control (target HbA1c <6.0%)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
            "article_query": "ACCORD glycemia trial intensive glucose lowering type 2 diabetes 2008",
        },
        "txa": {
            "treatment": "Pre-hospital tranexamic acid (TXA) administration",
            "outcome": "Survival (in-hospital mortality status)",
            "population": "Adult trauma patients in a pre-hospital TXA cohort",
            "article_query": "pre-hospital TXA trauma cohort retrospective study",
        },
    }

    trial_lower = trial_name.lower()
    if trial_lower not in trial_map:
        raise ValueError(
            f"Unknown trial: {trial_name}. Supported trials: {', '.join(trial_map.keys())}.\n"
            "Use --treatment, --outcome, --population arguments instead for custom trials."
        )
    return trial_map[trial_lower]


# ---------------------------------------------------------------------------
# MedGemma local pipeline wrapper (OpenAI-compatible interface)
# ---------------------------------------------------------------------------

class _MedGemmaMessage:
    """Mimics openai ChatCompletionMessage."""
    def __init__(self, content: str):
        self.content = content
        self.parsed = None
        self.refusal = None


class _MedGemmaChoice:
    def __init__(self, content: str):
        self.message = _MedGemmaMessage(content)


class _MedGemmaCompletion:
    def __init__(self, content: str):
        self.choices = [_MedGemmaChoice(content)]


class _MedGemmaCompletions:
    """Drop-in for `client.chat.completions`."""

    def __init__(self, pipe):
        self._pipe = pipe

    def create(self, *, model: str = "", messages: list, max_tokens: int = 4096, **kwargs) -> _MedGemmaCompletion:
        # Cap max_new_tokens to a practical limit for local inference
        capped_tokens = min(max_tokens, 8192)
        # Use sampling with temperature=1.0 to match the stochastic behaviour of
        # API-based models (GPT, Gemini, Qwen).  Pass through an optional seed so
        # callers can control reproducibility per run.
        seed = kwargs.get("seed", None)
        output = self._pipe(
            messages,
            max_new_tokens=capped_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=1.0,
            **({"seed": seed} if seed is not None else {}),
        )
        # HF pipeline returns list; last generated message is assistant reply
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            content = generated[-1].get("content", "")
        else:
            content = str(generated)
        return _MedGemmaCompletion(content)


class _MedGemmaBetaParsed:
    """Stub for `client.beta.chat.completions.parse` — always raises so
    `_parse_structured` falls through to the plain-create fallback."""

    def parse(self, **kwargs):
        raise NotImplementedError("MedGemma does not support structured outputs")


class _MedGemmaBeta:
    def __init__(self, pipe):
        self.chat = type("obj", (object,), {"completions": _MedGemmaBetaParsed()})()


class MedGemmaClient:
    """Lightweight wrapper around a HuggingFace text-generation pipeline that
    exposes the subset of the OpenAI Python client interface used by
    ``_parse_structured`` and the rest of the agent code.

    Usage (automatic via ``get_model_client('medgemma')``):
        client = MedGemmaClient(model_name="google/medgemma-27b-text-it")

    Authentication:
        MedGemma is a gated model. You must:
        1. Accept the license at https://huggingface.co/google/medgemma-27b-text-it
        2. Authenticate via one of:
           - ``huggingface-cli login``
           - Set HF_LOGIN (or HF_TOKEN) environment variable
           - Pass token= to this constructor
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        device: str = "cuda",
        token: Optional[str] = None,
    ):
        import torch

        # Check PyTorch version early
        torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if torch_version < (2, 4):
            raise RuntimeError(
                f"MedGemma requires PyTorch >= 2.4 but found {torch.__version__}. "
                "Upgrade with: pip install 'torch>=2.4'"
            )

        from transformers import pipeline as hf_pipeline

        # Resolve HF token: explicit arg > HF_LOGIN > HF_TOKEN > cached login
        hf_token = token or os.getenv("HF_LOGIN") or os.getenv("HF_TOKEN")

        print(f"Loading MedGemma model '{model_name}' on {device} …")
        try:
            self._pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.bfloat16,
                device=device,
                token=hf_token,
            )
        except OSError as e:
            if "gated repo" in str(e).lower() or "401" in str(e):
                raise RuntimeError(
                    f"Cannot access gated model '{model_name}'. "
                    "Please:\n"
                    "  1. Accept the license at https://huggingface.co/google/medgemma-27b-text-it\n"
                    "  2. Authenticate: run 'huggingface-cli login' or set HF_LOGIN env var\n"
                    f"Original error: {e}"
                ) from e
            raise
        self._model_name = model_name
        # Public attributes checked by clinical_agent._parse_structured
        self._base_url = None  # not an OpenRouter client
        self._is_medgemma = True
        self.chat = type("obj", (object,), {"completions": _MedGemmaCompletions(self._pipe)})()
        self.beta = _MedGemmaBeta(self._pipe)


def get_model_client(
    api_provider: str,
    api_base_url: Optional[str] = None,
    medgemma_model: Optional[str] = None,
    medgemma_device: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Create an OpenAI-compatible client for OpenAI, OpenRouter, or MedGemma (local)."""

    if api_provider == "medgemma":
        model = medgemma_model or "google/medgemma-27b-text-it"
        device = medgemma_device or "cuda"
        return MedGemmaClient(model_name=model, device=device, token=hf_token)

    from openai import OpenAI

    if api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY.")
        base_url = api_base_url or "https://openrouter.ai/api/v1"
        print(f"Using OpenRouter API with base URL: {base_url}")
        return OpenAI(api_key=api_key, base_url=base_url)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            from src.constants import openai_api_key

            api_key = openai_api_key
        except ImportError:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or define src/constants.py:openai_api_key"
            )

    if api_base_url:
        print(f"Using custom OpenAI-compatible API with base URL: {api_base_url}")
        return OpenAI(api_key=api_key, base_url=api_base_url)

    print("Using OpenAI API")
    return OpenAI(api_key=api_key)


def load_top_features(
    shap_json_path: str, n_features: int, dataset_override: str = None
) -> dict:
    with open(shap_json_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    explainer = meta.get("explainer", "unknown_explainer")
    dataset = dataset_override if dataset_override else meta.get("dataset", "unknown_dataset")
    learner = meta.get("learner", "unknown_learner")

    features = data.get("features", [])

    all_feature_names = [
        map_feature_label(f.get("feature"), dataset) for f in features if f.get("feature")
    ]

    if not features:
        return {
            "dataset": dataset,
            "learner": learner,
            "explainer": "baseline_no_shap",
            "top_feature_evidence": [],
            "available_features": [],
        }

    features_sorted = sorted(
        features, key=lambda x: float(x.get("shap_mean_abs", 0.0)), reverse=True
    )
    top = features_sorted[:n_features]

    top_evidence = [
        {
            "feature": map_feature_label(f.get("feature"), dataset),
            "feature_raw": f.get("feature"),
            "feature_index": f.get("feature_index"),
            "topN_frequency_pct": f.get("topN_frequency_pct"),
            "shap_mean_abs": f.get("shap_mean_abs"),
            "shap_mean": f.get("shap_mean"),
            "pearson_sign_pos_frac": f.get("pearson_sign_pos_frac"),
            "pearson_sign_neg_frac": f.get("pearson_sign_neg_frac"),
        }
        for f in top
    ]

    return {
        "dataset": dataset,
        "learner": learner,
        "explainer": explainer,
        "top_feature_evidence": top_evidence,
        "available_features": all_feature_names,
    }


def search_and_extract_article(
    query: str,
    trial_name: str,
    client,
    model_name: str = "gpt-4o-2024-08-06",
) -> Optional[ArticleExtraction]:
    """Search for trial article and extract key information."""

    known_articles = {
        "ist3": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext",
        "crash_2": "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext",
        "sprint": "https://www.nejm.org/doi/full/10.1056/NEJMoa1511939",
        "accord": "https://www.nejm.org/doi/full/10.1056/NEJMoa1001286",
        "accord_glycemia": "https://www.nejm.org/doi/full/10.1056/NEJMoa0802743",
    }

    trial_lower = trial_name.lower()
    if trial_lower not in known_articles:
        print(f"Warning: No known article URL for trial '{trial_name}'. Skipping article retrieval.")
        return None

    article_url = known_articles[trial_lower]

    extraction_system = (
        "You are a clinical research extraction assistant. Extract key information "
        "from a clinical trial article to provide context for explanation generation. "
        "Be accurate and cite only what is typically reported in such trials. "
        "If you don't know specific details, use 'not specified' or mark fields as null."
    )

    extraction_prompt = {
        "task": "Extract trial characteristics and results",
        "trial_name": trial_name,
        "query": query,
        "article_url": article_url,
        "instructions": [
            "Extract metadata (title, authors, journal, year, DOI)",
            "Extract trial design (sample size, intervention, control, outcomes)",
            "Extract key results including any subgroup analyses",
            "Note study limitations",
            "Explain how this context relates to ML-generated explanations about treatment heterogeneity",
        ],
        "note": "Use your knowledge of this published trial. Be conservative - don't invent details.",
    }

    try:
        extraction = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": json.dumps(extraction_prompt, indent=2)},
            ],
            response_format=ArticleExtraction,
        )
        return extraction.choices[0].message.parsed
    except Exception as e:
        print(f"Error extracting article information: {e}")
        return None


def _is_hypogenic_format(data: Dict) -> bool:
    """Return True if data is a HypoGeniC explanations.json (internal format).

    Accepts both the old list format and the new faithful dict-keyed format.
    """
    return data.get("method") == "HypoGeniC" and isinstance(data.get("explanations"), (list, dict))


def _convert_hypogenic_to_feature_format(data: Dict) -> Dict:
    """Convert HypoGeniC explanations.json to the feature_explanations format expected
    by the PubMed validator and judge.

    Handles two formats:
    - Old nested format: explanations is a list of {explanation: InternalExplanation, acc, ...}
    - New faithful format: explanations is a dict {text: {acc, reward, num_visits, subgroup_rule?, ...}}

    In both cases, each explanation becomes its own entry. importance_rank is assigned
    per unique feature in order of first appearance.
    """
    _benefit_map = {
        "high": "higher_benefit", "moderate": "higher_benefit",
        "low": "lower_benefit", "none": "lower_benefit", "harm": "higher_harm",
    }
    ctx = data.get("study_context", {})
    raw = data.get("explanations", {})

    # Normalise to a flat list of dicts with consistent keys
    if isinstance(raw, dict):
        # New faithful format: {text: {acc, reward, num_visits, subgroup_rule?, recommendation?}}
        normalised = []
        for text, stats in raw.items():
            rule = stats.get("subgroup_rule") or {}
            normalised.append({
                "_text": text,
                "_feature": rule.get("feature", "unknown"),
                "_rule_description": rule.get("description") or text,
                "_recommendation": stats.get("recommendation", "unclear"),
                "_expected_benefit": "",   # not stored in plain format
                "_explanation_id": "",
                "_title": "",
            })
    else:
        # Old nested format: list of {explanation: {...InternalExplanation fields...}, acc, ...}
        normalised = []
        for entry in raw:
            h = entry.get("explanation", {})
            rec = h.get("treatment_recommendation") or {}
            rule = rec.get("subgroup_rule") or {}
            text = h.get("explanation_statement", "")
            normalised.append({
                "_text": text,
                "_feature": rule.get("feature", "unknown"),
                "_rule_description": rule.get("description", "") or h.get("mechanism", "") or text,
                "_recommendation": rec.get("recommendation", "unclear"),
                "_expected_benefit": rec.get("expected_benefit", ""),
                "_explanation_id": h.get("explanation_id", ""),
                "_title": h.get("title", ""),
            })

    # Assign a stable rank per unique feature (order of first appearance)
    feature_rank: Dict[str, int] = {}
    for item in normalised:
        feat = item["_feature"]
        if feat not in feature_rank:
            feature_rank[feat] = len(feature_rank) + 1

    feature_explanations = []
    for idx, item in enumerate(normalised, 1):
        feat = item["_feature"]
        text = item["_text"]
        feature_explanations.append({
            "feature_name": feat,
            "explanation_id": item["_explanation_id"] or f"hyp_{idx}",
            "title": item["_title"] or "",
            "importance_rank": feature_rank[feat],
            "explanation_rank": idx,
            "shap_value": 0.0,
            "effect_direction": _benefit_map.get(item["_expected_benefit"], "ambiguous"),
            # The plain explanation text is used directly — no LLM expansion needed
            "clinical_interpretation": text,
            "why_important": "Treatment effect modifier identified by HypoGeniC",
            "mechanisms": [
                {
                    "mechanism_type": "biological",
                    # Use the explanation statement itself as the mechanism description;
                    # the judge and PubMed validator will score/search on this text.
                    "description": text,
                    "evidence_level": "moderate",
                }
            ],
            "subgroup_implications": item["_rule_description"],
            "validation_suggestions": [],
            "caveats": [],
        })

    return {
        "dataset": ctx.get("dataset", "unknown"),
        "model": "HypoGeniC",
        "treatment": ctx.get("treatment", ""),
        "outcome": ctx.get("outcome", ""),
        "population": ctx.get("population", ""),
        "summary": (
            f"HypoGeniC: {len(feature_rank)} features, {len(feature_explanations)} explanations "
            f"from {ctx.get('dataset', 'unknown')}"
        ),
        "feature_explanations": feature_explanations,
        "n_unique_features": len(feature_rank),
        "n_explanations_per_feature": len(feature_explanations) // max(len(feature_rank), 1),
        "cross_feature_patterns": None,
    }