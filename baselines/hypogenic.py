#!/usr/bin/env python3
"""hypogenic_baseline.py

Implement HypoGeniC (Explanation Generation in Context) algorithm for clinical trials.
This is an iterative explanation generation baseline that:
1. Generates initial explanations using LLM
2. Tests explanations on training samples
3. Refines explanations based on prediction errors
4. Generates new explanations from difficult samples

Based on: "Explanation Generation with Large Language Models"

Requires:
  pip install openai pydantic numpy pandas scikit-learn

Example:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_explanations.json \
    --num_init 20 \
    --top_k 10 \
    --alpha 0.5 \
    --update_batch_size 5 \
    --num_explanations_to_update 5

  Use custom number of samples:
  python hypogenic_baseline.py \
    --trial_name ist3 \
    --out_json docs/results/ist3/hypogenic_explanations.json \
    --max_samples 500

"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Dict, Type, TypeVar
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel

from src.dataset import Dataset


FIXED_TARGET_FEATURES = 5
FIXED_EXPLANATIONS_PER_FEATURE = 1
FIXED_TOTAL_EXPLANATIONS = FIXED_TARGET_FEATURES * FIXED_EXPLANATIONS_PER_FEATURE


def load_local_env(env_path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from a local .env file into os.environ.

    Existing environment variables are not overwritten.
    """
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    else:
        script_dir = Path(__file__).resolve().parent
        candidate_paths.extend([
            Path.cwd() / ".env",
            script_dir / ".env",
        ])

    env_file = next((path for path in candidate_paths if path.exists()), None)
    if not env_file:
        return

    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        print(f"Warning: unable to read .env file: {exc}")


def resolve_seeded_output_path(output_path: str, seed: int) -> str:
    """Insert seed_<seed> folder before filename unless already present."""
    path_obj = Path(output_path)
    if any(part.startswith("seed_") for part in path_obj.parts):
        return str(path_obj)

    parent = path_obj.parent
    seeded_parent = parent / f"seed_{seed}"
    return str(seeded_parent / path_obj.name)


# ---------------------------------------------------------------------------
# Robust structured-output helpers (Qwen / OpenRouter compatibility)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")
_NO_STRUCTURED = ('qwen', 'deepseek', 'llama', 'mistral', 'mixtral')

# Models that require max_completion_tokens instead of max_tokens
_NEW_OPENAI_PREFIXES = ('gpt-5', 'gpt-4.1', 'o1', 'o3', 'o4')

def _token_limit_kwarg(model_name: str, client, limit: int = 16384) -> dict:
    """Return the right token-limit kwarg for the model."""
    is_openrouter = getattr(client, '_base_url', None) and 'openrouter' in str(client._base_url)
    is_medgemma = getattr(client, '_is_medgemma', False)
    if not is_openrouter and not is_medgemma:
        ml = model_name.lower()
        if any(ml.startswith(p) for p in _NEW_OPENAI_PREFIXES):
            return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def _fix_unescaped_quotes(text: str) -> str:
    """Escape double-quotes that appear *inside* JSON string values.

    Heuristic: a `"` truly closes a string only if the next non-whitespace
    character is one of  : , } ] or end-of-input.  Otherwise it is an
    un-escaped in-string quote and gets replaced with `\\\"`.
    """
    out: list[str] = []
    i = 0
    in_str = False
    while i < len(text):
        ch = text[i]
        # handle backslash-escape inside strings
        if ch == '\\' and in_str:
            out.append(text[i:i+2] if i + 1 < len(text) else ch)
            i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                # peek ahead past whitespace
                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                if j >= len(text) or text[j] in ':,}]':
                    # structural close
                    in_str = False
                    out.append(ch)
                else:
                    # unescaped interior quote → escape it
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _lift_kv_from_array(text: str) -> str:
    """Fix Qwen output where top-level fields leak inside an array.

    Detects patterns like:
        "new_explanations": [
            "hyp1",
            "hyp2",
            "Refinement rationale": "...",   <-- key-value inside array!
            "addresses_patterns": [...]      <-- likewise
        ]
    and restructures to:
        "new_explanations": ["hyp1", "hyp2"],
        "refinement_rationale": "...",
        "addresses_patterns": [...]

    Known sibling fields that may be misplaced:
    """
    _KNOWN_KEYS = {
        "refinement_rationale", "refinement rationale",
        "addresses_patterns", "generation_context",
        "generation context", "addresses patterns",
    }
    # Pattern: comma + optional whitespace + "some key" + colon  (inside array)
    # We look for the first occurrence of a known key appearing as "key":
    pattern = re.compile(
        r',\s*"(' + '|'.join(re.escape(k) for k in _KNOWN_KEYS) + r')"\s*:',
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return text

    # Everything before the match → close the array there
    before = text[:m.start()].rstrip().rstrip(',')
    # Find the innermost unclosed [ before this point
    # We need to close it: append ]
    after = text[m.start() + 1:]  # skip the leading comma
    # Close the array and continue with the rest as sibling keys
    # Normalise key names: "Refinement rationale" → "refinement_rationale"
    after = re.sub(
        r'"(?:Refinement rationale|refinement rationale)"',
        '"refinement_rationale"', after, flags=re.IGNORECASE)
    after = re.sub(
        r'"(?:addresses patterns)"',
        '"addresses_patterns"', after, flags=re.IGNORECASE)
    after = re.sub(
        r'"(?:generation context)"',
        '"generation_context"', after, flags=re.IGNORECASE)
    result = before + '], ' + after.lstrip()
    # The old "]}" that closed the array+object is now just "}" because
    # we already closed the array.  Remove one stray "]" if present.
    result = re.sub(r'\]\s*\]\s*\}', ']}', result)
    return result


def _repair_truncated_json(text: str) -> str:
    # First fix structural issue: key-value pairs misplaced inside arrays
    text = _lift_kv_from_array(text)
    # Remove trailing unterminated string value (possibly multi-line)
    text = re.sub(r',?\s*"(?:[^"\\]|\\.)*$', '', text, flags=re.DOTALL)
    # Remove trailing unterminated key: "key":  (no value)
    text = re.sub(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$', '', text, flags=re.DOTALL)
    # Remove trailing unterminated key-value: "key": "partial value...
    text = re.sub(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*"(?:[^"\\]|\\.)*$', '', text, flags=re.DOTALL)
    text = text.rstrip().rstrip(',')
    # Fix missing commas between strings: "..." "..." → "...", "..."
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    # Also handle same-line missing commas: "..." "..." → "...", "..."
    text = re.sub(r'"\s+"', '", "', text)
    # Fix missing commas between } and {: } { → }, {
    text = re.sub(r'\}\s*\{', '}, {', text)
    # Fix missing commas between } and ": } "..." → }, "..."
    text = re.sub(r'\}\s*"', '}, "', text)
    # Fix missing commas between "..." and {: "..." { → "...", {
    text = re.sub(r'"\s*\{', '", {', text)
    # Fix missing commas between ] and string/number
    text = re.sub(r'\]\s*"', '], "', text)
    opens = brackets = 0
    for ch in text:
        if ch == '{': opens += 1
        elif ch == '}': opens -= 1
        elif ch == '[': brackets += 1
        elif ch == ']': brackets -= 1
    text += ']' * max(brackets, 0) + '}' * max(opens, 0)
    return text


def _normalize_hypogenic_dict(data: dict, response_format) -> dict:
    """Normalise raw LLM dict for GeneratedExplanationSet / RefinedExplanationSet."""
    key = "explanations" if "explanations" in data else "new_explanations"
    _REC_MAP = {"tre": "treat", "con": "control", "unc": "unclear"}
    _BEN_MAP = {"hig": "high", "mod": "moderate", "low": "low", "non": "none", "har": "harm"}
    _OP_VALID = {">", "<", ">=", "<=", "==", "!="}

    def _fix(val, mapping, default):
        parts = (val or "").lower().split()
        if not parts:
            return default
        v = parts[0].rstrip(".,;:(")
        for prefix, canonical in mapping.items():
            if v.startswith(prefix):
                return canonical
        return default

    raw_items = data.get(key, [])
    normalized_items = []
    for hyp in raw_items:
        if isinstance(hyp, str):
            # Plain string explanation → wrap in minimal dict
            hyp = {
                "explanation_statement": hyp,
                "treatment_recommendation": {
                    "recommendation": "unclear",
                    "expected_benefit": "moderate",
                    "rationale": "",
                    "subgroup_rule": {"feature": "unknown", "operator": ">=", "description": hyp},
                },
            }
        normalized_items.append(hyp)
    data[key] = normalized_items

    for hyp in data.get(key, []):
        hyp.setdefault("explanation_statement", hyp.get("explanation", ""))
        tr = hyp.get("treatment_recommendation")
        if tr is None or (isinstance(tr, dict) and not tr):
            # treatment_recommendation entirely missing — build a default
            stmt = hyp.get("explanation_statement", "")
            tr = {
                "recommendation": "unclear",
                "expected_benefit": "moderate",
                "rationale": "",
                "subgroup_rule": {"feature": "unknown", "operator": ">=", "description": stmt},
            }
            hyp["treatment_recommendation"] = tr
        if isinstance(tr, str):
            tr = {"recommendation": tr}
            hyp["treatment_recommendation"] = tr
        tr["recommendation"] = _fix(tr.get("recommendation", ""), _REC_MAP, "unclear")
        tr["expected_benefit"] = _fix(tr.get("expected_benefit", ""), _BEN_MAP, "moderate")
        tr.setdefault("rationale", "")
        if isinstance(tr.get("rationale"), (dict, list)):
            tr["rationale"] = json.dumps(tr["rationale"]) if isinstance(tr["rationale"], dict) else " ".join(str(x) for x in tr["rationale"])
        sr = tr.get("subgroup_rule", {})
        if isinstance(sr, str):
            sr = {"description": sr, "feature": "unknown", "operator": ">="}
        if not sr:
            # subgroup_rule entirely missing — create a default from the explanation text
            stmt = hyp.get("explanation_statement", "")
            sr = {"feature": "unknown", "operator": ">=", "description": stmt}
        tr["subgroup_rule"] = sr
        sr.setdefault("feature", "unknown")
        sr.setdefault("description", "")
        if sr.get("operator") not in _OP_VALID:
            sr["operator"] = ">="

    data.setdefault("generation_context", data.get("refinement_rationale", ""))
    data.setdefault("refinement_rationale", "")
    data.setdefault("addresses_patterns", "")
    # Coerce dict/list → string for string-typed fields
    for str_field in ("generation_context", "refinement_rationale", "addresses_patterns"):
        val = data.get(str_field)
        if isinstance(val, dict):
            data[str_field] = json.dumps(val)
        elif isinstance(val, list):
            data[str_field] = " ".join(str(x) for x in val)
    return data


def _truncate_to_valid_json(text: str) -> str:
    """Iteratively truncate at unterminated-string positions and re-close brackets."""
    for _ in range(5):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            msg = str(e)
            if "Unterminated string" in msg or "EOF while parsing" in msg:
                # Truncate everything from the opening " of the bad string
                text = text[:e.pos].rstrip().rstrip(',')
                # Also strip any trailing partial key: , "key":
                text = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', text)
                text = text.rstrip().rstrip(',')
                # Re-close brackets
                opens = brackets = 0
                for ch in text:
                    if ch == '{': opens += 1
                    elif ch == '}': opens -= 1
                    elif ch == '[': brackets += 1
                    elif ch == ']': brackets -= 1
                text += ']' * max(brackets, 0) + '}' * max(opens, 0)
            else:
                break
    return text


def _extract_json_from_content(content: str, response_format: Type[_T]) -> _T:
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    obj_match = re.search(r"\{[\s\S]*\}", content, flags=re.DOTALL)
    if obj_match:
        content = obj_match.group(0)
    # Strip trailing commas before ] or } (common Qwen issue)
    content = re.sub(r',\s*([\]\}])', r'\1', content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(content)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: fix unescaped inner quotes, then truncate at error pos
            fixed = _fix_unescaped_quotes(repaired)
            truncated = _truncate_to_valid_json(fixed)
            try:
                data = json.loads(truncated)
            except json.JSONDecodeError as exc:
                print(f"[hypogenic] JSON still invalid after all repairs "
                      f"({exc}); attempting model_validate_json …")
                print(f"[hypogenic] repaired content (last 500 chars):\n{truncated[-500:]}")
                return response_format.model_validate_json(truncated)
    if isinstance(data, dict):
        data = _normalize_hypogenic_dict(data, response_format)
    return response_format.model_validate(data)


def _parse_structured(
    client: OpenAI,
    model_name: str,
    messages: list,
    response_format: Type[_T],
) -> _T:
    is_openrouter = getattr(client, '_base_url', None) and 'openrouter' in str(client._base_url)
    is_medgemma = getattr(client, '_is_medgemma', False)
    model_lower = model_name.lower()
    skip_parse = is_medgemma or (is_openrouter and any(t in model_lower for t in _NO_STRUCTURED))

    if not skip_parse:
        try:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=response_format,
                **_token_limit_kwarg(model_name, client),
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
            content = completion.choices[0].message.content or ""
            return _extract_json_from_content(content, response_format)
        except Exception:
            pass

    schema = response_format.model_json_schema()
    top_fields = list(schema.get("properties", {}).keys())
    fields_hint = ", ".join(f'"{ f}"' for f in top_fields)
    augmented = list(messages)
    augmented[0] = dict(augmented[0])
    augmented[0]["content"] = (
        augmented[0]["content"]
        + f"\n\nIMPORTANT: Your entire response must be a single valid JSON object with "
        f"top-level keys: {fields_hint}. Do not include any explanation, markdown, or "
        f"schema definitions — only the filled JSON instance."
    )
    extra_body: dict = {}
    if is_openrouter:
        extra_body["reasoning"] = {"effort": "none"}
    kwargs: dict = dict(model=model_name, messages=augmented, **_token_limit_kwarg(model_name, client))
    if extra_body:
        kwargs["extra_body"] = extra_body
    fallback = client.chat.completions.create(**kwargs)
    content = fallback.choices[0].message.content or ""
    return _extract_json_from_content(content, response_format)


# ---------------------------------------------------------------------------
# Internal helpers for HypoGeniC algorithm
# ---------------------------------------------------------------------------
# Design principle (faithful to original ChicagoHAI repo):
#   - The explanation IS the plain text string.
#   - SubgroupRule / TreatmentRecommendation are lightweight inference
#     helpers only; they are NOT stored as "the explanation".
# ---------------------------------------------------------------------------

class SubgroupRule(BaseModel):
    feature: str
    operator: Literal[">=", "<=", ">", "<", "==", "!="]
    threshold: Optional[float] = None
    category: Optional[str] = None
    description: str


class TreatmentRecommendation(BaseModel):
    subgroup_rule: SubgroupRule
    recommendation: Literal["treat", "control", "unclear"]
    expected_benefit: Literal["high", "moderate", "low", "none", "harm"]
    rationale: str


class GeneratedExplanation(BaseModel):
    """Minimal explanation produced during the algorithm loop.

    Faithful to the original: the explanation IS the text statement.
    The subgroup_rule is kept only to enable fast rule-based inference
    without an LLM call per sample.
    """
    explanation_statement: str   # The explanation text  ←  this IS the explanation
    treatment_recommendation: TreatmentRecommendation


class GeneratedExplanationSet(BaseModel):
    """LLM response format for batch explanation generation."""
    explanations: List[GeneratedExplanation]
    generation_context: str


class RefinedExplanationSet(BaseModel):
    """LLM response format for refinement from difficult samples."""
    new_explanations: List[GeneratedExplanation]
    refinement_rationale: str
    addresses_patterns: str


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SummaryInformation:
    """Tracks a plain-text explanation and its UCB statistics.

    Faithful to the original ChicagoHAI SummaryInformation:
    the explanation bank key IS the explanation text string.
    `subgroup_rule` is a lightweight companion kept only for rule-based
    inference — it is NOT part of the explanation identity.
    """

    explanation: str                            # Plain text explanation string (faithful to original)
    subgroup_rule: Optional[SubgroupRule] = None  # Inference helper only; not stored in output
    recommendation: str = "unclear"            # Inference helper: "treat" / "control" / "unclear"
    acc: float = 0.0        # running accuracy (0-1), incremental average
    num_visits: int = 0     # number of times the explanation has been tested
    reward: float = 0.0     # UCB reward: acc + alpha * sqrt(log(n) / visits)

    def update_info_if_useful(self, current_sample: int, alpha: float) -> None:
        """Called when the explanation made a correct prediction."""
        self.acc = (self.acc * self.num_visits + 1) / (self.num_visits + 1)
        self.num_visits += 1
        self._update_reward(alpha, current_sample)

    def update_info_if_not_useful(self, current_sample: int, alpha: float) -> None:
        """Called when the explanation made a wrong prediction."""
        self.acc = (self.acc * self.num_visits) / (self.num_visits + 1)
        self.num_visits += 1
        self._update_reward(alpha, current_sample)

    def _update_reward(self, alpha: float, num_examples: int) -> None:
        """UCB reward: acc + alpha * sqrt(log(num_examples) / num_visits)."""
        if self.num_visits > 0 and num_examples > 1:
            self.reward = self.acc + alpha * math.sqrt(
                math.log(num_examples) / self.num_visits
            )

    def to_dict(self):
        """Serialize faithfully: explanation text + stats + lightweight subgroup info."""
        d: dict = {
            "explanation": self.explanation,
            "acc": float(self.acc),
            "num_visits": int(self.num_visits),
            "reward": float(self.reward),
        }
        if self.subgroup_rule is not None:
            d["subgroup_rule"] = {
                "feature": self.subgroup_rule.feature,
                "operator": self.subgroup_rule.operator,
                "threshold": self.subgroup_rule.threshold,
                "description": self.subgroup_rule.description,
            }
        if self.recommendation != "unclear":
            d["recommendation"] = self.recommendation
        return d


# -----------------------------
# Helper functions
# -----------------------------

def get_trial_metadata(trial_name: str) -> dict:
    """Return treatment/outcome/population metadata for known clinical trials."""
    trial_map = {
        "ist3": {
            "treatment": "IV alteplase (recombinant tissue plasminogen activator)",
            "outcome": "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
            "population": "Acute ischemic stroke patients within 6 hours of symptom onset",
        },
        "crash_2": {
            "treatment": "Tranexamic acid (TXA)",
            "outcome": "All-cause mortality at 28 days or in-hospital death",
            "population": "Trauma patients with significant bleeding or at risk of significant hemorrhage",
        },
        "sprint": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Composite of major cardiovascular events",
            "population": "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
        },
        "accord": {
            "treatment": "Intensive blood pressure control (systolic BP target <120 mmHg)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
        },
        "accord_glycemia": {
            "treatment": "Intensive glycemic control (target HbA1c <6.0%)",
            "outcome": "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
            "population": "Adults with type 2 diabetes and high cardiovascular risk",
        },
    }

    trial_lower = trial_name.lower()
    if trial_lower not in trial_map:
        raise ValueError(
            f"Unknown trial: {trial_name}. Supported trials: {', '.join(trial_map.keys())}."
        )
    return trial_map[trial_lower]


def load_trial_data_from_dataset(cohort_name: str, random_state: int = 42, max_samples: Optional[int] = None) -> tuple[Dataset, pd.DataFrame]:
    """Load trial data using Dataset class.

    Args:
        cohort_name: Name of the cohort to load
        random_state: Random state for reproducibility
        max_samples: Maximum number of samples to use (None = use all)

    Returns:
        (Dataset object, DataFrame with training samples)
    """
    dataset = Dataset(cohort_name=cohort_name, random_state=random_state, shuffle=False)

    # Reconstruct DataFrame from training data
    X_tr = dataset.x_train
    W_tr = dataset.w_train
    Y_tr = dataset.y_train

    # Get feature names (excluding treatment and outcome)
    feature_cols = [col for col in dataset.data.columns
                   if col not in [dataset.treatment, dataset.outcome]]

    # Create DataFrame
    df = pd.DataFrame(X_tr, columns=feature_cols)
    df[dataset.treatment] = W_tr
    df[dataset.outcome] = Y_tr

    # Limit to max_samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)
        print(f"Sampled {max_samples} samples from {len(X_tr)} training samples for {cohort_name}")
    else:
        print(f"Loaded {len(df)} training samples for {cohort_name}")

    print(f"  Treatment column: {dataset.treatment}")
    print(f"  Outcome column: {dataset.outcome}")
    print(f"  Features: {len(feature_cols)}")

    return dataset, df


# -----------------------------
# HypoGeniC Algorithm Components
# -----------------------------

def generate_initial_explanations(
    study_context: dict,
    available_features: List[str],
    num_explanations: int,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
    target_features: Optional[int] = None,
    explanations_per_feature: Optional[int] = None,
) -> List[SummaryInformation]:
    """Generate initial explanations using LLM (Algorithm 1, Line 2).

    Returns plain-text explanations as SummaryInformation objects, faithful to
    the original ChicagoHAI implementation where explanations are strings.
    The subgroup_rule is kept only as a lightweight inference helper.
    """

    system_prompt = (
        "You are a clinical research expert generating testable explanations about "
        "treatment effect heterogeneity.\n"
        "\n"
        "For each explanation produce:\n"
        "- explanation_statement: a single clear English sentence describing WHICH patients "
        "benefit (or are harmed) by treatment and WHY (e.g. 'Patients with SBP < 90 mmHg "
        "benefit more from TXA because severe hypotension indicates active haemorrhage.').\n"
        "- treatment_recommendation: the subgroup rule + recommendation (treat/control) + "
        "expected benefit level.\n"
        "\n"
        "Do NOT include sections for mechanism, evidence_basis, or testable_prediction \u2014 "
        "those will be added in post-processing.\n"
        "\n"
        "Focus on clinically meaningful subgroups that could inform treatment decisions."
    )

    instructions = [
        f"Generate {num_explanations} diverse explanations about treatment effect heterogeneity",
        "Each explanation_statement must be a single self-contained English sentence",
        "Base explanations on clinical literature and biological plausibility",
        "Make explanations testable with the available features",
    ]

    if target_features and explanations_per_feature:
        instructions += [
            f"IMPORTANT: Focus on exactly {target_features} features only - select the most clinically important ones",
            f"Generate exactly {explanations_per_feature} explanations per feature (different subgroup rules/thresholds or directions)",
            f"Total: {target_features} features × {explanations_per_feature} explanations = {num_explanations} explanations",
            "Do NOT use more than the specified number of unique features",
        ]
    else:
        instructions.append("Ensure diversity - cover different features and mechanisms")

    user_prompt = {
        "task": "Generate initial clinical explanations",
        "study_context": study_context,
        "available_features": available_features,
        "num_explanations": num_explanations,
        "instructions": instructions,
    }

    try:
        result = _parse_structured(
            client, model_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            GeneratedExplanationSet,
        )
        summaries = []
        for hyp in result.explanations:
            rule = hyp.treatment_recommendation.subgroup_rule
            summaries.append(SummaryInformation(
                explanation=hyp.explanation_statement,
                subgroup_rule=rule,
                recommendation=hyp.treatment_recommendation.recommendation,
            ))
        return summaries
    except Exception as e:
        print(f"Error generating explanations: {e}")
        return []


def inference(
    summary: SummaryInformation,
    sample: pd.Series,
) -> str:
    """Make treatment recommendation based on explanation (Algorithm 1, Line 7).

    Uses the lightweight subgroup_rule attached to the SummaryInformation for
    fast rule-based evaluation, faithful to the original structure where each
    explanation can be applied deterministically to a sample.
    """

    rule = summary.subgroup_rule
    if rule is None:
        return "unclear"

    feature_value = sample.get(rule.feature)

    if pd.isna(feature_value):
        return "unclear"

    # Evaluate subgroup rule
    in_subgroup = False
    try:
        if rule.operator == ">=":
            in_subgroup = float(feature_value) >= float(rule.threshold)
        elif rule.operator == "<=":
            in_subgroup = float(feature_value) <= float(rule.threshold)
        elif rule.operator == ">":
            in_subgroup = float(feature_value) > float(rule.threshold)
        elif rule.operator == "<":
            in_subgroup = float(feature_value) < float(rule.threshold)
        elif rule.operator == "==":
            if rule.category is not None:
                in_subgroup = str(feature_value) == str(rule.category)
            else:
                in_subgroup = float(feature_value) == float(rule.threshold)
        elif rule.operator == "!=":
            if rule.category is not None:
                in_subgroup = str(feature_value) != str(rule.category)
            else:
                in_subgroup = float(feature_value) != float(rule.threshold)
    except (ValueError, TypeError):
        return "unclear"

    # Return recommendation based on whether sample is in subgroup
    if in_subgroup:
        return summary.recommendation
    else:
        # Opposite recommendation for out-of-subgroup
        rec = summary.recommendation
        if rec == "treat":
            return "control"
        elif rec == "control":
            return "treat"
        else:
            return "unclear"


def is_correct_prediction(
    summary: SummaryInformation,
    sample: pd.Series,
    actual_treatment: int,
    actual_outcome: int,
) -> bool:
    """Check if the explanation correctly predicts treatment benefit for this sample."""
    recommendation = inference(summary, sample)

    if recommendation == "unclear":
        return False

    predicted_treat = 1 if recommendation == "treat" else 0

    if predicted_treat == actual_treatment:
        return actual_outcome == 1
    else:
        return actual_outcome == 0


def select_balanced_explanations(
    explanation_bank: Dict[str, SummaryInformation],
    target_features: int,
    explanations_per_feature: int,
) -> tuple:
    """Select top N features and their best explanations from the bank.

    Returns:
        (top_feature_names, feature_groups_dict)
    """
    feature_groups: Dict[str, List[SummaryInformation]] = {}
    for si in explanation_bank.values():
        feature = si.subgroup_rule.feature if si.subgroup_rule else "__unknown__"
        feature_groups.setdefault(feature, []).append(si)

    for feature in feature_groups:
        feature_groups[feature].sort(key=lambda x: x.reward, reverse=True)

    feature_scores = [
        (feature, max(h.reward for h in hyps), sum(h.reward for h in hyps))
        for feature, hyps in feature_groups.items()
    ]
    feature_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_features = [f[0] for f in feature_scores[:target_features]]

    print(f"\nSelecting balanced explanations:")
    print(f"  Top {target_features} features by reward: {top_features}")
    for feature in top_features:
        hyps = feature_groups[feature]
        print(f"  - {feature}: {len(hyps)} explanation(es) (rewards: {[round(h.reward,3) for h in hyps[:explanations_per_feature]]})")

    return top_features, feature_groups


def generate_explanations_for_feature(
    feature: str,
    count_needed: int,
    existing_summaries: List[SummaryInformation],
    study_context: dict,
    available_features: List[str],
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[SummaryInformation]:
    """Generate additional plain-text explanations for a specific feature to fill gaps."""

    existing_rules = [
        si.subgroup_rule.description for si in existing_summaries if si.subgroup_rule
    ]

    system_prompt = (
        "You are a clinical research expert generating additional explanations for a specific feature. "
        "Generate explanations that are DISTINCT from the existing ones "
        "(use different thresholds, operators, or directions of effect).\n"
        "\n"
        "Each explanation_statement must be a single self-contained English sentence.\n"
        "Do NOT include mechanism, evidence_basis, or testable_prediction fields."
    )

    user_prompt = {
        "task": f"Generate {count_needed} additional explanation(es) for feature '{feature}'",
        "study_context": study_context,
        "target_feature": feature,
        "count_needed": count_needed,
        "existing_explanations_for_this_feature": existing_rules,
        "available_features": available_features,
        "instructions": [
            f"Generate exactly {count_needed} new explanation(es) using ONLY the feature '{feature}'",
            "Each explanation MUST set subgroup_rule.feature = '" + feature + "'",
            "Use a different threshold or operator than the existing explanations listed above",
            "Cover complementary perspectives: e.g., high vs low values, moderate range, or opposite direction",
        ],
    }

    try:
        result = _parse_structured(
            client, model_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            GeneratedExplanationSet,
        )
        on_target = [h for h in result.explanations
                     if h.treatment_recommendation.subgroup_rule.feature == feature]
        candidates = on_target if on_target else result.explanations
        for h in candidates:
            h.treatment_recommendation.subgroup_rule.feature = feature
        summaries = []
        for h in candidates[:count_needed]:
            rule = h.treatment_recommendation.subgroup_rule
            summaries.append(SummaryInformation(
                explanation=h.explanation_statement,
                subgroup_rule=rule,
                recommendation=h.treatment_recommendation.recommendation,
            ))
        return summaries
    except Exception as e:
        print(f"  Error generating explanations for feature '{feature}': {e}")
        return []


def build_fallback_explanation_for_feature(
    feature: str,
    ordinal: int,
    study_context: dict,
) -> SummaryInformation:
    """Create a minimal deterministic explanation for a feature as a last-resort filler."""
    treatment = study_context.get("treatment", "treatment")
    outcome = study_context.get("outcome", "outcome")
    statement = (
        f"{feature} may modify the effect of {treatment} on {outcome} "
        f"(placeholder {ordinal}; replace with model-generated content)."
    )
    return SummaryInformation(
        explanation=statement,
        subgroup_rule=SubgroupRule(
            feature=feature,
            operator=">=",
            threshold=0.0,
            category=None,
            description=f"{feature} >= 0.0 (fallback placeholder rule)",
        ),
        recommendation="unclear",
    )


def ensure_balanced_explanations(
    explanation_bank: Dict[str, SummaryInformation],
    target_features: int,
    explanations_per_feature: int,
    study_context: dict,
    available_features: List[str],
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[SummaryInformation]:
    """Build a strictly balanced list of target_features * explanations_per_feature explanations."""
    target_total = target_features * explanations_per_feature

    ranked_features, feature_groups = select_balanced_explanations(
        explanation_bank, target_features, explanations_per_feature
    )

    top_features = list(ranked_features)
    if len(top_features) < target_features:
        missing = target_features - len(top_features)
        backfill = [f for f in available_features if f not in set(top_features)][:missing]
        top_features.extend(backfill)
        print(f"  Backfilling {len(backfill)} feature(s) to reach target coverage: {backfill}")

    selected: List[SummaryInformation] = []
    for feature in top_features:
        existing = feature_groups.get(feature, [])
        have = existing[:explanations_per_feature]
        selected.extend(have)

        gap = explanations_per_feature - len(have)
        if gap > 0:
            print(f"  Gap for '{feature}': have {len(have)}, need {gap} more — generating...")
            generated: List[SummaryInformation] = []
            attempts = 0
            max_attempts = 4
            while len(generated) < gap and attempts < max_attempts:
                attempts += 1
                need = gap - len(generated)
                new_summaries = generate_explanations_for_feature(
                    feature=feature,
                    count_needed=need,
                    existing_summaries=have + generated,
                    study_context=study_context,
                    available_features=available_features,
                    client=client,
                    model_name=model_name,
                )
                if not new_summaries:
                    print(f"    Attempt {attempts}/{max_attempts}: no explanations returned for '{feature}'")
                    continue

                existing_sigs = {
                    (si.subgroup_rule.operator, si.subgroup_rule.threshold,
                     si.subgroup_rule.category, si.subgroup_rule.description)
                    for si in have + generated if si.subgroup_rule
                }
                for si in new_summaries:
                    rule = si.subgroup_rule
                    sig = (rule.operator, rule.threshold, rule.category, rule.description) if rule else None
                    if sig in existing_sigs:
                        continue
                    generated.append(si)
                    if sig:
                        existing_sigs.add(sig)
                    if len(generated) >= gap:
                        break

            if len(generated) < gap:
                before_fill = len(generated)
                for idx in range(len(generated) + 1, gap + 1):
                    generated.append(build_fallback_explanation_for_feature(feature, idx, study_context))
                print(f"    Added {len(generated) - before_fill} fallback explanation(es) for '{feature}'")

            selected.extend(generated)

    print(f"\nFinal balanced selection: {len(selected)}/{target_total} explanations "
          f"across {len(top_features)} features")
    return selected


def generate_new_explanations_from_difficult_samples(
    difficult_samples: List[pd.Series],
    study_context: dict,
    available_features: List[str],
    num_explanations: int,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
) -> List[SummaryInformation]:
    """Generate new plain-text explanations from difficult samples (Algorithm 1, Line 13)."""

    if not difficult_samples:
        return []

    df_difficult = pd.DataFrame(difficult_samples)

    # Summarize patterns in difficult samples
    patterns = []
    for col in available_features:
        if col in df_difficult.columns:
            if df_difficult[col].dtype in ['int64', 'float64']:
                patterns.append({
                    "feature": col,
                    "mean": float(df_difficult[col].mean()) if not df_difficult[col].isna().all() else None,
                    "median": float(df_difficult[col].median()) if not df_difficult[col].isna().all() else None,
                    "min": float(df_difficult[col].min()) if not df_difficult[col].isna().all() else None,
                    "max": float(df_difficult[col].max()) if not df_difficult[col].isna().all() else None,
                })

    system_prompt = (
        "You are a clinical research expert refining explanations based on difficult cases. "
        "Analyze the patterns in samples where current explanations failed and generate "
        "NEW explanations that better explain treatment effect heterogeneity in these cases.\n"
        "\n"
        "Each explanation_statement must be a single self-contained English sentence.\n"
        "Do NOT include mechanism, evidence_basis, or testable_prediction fields.\n"
        "\n"
        "Focus on:\n"
        "1. Features that distinguish difficult samples\n"
        "2. Alternative subgroup definitions\n"
        "3. Novel patterns not covered by previous explanations\n"
        "4. Interactions between features"
    )

    user_prompt = {
        "task": "Generate refined explanations from difficult samples",
        "study_context": study_context,
        "difficult_sample_patterns": patterns,
        "num_difficult_samples": len(difficult_samples),
        "available_features": available_features,
        "num_new_explanations": num_explanations,
        "instructions": [
            f"Analyze patterns in {len(difficult_samples)} samples where explanations failed",
            f"Generate {num_explanations} NEW explanations targeting these difficult cases",
            "Focus on features and thresholds that distinguish these samples",
        ],
    }

    try:
        result = _parse_structured(
            client, model_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            RefinedExplanationSet,
        )
        summaries = []
        for h in result.new_explanations:
            rule = h.treatment_recommendation.subgroup_rule
            summaries.append(SummaryInformation(
                explanation=h.explanation_statement,
                subgroup_rule=rule,
                recommendation=h.treatment_recommendation.recommendation,
            ))
        return summaries
    except Exception as e:
        print(f"Error generating refined explanations: {e}")
        return []


def hypogenic_algorithm(
    data: pd.DataFrame,
    study_context: dict,
    num_init: int,
    top_k: int,
    treatment_col: str,
    outcome_col: str,
    client: OpenAI,
    model_name: str = "gpt-4o-2024-08-06",
    alpha: float = 0.5,
    num_wrong_scale: float = 0.8,
    update_batch_size: int = 5,
    num_explanations_to_update: int = 5,
    update_explanations_per_batch: int = 5,
    target_features: Optional[int] = None,
    explanations_per_feature: Optional[int] = None,
) -> List[SummaryInformation]:
    """
    HypoGeniC algorithm — faithful to the original ChicagoHAI implementation.

    The explanation bank is Dict[str, SummaryInformation] where the *key* is the
    plain-text explanation string, exactly as in the original repo.
    SubgroupRule is kept only as an inference helper on each SummaryInformation
    object; it is NOT saved in the output JSON.

    Key design decisions matching the original:
    - UCB reward: acc + alpha * sqrt(log(current_sample) / num_visits)
    - Adaptive wrong threshold: grows as len(top_k) * (i / n) * num_wrong_scale
    - Set-based wrong example accumulation, trigger when
      |wrong_example_ids| == update_batch_size * num_explanations_to_update
    - DefaultReplace policy: merge new + existing bank, keep top-k by reward

    Returns:
        List of top SummaryInformation objects (explanation text + UCB stats)
    """

    available_features = [col for col in data.columns
                         if col not in [treatment_col, outcome_col]]
    num_train_examples = len(data)

    # Initialize explanation bank: {explanation_text -> SummaryInformation}
    print(f"Generating {num_init} initial explanations...")
    if target_features and explanations_per_feature:
        print(f"  Constrained to {target_features} features × {explanations_per_feature} explanations each")
    initial_summaries = generate_initial_explanations(
        study_context=study_context,
        available_features=available_features,
        num_explanations=num_init,
        client=client,
        model_name=model_name,
        target_features=target_features,
        explanations_per_feature=explanations_per_feature,
    )

    # Bank keyed by explanation text (faithful to original)
    H: Dict[str, SummaryInformation] = {si.explanation: si for si in initial_summaries}

    wrong_example_ids: set = set()

    # Iterate over training examples (mirrors original DefaultUpdate.update loop)
    for idx, (_, sample) in enumerate(data.iterrows()):
        current_sample = idx + 1  # 1-based like the original

        if idx % 100 == 0:
            print(f"Processing sample {idx}/{num_train_examples}...")

        actual_treatment = int(sample[treatment_col])
        actual_outcome = int(sample[outcome_col])

        # Get top-k explanations sorted by UCB reward
        top_k_keys = sorted(H.keys(), key=lambda x: H[x].reward, reverse=True)[:top_k]

        # Adaptive wrong threshold (matches original num_wrong_to_add_bank)
        num_wrong_to_add_bank = (
            len(top_k_keys) * idx / num_train_examples
        ) * num_wrong_scale if num_wrong_scale > 0 else 0

        num_wrong_explanations = 0

        for h_text in top_k_keys:
            si = H[h_text]
            correct = is_correct_prediction(si, sample, actual_treatment, actual_outcome)
            if correct:
                si.update_info_if_useful(current_sample, alpha)
            else:
                si.update_info_if_not_useful(current_sample, alpha)
                num_wrong_explanations += 1

        if num_wrong_explanations >= num_wrong_to_add_bank or len(top_k_keys) == 0:
            wrong_example_ids.add(idx)

        # When wrong-example set is full, generate new explanations (DefaultUpdate style)
        if len(wrong_example_ids) == update_batch_size * num_explanations_to_update:
            print(
                f"\nGenerating explanations from {len(wrong_example_ids)} difficult examples "
                f"(sample {current_sample}/{num_train_examples})..."
            )

            difficult_samples = [data.iloc[i] for i in wrong_example_ids]
            new_hyp_bank: Dict[str, SummaryInformation] = {}

            for _ in range(num_explanations_to_update):
                new_summaries = generate_new_explanations_from_difficult_samples(
                    difficult_samples=difficult_samples,
                    study_context=study_context,
                    available_features=available_features,
                    num_explanations=update_explanations_per_batch,
                    client=client,
                    model_name=model_name,
                )
                for si in new_summaries:
                    new_hyp_bank[si.explanation] = si

            wrong_example_ids = set()

            # DefaultReplace: merge new + existing bank, keep top-k by reward
            merged = {**new_hyp_bank, **H}
            H = dict(
                sorted(merged.items(), key=lambda x: x[1].reward, reverse=True)[:top_k]
            )

            print(f"  Updated explanation bank size: {len(H)}")

    print(f"\nCompleted HypoGeniC algorithm. Final bank size: {len(H)}")

    if target_features and explanations_per_feature:
        final_explanations = ensure_balanced_explanations(
            H,
            target_features,
            explanations_per_feature,
            study_context=study_context,
            available_features=available_features,
            client=client,
            model_name=model_name,
        )
        expected_total = target_features * explanations_per_feature
        if len(final_explanations) != expected_total:
            raise RuntimeError(
                f"Expected exactly {expected_total} explanations "
                f"({target_features} features × {explanations_per_feature}), "
                f"but got {len(final_explanations)}."
            )
    else:
        final_explanations = sorted(H.values(), key=lambda x: x.reward, reverse=True)[:top_k]

    # Gap-fill for unconstrained mode
    if not (target_features and explanations_per_feature):
        while len(final_explanations) < top_k:
            gap = top_k - len(final_explanations)
            print(f"  Gap-fill: have {len(final_explanations)}, need {gap} more explanations...")
            extra = generate_initial_explanations(
                study_context=study_context,
                available_features=available_features,
                num_explanations=gap * 2,
                client=client,
                model_name=model_name,
            )
            existing_texts = {h.explanation for h in final_explanations}
            for si in extra:
                if si.explanation not in existing_texts and len(final_explanations) < top_k:
                    final_explanations.append(si)
                    existing_texts.add(si.explanation)
            if not extra:
                print(f"  Warning: could not fill gap, returning {len(final_explanations)} explanations")
                break

    print(f"  Returning {len(final_explanations)} explanations")
    return final_explanations


def main():
    parser = argparse.ArgumentParser(
        description="HypoGeniC: Iterative explanation generation for clinical trials"
    )

    # Data arguments
    parser.add_argument(
        "--trial_name",
        required=True,
        help="Trial name (ist3, crash_2, sprint, accord) to load data from Dataset class",
    )
    parser.add_argument(
        "--out_json",
        required=True,
        help="Output path for generated explanations JSON",
    )

    # Optional arguments
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for data splitting",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of training samples to use (default: 200)",
    )

    # HypoGeniC algorithm parameters
    parser.add_argument(
        "--num_init",
        type=int,
        default=20,
        help="Number of initial explanations to generate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=FIXED_TOTAL_EXPLANATIONS,
        help=f"Number of top explanations to maintain (fixed at {FIXED_TOTAL_EXPLANATIONS})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="UCB exploration constant (default: 0.5, matches original HypoGeniC)",
    )
    parser.add_argument(
        "--num_wrong_scale",
        type=float,
        default=0.8,
        help="Scale for adaptive wrong-prediction threshold (default: 0.8)",
    )
    parser.add_argument(
        "--update_batch_size",
        type=int,
        default=5,
        help="Number of wrong examples to accumulate per update batch (default: 5)",
    )
    parser.add_argument(
        "--num_explanations_to_update",
        type=int,
        default=5,
        help="Generation rounds per update batch (default: 5)",
    )
    parser.add_argument(
        "--update_explanations_per_batch",
        type=int,
        default=5,
        help="Explanations generated per generation round (default: 5)",
    )
    parser.add_argument(
        "--target_features",
        type=int,
        default=FIXED_TARGET_FEATURES,
        help=f"Number of unique features to focus on (fixed at {FIXED_TARGET_FEATURES})",
    )
    parser.add_argument(
        "--explanations_per_feature",
        type=int,
        default=FIXED_EXPLANATIONS_PER_FEATURE,
        help=f"Number of explanations per feature (fixed at {FIXED_EXPLANATIONS_PER_FEATURE})",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name",
    )

    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "openrouter", "medgemma"],
        help="API provider to use (default: openai). Use 'medgemma' for local MedGemma model.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Optional API base URL override",
    )
    parser.add_argument(
        "--medgemma_model",
        type=str,
        default="google/medgemma-27b-text-it",
        help="HuggingFace model ID for MedGemma (default: google/medgemma-27b-text-it).",
    )
    parser.add_argument(
        "--medgemma_device",
        type=str,
        default="cuda",
        help="Device for MedGemma inference (default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed index used to place outputs under seed_<seed> subfolder (default: 0)",
    )

    args = parser.parse_args()

    # Enforce fixed balanced output: exactly 5 features × 3 explanations = 15 explanations.
    if args.top_k != FIXED_TOTAL_EXPLANATIONS:
        print(
            f"Overriding --top_k={args.top_k} to fixed value {FIXED_TOTAL_EXPLANATIONS} "
            f"({FIXED_TARGET_FEATURES}x{FIXED_EXPLANATIONS_PER_FEATURE})."
        )
    if args.target_features != FIXED_TARGET_FEATURES:
        print(
            f"Overriding --target_features={args.target_features} to fixed value {FIXED_TARGET_FEATURES}."
        )
    if args.explanations_per_feature != FIXED_EXPLANATIONS_PER_FEATURE:
        print(
            f"Overriding --explanations_per_feature={args.explanations_per_feature} to fixed value {FIXED_EXPLANATIONS_PER_FEATURE}."
        )

    args.top_k = FIXED_TOTAL_EXPLANATIONS
    args.target_features = FIXED_TARGET_FEATURES
    args.explanations_per_feature = FIXED_EXPLANATIONS_PER_FEATURE

    resolved_out_json = resolve_seeded_output_path(args.out_json, args.seed)
    if resolved_out_json != args.out_json:
        print(f"Resolved output path with seed folder: {resolved_out_json}")
    args.out_json = resolved_out_json

    # Load .env (if present), then resolve API key
    load_local_env()

    # Get API key and build client
    if args.api_provider == "medgemma":
        from src.agent_utils import MedGemmaClient
        client = MedGemmaClient(
            model_name=args.medgemma_model,
            device=args.medgemma_device,
        )
    elif args.api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it in your environment or .env file."
            )
        base_url = args.api_base_url or "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your environment "
                "or in a local .env file."
            )
        kwargs = {"api_key": api_key}
        if args.api_base_url:
            kwargs["base_url"] = args.api_base_url
        client = OpenAI(**kwargs)

    # Get trial metadata and load data using Dataset class
    trial_meta = get_trial_metadata(args.trial_name)
    treatment = trial_meta["treatment"]
    outcome = trial_meta["outcome"]
    population = trial_meta["population"]

    # Load data from Dataset class
    print(f"Loading data for {args.trial_name}...")
    dataset, data = load_trial_data_from_dataset(
        args.trial_name,
        args.random_state,
        max_samples=args.max_samples
    )

    # Get treatment and outcome column names from dataset
    treatment_col = dataset.treatment
    outcome_col = dataset.outcome

    # Prepare study context
    study_context = {
        "dataset": args.trial_name,
        "treatment": treatment,
        "outcome": outcome,
        "population": population,
        "sample_size": len(data),
        "method": "HypoGeniC",
    }

    # Run HypoGeniC algorithm
    print("=" * 80)
    print("Running HypoGeniC Algorithm")
    print("=" * 80)

    # Derive num_init from target_features * explanations_per_feature if constrained
    num_init = args.num_init
    if args.target_features and args.explanations_per_feature:
        constrained_total = args.target_features * args.explanations_per_feature
        # Generate more initially to ensure enough diversity, then select balanced
        num_init = max(args.num_init, constrained_total * 2)
        print(f"Constrained mode: {args.target_features} features × {args.explanations_per_feature} explanations = {constrained_total} total")
        print(f"Generating {num_init} initial explanations to ensure coverage")

    final_explanations = hypogenic_algorithm(
        data=data,
        study_context=study_context,
        num_init=num_init,
        top_k=args.top_k,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        client=client,
        model_name=args.model,
        alpha=args.alpha,
        num_wrong_scale=args.num_wrong_scale,
        update_batch_size=args.update_batch_size,
        num_explanations_to_update=args.num_explanations_to_update,
        update_explanations_per_batch=args.update_explanations_per_batch,
        target_features=args.target_features,
        explanations_per_feature=args.explanations_per_feature,
    )

    print("\n" + "=" * 80)
    print("HypoGeniC Algorithm Completed")
    print("=" * 80)

    # Explanation text is the key, stats are the value.
    # Faithful to the original ChicagoHAI design: no LLM expansion step.
    output = {
        "method": "HypoGeniC",
        "study_context": study_context,
        "algorithm_parameters": {
            "num_init": args.num_init,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "num_wrong_scale": args.num_wrong_scale,
            "update_batch_size": args.update_batch_size,
            "num_explanations_to_update": args.num_explanations_to_update,
            "update_explanations_per_batch": args.update_explanations_per_batch,
        },
        # Faithful to original HypoGeniC: explanations stored as dict keyed by plain text
        "explanations": {si.explanation: si.to_dict() for si in final_explanations},
        "summary": {
            "total_explanations": len(final_explanations),
            "avg_reward": float(np.mean([si.reward for si in final_explanations])),
            "avg_accuracy": float(np.mean([si.acc for si in final_explanations])),
        },
    }

    # Save internal format output
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote HypoGeniC explanations to: {args.out_json}")

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"HypoGeniC Algorithm Summary")
    print(f"{'=' * 80}")
    print(f"\nTop {len(final_explanations)} Explanations:")
    for i, si in enumerate(final_explanations, 1):
        truncated = si.explanation[:80] + ("..." if len(si.explanation) > 80 else "")
        print(f"  {i}. {truncated}")
        print(f"     Reward: {si.reward:.4f}, Acc: {si.acc:.2%}, Visits: {si.num_visits}")

if __name__ == "__main__":
    main()
