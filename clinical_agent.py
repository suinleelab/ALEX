#!/usr/bin/env python3
"""clinical_agent.py

Generate clinical research explanations from a Shapley summary JSON using OpenAI's API
or OpenRouter, returning structured JSON output via Structured Outputs (Pydantic schema).
Optionally verify explanations with a separate verifier model.

Requires:
  pip install openai pydantic
  pip install transformers torch  # for MedGemma local inference

API Keys:
  - OpenAI: Set OPENAI_API_KEY environment variable
  - OpenRouter: Set OPENROUTER_API_KEY environment variable

Example:
  python clinical_agent.py \
    --shap_json results/ist3/baseline_shapley_value_sampling_summary_shuffle_True_RLearner_zero_baseline_True.json \
    --out_json results/ist3/explanations_baseline_shapley_RLearner.json \
    --trial_name ist3 \
    --n_features 15 \
    --n_explanations 8

  Or with OpenRouter:
  export OPENROUTER_API_KEY=your_key_here
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/explanations.json \
    --trial_name ist3 \
    --model anthropic/claude-3.5-sonnet \
        --api_provider openrouter

  Or with local MedGemma:
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/explanations.json \
    --trial_name ist3 \
    --api_provider medgemma

  Or with manual metadata:
  python clinical_agent.py \
    --shap_json results/custom/shap_summary.json \
    --out_json results/custom/explanations.json \
    --treatment "Custom treatment" \
    --outcome "Custom outcome" \
    --population "Custom population" \
    --n_features 15 \
    --n_explanations 8

    With verification:
  python clinical_agent.py \
    --shap_json results/ist3/shap_summary.json \
    --out_json results/ist3/explanations.json \
    --trial_name ist3 \
        --enable_verifier
"""

import argparse
import json
import os
import re
from typing import List, Literal, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field
from src.agent_utils import (
    get_model_client,
    get_trial_metadata,
    load_top_features,
    load_local_env,
    resolve_seeded_output_path,
    search_and_extract_article,
    write_json_file,
)

# -----------------------------
# Structured output schema
# -----------------------------


class SubgroupDefinition(BaseModel):
    feature: str = Field(
        ..., description="Feature name used to define a subgroup/effect modifier."
    )
    split_rule: str = Field(
        ...,
        description="Human-readable subgroup rule (e.g., 'age >= 75', 'lactate > 2').",
    )
    notes: Optional[str] = Field(
        None, description="Any nuance about encoding, bins, or clinical interpretation."
    )


class ValidationPlan(BaseModel):
    analyses: List[str] = Field(
        ...,
        description=(
            "Concrete follow-up analyses to validate the explanation, e.g., "
            "DR estimator within strata, interaction term, sensitivity checks."
        ),
    )
    negative_controls: Optional[List[str]] = Field(
        None, description="Optional negative control ideas / falsification tests."
    )
    robustness: Optional[List[str]] = Field(
        None,
        description="Optional robustness checks (baseline sensitivity, subgroup stability, etc.).",
    )


class ClinicalExplanation(BaseModel):
    title: str = Field(..., description="Short, specific explanation title.")
    explanation: str = Field(
        ...,
        description="A testable statement about treatment-effect heterogeneity or subgroup benefit/harm.",
    )
    expected_direction: Literal[
        "higher_benefit", "lower_benefit", "higher_harm", "lower_harm", "ambiguous"
    ] = Field(
        ...,
        description="Direction of effect modification relative to the subgroup rule.",
    )
    subgroup: SubgroupDefinition
    rationale: List[str] = Field(
        ...,
        description="Bullet-like rationales grounded in features + plausible clinical mechanism.",
    )
    key_features: List[str] = Field(
        ...,
        description="Top features (from Shapley summary) that support this explanation.",
    )
    confounders_and_bias_risks: List[str] = Field(
        ...,
        description="Potential confounding, bias, measurement error, or collider risks.",
    )
    validation: ValidationPlan
    caveats: Optional[List[str]] = Field(
        None,
        description="Any cautions about interpretation (attribution ≠ causality, encoding, baseline sensitivity).",
    )


class ExplanationSet(BaseModel):
    dataset: str
    learner: str
    treatment: str
    outcome: str
    population: str
    source_explainer: str
    explanations: List[ClinicalExplanation]


class ExplanationIssue(BaseModel):
    type: Literal[
        "overclaiming_causality",
        "not_testable",
        "subgroup_rule_ambiguous",
        "direction_not_supported",
        "feature_not_in_evidence",
        "confounding_missing",
        "validation_plan_weak",
        "clinical_implausible",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    message: str


class ExplanationReview(BaseModel):
    title: str
    verdict: Literal["approve", "revise", "reject"]
    issues: List[ExplanationIssue] = Field(default_factory=list)
    suggested_edits: Optional[str] = Field(
        None,
        description="Concrete rewrite guidance or a rewritten version for the explanation text/subgroup rule.",
    )
    evidence_alignment: Literal["strong", "moderate", "weak"] = "moderate"
    confidence: Literal["low", "medium", "high"] = "medium"


class VerificationOutput(BaseModel):
    overall_verdict: Literal["approve", "revise", "reject"]
    summary: str
    per_explanation: List[ExplanationReview]
    revised: Optional[ExplanationSet] = Field(
        None,
        description="If overall_verdict is revise, provide a corrected ExplanationSet.",
    )


class MechanismExplanation(BaseModel):
    description: str = Field(
        ..., description="An explanation of how this feature modifies the treatment effect."
    )

class FeatureExplanation(BaseModel):
    feature_name: str = Field(..., description="Name of the feature")
    importance_rank: int = Field(..., description="Rank by SHAP importance (1=most important)")
    shap_value: float = Field(..., description="Mean absolute SHAP value")
    effect_direction: Literal["positive", "negative", "bidirectional", "unclear"] = Field(
        ..., description="Direction of feature's influence on treatment effect"
    )
    clinical_interpretation: str = Field(
        ..., description="What this feature represents clinically"
    )
    why_important: str = Field(
        ...,
        description="Why this feature is important for treatment effect heterogeneity"
    )
    mechanisms: List[MechanismExplanation] = Field(
        ..., description="Possible mechanisms explaining importance"
    )
    subgroup_implications: str = Field(
        ...,
        description="What subgroups this suggests might have differential treatment effects"
    )
    validation_suggestions: List[str] = Field(
        ..., description="How to test these explanations"
    )
    caveats: List[str] = Field(
        ..., description="Limitations and alternative explanations"
    )


class FeatureExplanationsSet(BaseModel):
    dataset: str
    model: str
    summary: str = Field(
        ..., description="Overall summary of feature importance patterns"
    )
    feature_explanations: List[FeatureExplanation]
    cross_feature_patterns: Optional[str] = Field(
        None, description="Patterns across multiple features"
    )


class FeatureExplanationIssue(BaseModel):
    type: Literal[
        "mechanism_implausible",
        "clinical_interpretation_wrong",
        "effect_direction_unsupported",
        "validation_plan_weak",
        "missing_caveats",
        "overclaiming_certainty",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    message: str


class MechanismReview(BaseModel):
    verdict: Literal["approve", "revise", "reject"]
    plausibility: Literal["high", "moderate", "low", "implausible"]
    comments: str = Field(..., description="Detailed comments on this explanation")
    suggested_revision: Optional[str] = Field(
        None, description="Suggested revision for the explanation if needed"
    )


class FeatureExplanationReview(BaseModel):
    feature_name: str
    verdict: Literal["approve", "revise", "reject"]
    issues: List[FeatureExplanationIssue] = Field(default_factory=list)
    per_mechanism: List[MechanismReview] = Field(
        default_factory=list,
        description="Review of each individual mechanism explanation for this feature"
    )
    suggested_edits: Optional[str] = Field(
        None,
        description="Concrete suggestions for improving this feature explanation.",
    )
    mechanism_quality: Literal["strong", "moderate", "weak"] = "moderate"
    confidence: Literal["low", "medium", "high"] = "medium"


class FeatureVerificationOutput(BaseModel):
    overall_verdict: Literal["approve", "revise", "reject"]
    summary: str
    per_feature: List[FeatureExplanationReview]
    revised: Optional[FeatureExplanationsSet] = Field(
        None,
        description="If overall_verdict is revise, provide a corrected FeatureExplanationsSet.",
    )


# -----------------------------
# Helpers
# -----------------------------

_T = TypeVar("_T")


def _coerce_to_str(val) -> str:
    """Coerce a value to a string.  If *val* is a dict, try to extract the
    most informative single value; otherwise fall back to ``str(val)``."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        # Pick the first long-ish string value, or join all values
        str_vals = [str(v) for v in val.values() if v]
        return " ".join(str_vals) if str_vals else str(val)
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def _normalize_explanations_dict(data: dict) -> dict:
    """Normalize a raw LLM-generated dict to match FeatureExplanationsSet schema.

    Handles common simplifications made by non-OpenAI models:
    - mechanisms as list of strings → list of {description: str} objects
    - effect_direction free text → nearest allowed literal
    - missing optional-ish fields → sensible defaults
    - dict / list in string fields → coerced to str
    """
    _DIRECTION_MAP = {
        "pos": "positive", "neg": "negative",
        "bi": "bidirectional", "unclear": "unclear", "neutral": "unclear",
        "none": "unclear",
    }

    def _fix_direction(val: str) -> str:
        parts = (val or "").lower().split()
        if not parts:
            return "unclear"
        v = parts[0].rstrip(".,;:(")
        for prefix, canonical in _DIRECTION_MAP.items():
            if v.startswith(prefix):
                return canonical
        return "unclear"

    # --- Top-level string fields: coerce dicts / lists → str ---
    for str_field in ("dataset", "model", "summary"):
        if str_field in data and not isinstance(data[str_field], str):
            data[str_field] = _coerce_to_str(data[str_field])
    # Provide defaults for required top-level fields
    data.setdefault("dataset", "unknown")
    data.setdefault("model", "unknown")
    data.setdefault("summary", "")

    # --- feature_explanations missing: try to rescue from nested wrapper keys ---
    def _hoist_list(src: dict, dst: dict) -> bool:
        """Try named aliases, then any list-of-dicts value. Returns True if found."""
        _fh_aliases = ("feature_explanations", "explanations", "features", "feature_list",
                       "feature_analyses", "individual_features", "feature_importance",
                       "feature_explanation_list", "individual_explanations")
        for alias in _fh_aliases:
            if isinstance(src.get(alias), list):
                dst["feature_explanations"] = src[alias]
                return True
        # fallback: first list of dicts in src
        for val in src.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                dst["feature_explanations"] = val
                return True
        return False

    if "feature_explanations" not in data or not isinstance(data.get("feature_explanations"), list):
        # 1. Nested inside "study_context" or similar wrapper dicts
        for wrapper_key in ("study_context", "context", "analysis", "result", "output",
                            "response", "analysis_results", "feature_analysis"):
            wrapper = data.get(wrapper_key)
            if isinstance(wrapper, dict):
                if _hoist_list(wrapper, data):
                    # also hoist metadata if missing at top level
                    for meta in ("dataset", "model", "summary"):
                        if data[meta] in ("unknown", "") and meta in wrapper:
                            data[meta] = wrapper[meta]
                    break
                # wrapper itself looks like the FeatureExplanationsSet — merge it up
                if "dataset" in wrapper or "summary" in wrapper:
                    for k, v in wrapper.items():
                        data.setdefault(k, v)
                    if "feature_explanations" in data:
                        break
        # 2. Common top-level aliases (LLM used wrong key name)
        if "feature_explanations" not in data:
            _hoist_list(data, data)
        # 3. Last resort: if still missing, log the top-level keys for debugging
        if "feature_explanations" not in data:
            print(
                f"[DEBUG] _normalize_explanations_dict: 'feature_explanations' still missing. "
                f"Top-level keys: {list(data.keys())}. "
                + (f"'study_context' keys: {list(data['study_context'].keys())}"
                   if isinstance(data.get('study_context'), dict) else "")
            )

    for fh in data.get("feature_explanations", []):
        # LLM aliases for feature name
        if "feature_name" not in fh:
            for alias in ("feature", "feature_raw", "feature_label", "name", "feature_id"):
                if alias in fh:
                    fh["feature_name"] = fh.pop(alias)
                    break

        # mechanisms: missing → empty list (gap-fill will populate later)
        fh.setdefault("mechanisms", [])

        # mechanisms: list[str] → list[{description}]
        mechs = fh.get("mechanisms", [])
        if mechs and isinstance(mechs[0], str):
            fh["mechanisms"] = [{"description": m} for m in mechs]
        # mechanisms: list[dict] with wrong key (e.g. "explanation") → {description}
        elif mechs and isinstance(mechs[0], dict) and "description" not in mechs[0]:
            normalized_mechs = []
            for m in mechs:
                desc = m.get("description") or m.get("explanation") or m.get("text") or str(m)
                normalized_mechs.append({"description": desc})
            fh["mechanisms"] = normalized_mechs

        # effect_direction: free text → literal; missing → "unclear"
        if "effect_direction" not in fh or not fh["effect_direction"]:
            fh["effect_direction"] = "unclear"
        else:
            fh["effect_direction"] = _fix_direction(fh["effect_direction"])

        # fill required fields with defaults if absent
        fh.setdefault("importance_rank", 0)
        fh.setdefault("shap_value", 0.0)
        fh.setdefault("clinical_interpretation", "")
        fh.setdefault("why_important", "")
        fh.setdefault("subgroup_implications", "")
        fh.setdefault("validation_suggestions", [])
        fh.setdefault("caveats", [])

        # Coerce non-string values in string fields → str
        for str_field in ("feature_name", "clinical_interpretation", "why_important", "subgroup_implications"):
            if str_field in fh and not isinstance(fh[str_field], str):
                fh[str_field] = _coerce_to_str(fh[str_field])

        # coerce string → single-element list for list fields
        for list_field in ("validation_suggestions", "caveats"):
            if isinstance(fh.get(list_field), str):
                fh[list_field] = [fh[list_field]]

    # cross_feature_patterns: list → joined string
    cfp = data.get("cross_feature_patterns")
    if isinstance(cfp, list):
        data["cross_feature_patterns"] = " ".join(str(x) for x in cfp)
    elif cfp is not None and not isinstance(cfp, str):
        data["cross_feature_patterns"] = _coerce_to_str(cfp)

    return data


def _normalize_verification_dict(data: dict) -> dict:
    """Normalize a raw LLM dict to match FeatureVerificationOutput schema."""
    _VERDICT3 = {"app": "approve", "rev": "revise", "rej": "reject"}
    _PLAUS = {"high": "high", "mod": "moderate", "low": "low", "imp": "implausible"}
    _QUALITY = {"str": "strong", "mod": "moderate", "wea": "weak"}
    _CONF = {"low": "low", "med": "medium", "hig": "high"}

    def _fix3(val: str, mapping: dict, default: str) -> str:
        parts = (val or "").lower().split()
        if not parts:
            return default
        v = parts[0].rstrip(".,;:(").strip("-_ ")
        for prefix, canonical in mapping.items():
            if v.startswith(prefix):
                return canonical
        return default

    # overall_verdict: free text → literal
    data["overall_verdict"] = _fix3(data.get("overall_verdict", ""), _VERDICT3, "revise")

    # summary: required string — default to empty string if missing
    if not data.get("summary"):
        data["summary"] = ""

    # per_feature: required list — default to empty list if missing/None
    if data.get("per_feature") is None:
        data["per_feature"] = []

    # per_feature: dict keyed by feature_name → list
    pf = data.get("per_feature", [])
    if isinstance(pf, dict):
        items = []
        for fname, val in pf.items():
            if isinstance(val, dict):
                val.setdefault("feature_name", fname)
                items.append(val)
            else:
                items.append({"feature_name": fname, "verdict": "approve"})
        data["per_feature"] = items
        pf = items

    for review in pf:
        if not isinstance(review, dict):
            continue
        review.setdefault("feature_name", "unknown")
        review["verdict"] = _fix3(review.get("verdict", ""), _VERDICT3, "approve")
        review["mechanism_quality"] = _fix3(review.get("mechanism_quality", ""), _QUALITY, "moderate")
        review["confidence"] = _fix3(review.get("confidence", ""), _CONF, "medium")
        review.setdefault("issues", [])
        review.setdefault("per_mechanism", [])
        # per_mechanism normalisation
        for mr in review.get("per_mechanism", []):
            if not isinstance(mr, dict):
                continue
            mr["verdict"] = _fix3(mr.get("verdict", ""), _VERDICT3, "approve")
            mr["plausibility"] = _fix3(mr.get("plausibility", ""), _PLAUS, "moderate")
            mr.setdefault("comments", "")

    # revised field: if present, normalise as explanations set
    if isinstance(data.get("revised"), dict):
        data["revised"] = _normalize_explanations_dict(data["revised"])

    return data


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair JSON truncated by token limits.

    Strips trailing incomplete strings/values, then closes any open brackets
    and braces so json.loads can succeed on partial output.
    """
    # Remove trailing incomplete string (un-closed quote)
    text = re.sub(r',?\s*"[^"]*$', '', text)
    # Remove trailing key without value  e.g.  , "some_key":
    text = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', text)
    # Remove trailing comma
    text = text.rstrip().rstrip(',')
    # Fix missing commas between strings: "..." "..." → "...", "..."
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    # Fix missing commas between } and {
    text = re.sub(r'\}\s*\{', '}, {', text)
    # Replay the open/close sequence to determine the correct closing order.
    stack = []
    for ch in text:
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    # Close in reverse order of opening (innermost first).
    for opener in reversed(stack):
        text += ']' if opener == '[' else '}'
    return text


def _extract_json_from_content(content: str, response_format: Type[_T]) -> _T:
    """Extract and validate a Pydantic model from raw LLM text content."""
    # Strip <think>...</think> reasoning blocks (Qwen3 / DeepSeek-R1 style)
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.DOTALL).strip()
    # Unwrap markdown code fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    # Find outermost JSON object (prefer object over array since all our schemas are objects)
    obj_match = re.search(r"\{[\s\S]*\}", content, flags=re.DOTALL)
    arr_match = re.search(r"\[[\s\S]*\]", content, flags=re.DOTALL)
    if obj_match and arr_match:
        content = obj_match.group(0) if obj_match.start() <= arr_match.start() else arr_match.group(0)
    elif obj_match:
        content = obj_match.group(0)
    elif arr_match:
        content = arr_match.group(0)
    # Strip trailing commas before ] or } (common Qwen issue)
    content = re.sub(r',\s*([\]\}])', r'\1', content)
    # Parse to dict and normalise before Pydantic validation
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open brackets/braces
        repaired = _repair_truncated_json(content)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            # last resort: let Pydantic try and surface a useful error
            return response_format.model_validate_json(content)
    if isinstance(data, dict) and response_format is FeatureExplanationsSet:
        data = _normalize_explanations_dict(data)
    elif isinstance(data, dict) and response_format is FeatureVerificationOutput:
        data = _normalize_verification_dict(data)
    return response_format.model_validate(data)


def _token_limit_kwarg(model_name: str, client: OpenAI, limit: int = 16384) -> dict:
    """Return the correct max-token parameter for the model/provider.

    Newer OpenAI models (gpt-5*, o1*, o3*, o4*) require 'max_completion_tokens'.
    Older models, OpenRouter, and MedGemma still use 'max_tokens'.
    """
    is_openrouter = getattr(client, '_base_url', None) and 'openrouter' in str(client._base_url)
    is_medgemma = getattr(client, '_is_medgemma', False)
    model_lower = model_name.lower()

    # Use max_completion_tokens for native OpenAI models that require it
    _new_style = ('gpt-5', 'gpt-4.1', 'o1', 'o3', 'o4')
    if not is_openrouter and not is_medgemma and any(model_lower.startswith(p) for p in _new_style):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def _parse_structured(
    client: OpenAI,
    model_name: str,
    messages: list,
    response_format: Type[_T],
) -> _T:
    """Call beta.chat.completions.parse and return the parsed Pydantic model.

    Falls back to a plain chat.completions.create call with manual JSON
    extraction for providers (e.g. OpenRouter + Qwen) that do not support
    OpenAI-style structured outputs or return thinking blocks before JSON.
    """
    is_openrouter = getattr(client, '_base_url', None) and 'openrouter' in str(client._base_url)
    is_medgemma = getattr(client, '_is_medgemma', False)
    # Models known to NOT support OpenAI-style structured outputs via OpenRouter
    _no_structured = ('qwen', 'deepseek', 'llama', 'mistral', 'mixtral')
    model_lower = model_name.lower()
    skip_parse = is_medgemma or (is_openrouter and any(t in model_lower for t in _no_structured))

    # --- Primary: try OpenAI structured outputs (skip for unsupported models) ---
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
            # .parsed is None but content may have raw JSON
            content = completion.choices[0].message.content or ""
            return _extract_json_from_content(content, response_format)
        except Exception as e:
            print(f"[_parse_structured] structured parse failed, falling back to plain completion: {e}")

    # --- Fallback (or primary for OpenRouter): plain create with JSON instructions ---
    schema = response_format.model_json_schema()
    top_fields = list(schema.get("properties", {}).keys())
    fields_hint = ", ".join(f'"{f}"' for f in top_fields)
    augmented = list(messages)
    augmented[0] = dict(augmented[0])
    augmented[0]["content"] = (
        augmented[0]["content"]
        + f"\n\nIMPORTANT: Your entire response must be a single valid JSON object with "
        f"top-level keys: {fields_hint}. Do not include any explanation, markdown, or "
        f"schema definitions — only the filled JSON instance."
    )
    # Disable thinking tokens for models that support it (saves ~1-3K output tokens)
    extra_body: dict = {}
    if is_openrouter:
        extra_body["reasoning"] = {"effort": "none"}
    kwargs: dict = dict(
        model=model_name,
        messages=augmented,
        **_token_limit_kwarg(model_name, client),
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    fallback = client.chat.completions.create(**kwargs)
    content = fallback.choices[0].message.content or ""
    return _extract_json_from_content(content, response_format)


def _fill_mechanisms_for_feature(
    feature_explanation: "FeatureExplanation",
    count_needed: int,
    study_context: dict,
    client: OpenAI,
    model_name: str,
) -> List["MechanismExplanation"]:
    """Request additional explanations for a feature that was under-generated."""

    existing_descriptions = [m.description for m in (feature_explanation.mechanisms or [])]

    class _MechanismList(BaseModel):
        mechanisms: List[MechanismExplanation]

    system = (
        "You are a clinical research expert. Generate additional distinct explanations "
        "for a specific feature explaining treatment effect heterogeneity. "
        "Each explanation must be different from the existing ones."
    )
    user_prompt = {
        "task": f"Generate {count_needed} additional explanation(es) for feature '{feature_explanation.feature_name}'",
        "study_context": study_context,
        "feature_name": feature_explanation.feature_name,
        "clinical_interpretation": feature_explanation.clinical_interpretation,
        "existing_explanations": existing_descriptions,
        "count_needed": count_needed,
        "instructions": [
            f"Generate exactly {count_needed} new explanation(es) DISTINCT from the existing ones above",
            "Each must explain how this feature modifies the TREATMENT EFFECT (not just prognosis)",
            "Each description must be framed as a treatment-response INTERACTION (who benefits more/less from treatment and why), "
            "not as a prognostic statement about the feature's effect on outcome regardless of treatment. "
            "PRECISION: avoid oversimplified monotone claims — if the interaction depends on additional moderators "
            "(e.g., age × comorbidity, eGFR × diabetes duration), describe the joint subgroup precisely rather than a blanket directional statement.",
            "ABSOLUTE CATE: SHAP values measure absolute treatment effect differences (absolute CATE), not relative risk ratios. "
            "Frame mechanisms as ABSOLUTE benefit modifiers: which subgroup gains more in absolute terms (greater ARR, lower NNT). "
            "Do not rely solely on relative efficacy claims. "
            "Example (good): 'Patients with elevated baseline LDL cholesterol derive greater ABSOLUTE benefit from statin therapy "
            "because their higher baseline event rate translates even a modest relative risk reduction into a larger absolute risk reduction per patient treated.'",
        ],
    }
    try:
        parsed = _parse_structured(
            client, model_name,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            _MechanismList,
        )
        return (parsed.mechanisms or [])[:count_needed]
    except Exception as e:
        print(f"    Error in explanation gap-fill for '{feature_explanation.feature_name}': {e}")
        return []


def generate_feature_explanations(
    top_features: List[dict],
    study_context: dict,
    client: OpenAI,
    model_name: str = "gpt-5-mini"
) -> Optional[FeatureExplanationsSet]:

    feature_label_map = {}
    for item in top_features:
        raw_name = (item or {}).get("feature_raw")
        mapped_name = (item or {}).get("feature")
        if raw_name and mapped_name:
            feature_label_map[str(raw_name).strip().lower()] = mapped_name
            feature_label_map[str(mapped_name).strip().lower()] = mapped_name

    # 1. CORE DEFINITIONS (Strict Definitions to prevent drift)
    definitions = (
        "DEFINITIONS:\n"
        "- PROGNOSTIC FACTOR: A feature that predicts the outcome regardless of treatment (e.g., 'Age increases mortality'). "
        "-> IGNORE these unless they also modify treatment response.\n"
        "- PREDICTIVE FACTOR (EFFECT MODIFIER): A feature that changes the MAGNITUDE or DIRECTION of the treatment benefit "
        "(e.g., 'Drug works better in Young people than Old'). -> FOCUS on these.\n"
    )

    # 2. EXPLANATION DIVERSITY INSTRUCTION
    diversity_instruction = (
        "For each feature, propose distinct explanations of how it modifies the treatment effect.\n"
        "Each explanation should offer a different perspective or mechanism (e.g., direct biological effect, "
        "proxy/confounding role, pharmacological interaction).\n"
    )

    # 3. CONTEXTUAL LOGIC
    use_data_summary = study_context.get("use_data_summary", False)

    if len(top_features) > 0:
        # MODE 1: WITH SHAP (Interpretation)
        role_type = "Clinical Expert"
        directive = (
            "You are analyzing feature attribution from a conditional average treatment effect model using trial metadata (treatment/outcome/population).\n"
            "TASK: Explain WHY these specific features could modify treatment response.\n"
            "Treat SHAP-ranked features as model signals to interpret, not proof of causality.\n"
            "If a feature is coded/ambiguous or not strongly established in literature, keep it and explain it as a plausible proxy or exploratory/novel modifier with explicit uncertainty.\n"
        )
    elif use_data_summary:
        # MODE 2: BLINDED PREDICTION (Available Features Only)
        role_type = "Clinical Expert"
        available_cols = study_context.get("available_features", [])
        directive = (
            f"You know only the study design and the list of measured features: {available_cols}.\n"
            "TASK: identify which of these available features would modify the treatment response.\n"
        )
    else:
        # MODE 3: PURE THEORY (Literature Only)
        role_type = "Clinical Expert"
        directive = (
            "Based ONLY on the trial metadata (Treatment/Outcome), "
            "TASK: identify which patient characteristics would modify the treatment response."
        )

    # 4. ASSEMBLE SYSTEM PROMPT
    system_instructions = (
        f"You are a {role_type}.\n\n"
        f"{definitions}\n"
        f"{directive}\n\n"
        f"{diversity_instruction}\n"
        "REQUIREMENTS:\n"
        "1. Focus STRICTLY on Heterogeneous Treatment Effects (Interaction), not just main effects.\n"
        "2. If a feature has a bidirectional effect (e.g., good for some, bad for others), specify that.\n"
        "3. grounding: Cite known trials or physiological principles.\n"
        "4. When data_to_interpret provides mapped clinical feature labels, use those labels exactly in feature_name; do not output raw codes alone.\n"
        "5. For each feature, distinguish whether support is established vs exploratory; include caveats when evidence is limited rather than dropping the feature.\n"
        "6. INTERACTION FRAMING (critical): Every mechanism description MUST be framed as a treatment-response interaction, NOT a prognostic statement. "
        "BAD (prognostic): 'Older patients have higher mortality.' "
        "GOOD (interaction): 'Older patients show attenuated benefit from chemotherapy due to reduced tolerability and higher rates of dose-limiting adverse events, "
        "resulting in smaller net absolute benefit compared to younger patients with equivalent disease burden.' "
        "Always specify: who benefits more/less, from which treatment, and why the magnitude of benefit differs between subgroups.\n"
        "7. PRECISION — avoid oversimplified monotone subgroup claims: Real treatment-effect heterogeneity is often conditional on multiple factors. "
        "If a feature's interaction with treatment depends on a second moderator (e.g., age × comorbidity, eGFR × diabetes duration, LDL × CVD history), "
        "name that conditionality explicitly rather than stating a blanket directional claim. "
        "BAD: 'Older patients uniformly benefit less from statin therapy.' "
        "GOOD: 'Among patients aged ≥75 WITHOUT established CVD, statin therapy yields smaller absolute benefit due to competing mortality risks and shorter life expectancy; "
        "however, among older patients WITH pre-existing CVD, statin treatment still substantially reduces cardiovascular events — "
        "indicating the age-treatment interaction is moderated by CVD burden, not a simple monotone attenuation.' "
        "This level of specificity is required whenever the feature's effect modifier role is likely non-uniform across its range.\n"
    )

    # 5. USER PROMPT
    n_mechanisms = study_context.get("n_explanations_per_feature", 3)
    n_features = study_context.get("n_features", len(top_features) if top_features else 5)

    # Check if SHAP directional info is available
    has_shap_direction = (
        top_features
        and any(
            item.get("shap_mean") is not None and item.get("shap_mean") != 0
            for item in top_features
        )
    )

    task_constraints = [
        f"Generate explanations for the top {n_features} features.",
        f"For each feature, provide exactly {n_mechanisms} explanation(s) in the 'mechanisms' list.",
        "Each explanation should describe how the feature modifies the TREATMENT EFFECT.",
        "Ensure 'effect_direction' describes how the feature changes the TREATMENT BENEFIT (e.g., 'Positive' = Feature increases benefit).",
    ]

    if has_shap_direction:
        task_constraints.append(
            "DIRECTIONAL CONSISTENCY: Each feature in data_to_interpret has a 'shap_mean' (signed) field. "
            "A negative shap_mean means higher values of the feature are associated with a MORE NEGATIVE treatment effect (less benefit or more harm from treatment). "
            "A positive shap_mean means higher values are associated with a MORE POSITIVE treatment effect (more benefit from treatment). "
            "Your explanation direction MUST be consistent with the shap_mean sign. "
            "For example, if shap_mean is negative for serum creatinine, your explanation should state why HIGHER creatinine leads to LESS benefit from treatment (not more)."
        )

    task_constraints.extend([
        "CRITICAL — interaction framing: each mechanism description must answer 'which subgroup benefits MORE (or LESS) from the treatment and WHY the magnitude of treatment effect differs.' "
        "It must NOT be a prognostic statement about the feature's effect on outcome regardless of treatment. "
        "BAD: 'Time from sepsis onset may affect treatment efficacy.' "
        "GOOD: 'Patients with septic shock treated within 1h of presentation benefit more from broad-spectrum antibiotics because early source control prevents progression to multi-organ failure; delayed treatment beyond 3h allows the systemic inflammatory cascade to become self-sustaining, substantially attenuating the absolute survival benefit.' "
        "PRECISION: avoid oversimplified monotone claims. If the interaction is moderated by a second factor (e.g., age × comorbidity, eGFR × diabetes duration, LDL × CVD history), "
        "describe the joint subgroup precisely. "
        "BAD: 'Older patients benefit less from statin therapy.' "
        "GOOD: 'Older patients WITHOUT prior CVD benefit less in absolute terms due to competing mortality risks; older patients WITH CVD still benefit substantially — the age-treatment interaction is moderated by CVD burden, not a uniform attenuation.'",
    ])

    user_prompt = {
        "study_context": study_context,
        "data_to_interpret": top_features if top_features else "NONE (Blinded Mode)",
        "task_constraints": task_constraints,
    }

    try:
        result = _parse_structured(
            client, model_name,
            [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            FeatureExplanationsSet,
        )
    except Exception as e:
        print(f"Error in explanation generation: {e}")
        return None

    # --- Gap-fill: ensure every feature has exactly n_mechanisms mechanisms ---
    if result and result.feature_explanations:
        # Normalize any raw feature names back to mapped clinical labels.
        for fh in result.feature_explanations:
            current_name = (fh.feature_name or "").strip()
            mapped_name = feature_label_map.get(current_name.lower())
            if mapped_name:
                fh.feature_name = mapped_name

        for fh in result.feature_explanations:
            existing = fh.mechanisms or []
            gap = n_mechanisms - len(existing)
            if gap <= 0:
                continue
            print(f"  Gap-fill '{fh.feature_name}': have {len(existing)}, need {gap} more mechanisms...")
            extra = _fill_mechanisms_for_feature(
                feature_explanation=fh,
                count_needed=gap,
                study_context=study_context,
                client=client,
                model_name=model_name,
            )
            fh.mechanisms = existing + extra
            if len(fh.mechanisms) < n_mechanisms:
                print(f"    Warning: still only {len(fh.mechanisms)}/{n_mechanisms} after fill")

    return result

# -----------------------------
# Main
# -----------------------------


def main():
    load_local_env()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shap_json",
        required=True,
        help="Path to Shapley summary JSON created earlier.",
    )
    parser.add_argument(
        "--out_json", required=True, help="Where to write generated explanations JSON."
    )

    # Option 1: Use trial name for automatic metadata lookup
    parser.add_argument(
        "--trial_name",
        help="Trial name (ist3, crash_2, sprint, accord, txa) - auto-populates metadata.",
    )

    # Option 2: Manual metadata (used if --trial_name not provided)
    parser.add_argument(
        "--treatment",
        help="Treatment/exposure description (required if no --trial_name).",
    )
    parser.add_argument(
        "--outcome", help="Outcome description (required if no --trial_name)."
    )
    parser.add_argument(
        "--population",
        help="Population/cohort description (required if no --trial_name).",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (overrides metadata from SHAP JSON; defaults to trial_name if provided).",
    )

    parser.add_argument(
        "--n_features",
        type=int,
        default=15,
        help="Number of top Shapley features to include.",
    )
    parser.add_argument(
        "--n_explanations", type=int, default=8, help="How many explanations to generate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed identifier used to create output subfolder seed_<seed>.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model name supporting structured outputs (e.g., gpt-5-mini for OpenAI, openai/gpt-4o for OpenRouter).",
    )
    parser.add_argument(
        "--api_provider",
        default="openai",
        choices=["openai", "openrouter", "medgemma"],
        help="API provider to use (default: openai). Use 'medgemma' for local MedGemma model.",
    )
    parser.add_argument(
        "--medgemma_model",
        default="google/medgemma-27b-text-it",
        help="HuggingFace model ID for MedGemma (default: google/medgemma-27b-text-it).",
    )
    parser.add_argument(
        "--medgemma_device",
        default="cuda",
        help="Device for MedGemma inference (default: cuda).",
    )
    parser.add_argument(
        "--api_base_url",
        help="Custom API base URL (e.g., https://openrouter.ai/api/v1 for OpenRouter).",
    )
    parser.add_argument(
        "--enable_verifier",
        action="store_true",
        help="Enable verification/refinement pass using the model specified by --model.",
    )
    parser.add_argument(
        "--use_data_summary",
        action="store_true",
        help="Use data summary baseline: provide available features but no SHAP values (intermediate between with_shap and without_shap).",
    )
    parser.add_argument(
        "--verifier_iterations",
        type=int,
        default=1,
        help="Number of refinement iterations with verifier (default: 1)",
    )
    parser.add_argument(
        "--retrieve_article",
        action="store_true",
        help="Search for and extract information from the original trial article.",
    )
    parser.add_argument(
        "--fail_on_reject",
        action="store_true",
        help="Exit non-zero if verifier rejects.",
    )
    args = parser.parse_args()

    seeded_out_json = resolve_seeded_output_path(args.out_json, args.seed)
    if seeded_out_json != args.out_json:
        print(f"Using seeded output path: {seeded_out_json}")
    args.out_json = seeded_out_json

    # Set verifier model based on flags
    verifier_model = args.model if args.enable_verifier else None

    client = get_model_client(args.api_provider, args.api_base_url, medgemma_model=args.medgemma_model, medgemma_device=args.medgemma_device)

    # Determine treatment/outcome/population and fetch trial_meta once
    trial_meta = None
    if args.trial_name:
        trial_meta = get_trial_metadata(args.trial_name)
        treatment = trial_meta["treatment"]
        outcome = trial_meta["outcome"]
        population = trial_meta["population"]
    else:
        if not all([args.treatment, args.outcome, args.population]):
            parser.error(
                "Must provide either --trial_name OR all of (--treatment, --outcome, --population)"
            )
        treatment = args.treatment
        outcome = args.outcome
        population = args.population

    # Determine dataset name: explicit --dataset > trial_name > SHAP JSON metadata
    dataset_name = args.dataset if args.dataset else (args.trial_name.lower() if args.trial_name else None)
    evidence = load_top_features(args.shap_json, args.n_features, dataset_override=dataset_name)

    # ---------- RETRIEVE ARTICLE (optional) ----------
    article_extraction = None
    if args.retrieve_article and args.trial_name:
        print(f"Retrieving article information for {args.trial_name}...")
        article_query = trial_meta.get("article_query", f"{args.trial_name} clinical trial")

        article_extraction = search_and_extract_article(
            article_query, args.trial_name, client, model_name=args.model
        )
        if article_extraction:
            print("Successfully extracted article information.")
    elif args.retrieve_article and not args.trial_name:
        print("Warning: --retrieve_article requires --trial_name. Skipping article retrieval.")

    # ---------- GENERATE FEATURE EXPLANATIONS ----------
    print("Generating mechanistic explanations for individual features...")

    study_context = {
        "dataset": evidence["dataset"],
        "learner": evidence["learner"],
        "population": population,
        "treatment": treatment,
        "outcome": outcome,
        "source_explainer": evidence["explainer"],
        "n_explanations_per_feature": args.n_explanations,
        "n_features": args.n_features,  # Pass n_features so generator knows how many to propose if no SHAP
        "available_features": evidence.get("available_features", []),
        "use_data_summary": args.use_data_summary,  # New baseline mode flag
    }

    feature_explanations = generate_feature_explanations(
        top_features=evidence["top_feature_evidence"],
        study_context=study_context,
        client=client,
        model_name=args.model,
    )

    if not feature_explanations:
        raise RuntimeError("Failed to generate feature explanations")

    print(f"Generated feature-level explanations for {len(feature_explanations.feature_explanations)} features.")

    final_output = feature_explanations
    verification_report = None

    # Save original explanations before refinement
    original_output = feature_explanations

    # ---------- VERIFY (optional) ----------
    if verifier_model:
        print(f"Refining feature explanations with verifier ({args.verifier_iterations} iteration(s))...")

        for iteration in range(args.verifier_iterations):
            if iteration > 0:
                print(f"  Refinement iteration {iteration + 1}/{args.verifier_iterations}...")

            verifier_system = (
                "You are a collaborative scientific advisor helping to REFINE feature-level mechanistic explanations.\n"
                "Your role is to IMPROVE the explanations, not just evaluate them.\n"
                "\n"
                "For each explanation:\n"
                "1. Identify strengths worth preserving\n"
                "2. Spot weaknesses that need improvement\n"
                "3. Provide constructive refinement suggestions\n"
                "4. ALWAYS return a revised, improved version\n"
                "\n"
                "Focus on:\n"
                "- Strengthening mechanism plausibility with more specific biological/clinical details\n"
                "- Sharpening clinical interpretation to be more precise\n"
                "- Better aligning with SHAP evidence (direction, magnitude)\n"
                "- Making subgroup implications more actionable\n"
                "- Enhancing validation plans with concrete, feasible steps\n"
                "- Adding important caveats and alternative explanations\n"
                "- INTERACTION FRAMING: Rewrite any mechanism description that is framed as a prognostic statement into a treatment-response interaction claim. "
                "Every description must answer: which subgroup benefits more (or less) from treatment, and why the magnitude of treatment effect differs. "
                "BAD: 'High baseline LDL leads to worse cardiovascular outcomes.' "
                "GOOD: 'Patients with high baseline LDL (>190 mg/dL) derive greater absolute benefit from statin therapy because their elevated baseline event rate translates even a moderate relative risk reduction into a larger absolute risk reduction per patient treated.' "
                "If a description does not specify differential treatment response, rewrite it so it does.\n"
                "- PRECISION CHECK: Flag and rewrite any oversimplified monotone subgroup claim. "
                "If the feature's interaction with treatment is moderated by a second factor (e.g., age × comorbidity, eGFR × diabetes duration, LDL × CVD history), "
                "the refined description must capture that conditionality explicitly rather than stating a blanket direction. "
                "BAD: 'Older patients benefit less from statin therapy.' "
                "GOOD: 'Among older patients WITHOUT established CVD, statin therapy yields smaller absolute benefit due to competing mortality risks; "
                "among older patients WITH CVD, statin treatment still substantially reduces events — the age-treatment interaction is moderated by CVD burden, not a simple attenuation.'\n"
                "\n"
                "For EACH explanation:\n"
                "- Assess plausibility (high/moderate/low/implausible)\n"
                "- Suggest specific improvements to the description\n"
                "- Mark for revision if implausible or poorly supported\n"
                "\n"
                "If trial article context provided:\n"
                "- Refine mechanisms to align with known trial physiology\n"
                "- Adjust interpretations to match population characteristics\n"
                "- Tailor validation plans to be feasible within trial design\n"
                "- Flag any contradictions with trial findings\n"
                "\n"
                "IMPORTANT: Your goal is to help create the BEST possible explanations.\n"
                "Always provide a complete revised FeatureExplanationsSet with improvements,\n"
                "even if changes are minor. Build on strengths and fix weaknesses.\n"
                "Stay grounded in evidence - improve but don't add unsupported claims.\n"
                "\n"
                "CRITICAL: You MUST preserve ALL explanations for every feature. Never reduce the number of\n"
                "explanations. If the input has 2 explanations for a feature, the output must also have exactly\n"
                "2 explanations. Refine or rewrite descriptions, but do NOT drop or omit any."
                )

            # Use current explanations (either original or from previous iteration)
            current_explanations = final_output if iteration > 0 else feature_explanations

            verifier_prompt = {
                "iteration": iteration + 1,
                "total_iterations": args.verifier_iterations,
                "evidence": {
                    "study_context": study_context,
                    "top_feature_evidence": evidence["top_feature_evidence"],
                },
                "current_explanations": current_explanations.model_dump(),
                "refinement_goals": {
                    "strengthen_mechanisms": True,
                    "sharpen_clinical_interpretation": True,
                    "improve_evidence_alignment": True,
                    "make_subgroups_actionable": True,
                    "enhance_validation_plans": True,
                    "add_important_caveats": True,
                },
                "instructions": [
                    "Review each feature explanation and its mechanisms",
                    "Identify specific improvements to make",
                    "Provide detailed per-mechanism reviews",
                    "Return a complete revised FeatureExplanationsSet",
                    "Build on what works, fix what doesn't",
                    "Make explanations more specific, actionable, and evidence-based",
                ],
            }

            # Add article context if available
            if article_extraction is not None:
                verifier_prompt["trial_article_context"] = article_extraction.model_dump()
                verifier_prompt["refinement_goals"]["align_with_trial_context"] = (
                    "Use trial article to refine mechanisms and interpretations to match trial physiology"
                )

            verification_report: FeatureVerificationOutput = _parse_structured(
                client, verifier_model,
                [
                    {"role": "system", "content": verifier_system},
                    {"role": "user", "content": json.dumps(verifier_prompt, indent=2)},
                ],
                FeatureVerificationOutput,
            )

            # Track refinement progress
            if verification_report.revised is not None:
                final_output = verification_report.revised
                num_revisions = sum(
                    1 for review in verification_report.per_feature
                    if review.verdict in ["revise", "reject"]
                )
                print(f"    Refined {num_revisions}/{len(verification_report.per_feature)} features")
            else:
                print(f"    No revisions produced in iteration {iteration + 1}")

        print(f"Completed {args.verifier_iterations} refinement iteration(s). Using final refined explanations.")

    # ---------- WRITE OUTPUTS ----------
    # Always save original explanations
    write_json_file(args.out_json, original_output.model_dump())
    print(f"Wrote original feature explanations to: {args.out_json}")

    # If refined, save revised version separately
    if verification_report is not None:
        revised_path = os.path.splitext(args.out_json)[0] + "_revised.json"
        write_json_file(revised_path, final_output.model_dump())
        print(f"Wrote revised feature explanations to: {revised_path}")

        # Write verifier report
        report_path = os.path.splitext(args.out_json)[0] + "_verification.json"
        write_json_file(report_path, verification_report.model_dump())
        print(f"Wrote verifier report to: {report_path}")

    # Write article extraction if available
    if article_extraction is not None:
        article_path = os.path.splitext(args.out_json)[0] + "_article_context.json"
        write_json_file(article_path, article_extraction.model_dump())
        print(f"Wrote article context to: {article_path}")

    if (
        args.fail_on_reject
        and verification_report is not None
        and verification_report.overall_verdict == "reject"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
