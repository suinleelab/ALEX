#!/usr/bin/env python3
"""cot_baseline.py

Simple Chain-of-Thought (CoT) baseline for clinical explanation generation.

Two-step approach:
  1. CoT reasoning: ask the LLM to reason step-by-step about which patient
     features predict treatment effect heterogeneity for the given trial.
  2. Structured extraction: ask the LLM to convert its own reasoning into
     the standard feature_explanations JSON format consumed by judge and
     PubMed validator.

No SHAP values, no iterative refinement, no verifier — just CoT + structured output.

Example:
  python ALEX/baselines/cot_baseline.py \
    --trial_name crash_2 \
    --out_json ALEX/results/crash_2/gpt-5-mini/cot/seed_0/explanations.json \
    --seed 0 \
    --n_features 5 \
    --n_explanations 1 \
    --model gpt-5-mini
"""

import argparse
import json
import os
import re
import sys
from typing import List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.agent_utils import (
    get_model_client,
    get_trial_metadata,
    load_local_env,
    resolve_seeded_output_path,
    write_json_file,
)


# ---------------------------------------------------------------------------
# Output schema — minimal, matching HypoGeniC's plain-explanation format so
# that downstream judge/PubMed only sees the core explanation text, not LLM
# augmentations (rank, direction, caveats, etc.).
# ---------------------------------------------------------------------------

class MechanismExplanation(BaseModel):
    description: str = Field(
        ...,
        description="A plain-text explanation of how this feature modifies the treatment effect.",
    )


class FeatureExplanation(BaseModel):
    feature_name: str = Field(..., description="Name of the patient feature")
    mechanisms: List[MechanismExplanation] = Field(
        ..., description="One explanation per entry describing the feature's role in treatment heterogeneity"
    )


class FeatureExplanationsSet(BaseModel):
    dataset: str
    model: str
    feature_explanations: List[FeatureExplanation]


# ---------------------------------------------------------------------------
# Robust structured-output helpers (same pattern as clinical_agent.py)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")
_NO_STRUCTURED = ('qwen', 'deepseek', 'llama', 'mistral', 'mixtral')


def _repair_truncated_json(text: str) -> str:
    text = re.sub(r',?\s*"[^"]*$', '', text)
    text = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', text)
    text = text.rstrip().rstrip(',')
    opens = brackets = 0
    for ch in text:
        if ch == '{': opens += 1
        elif ch == '}': opens -= 1
        elif ch == '[': brackets += 1
        elif ch == ']': brackets -= 1
    text += ']' * max(brackets, 0) + '}' * max(opens, 0)
    return text


def _normalize_cot_dict(data: dict) -> dict:
    """Normalise a raw LLM dict to match FeatureExplanationsSet schema.

    Handles common issues with non-OpenAI models (MedGemma, Qwen, etc.):
    - Nested wrapper keys (study_context, analysis, …)
    - Alternate field names (feature → feature_name, explanations → feature_explanations)
    - mechanisms as list[str] → list[{description}]
    - Missing mechanisms → empty list
    """
    # --- Top-level defaults (also replace None values from LLM null outputs) ---
    if not data.get("dataset"):
        data["dataset"] = "unknown"
    if not data.get("model"):
        data["model"] = "unknown"

    # --- feature_explanations missing: try to rescue from nested wrapper keys ---
    def _hoist_list(src: dict, dst: dict) -> bool:
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
        for wrapper_key in ("study_context", "context", "analysis", "result", "output",
                            "response", "analysis_results", "feature_analysis"):
            wrapper = data.get(wrapper_key)
            if isinstance(wrapper, dict):
                if _hoist_list(wrapper, data):
                    for meta in ("dataset", "model", "summary"):
                        if data.get(meta) in ("unknown", "", None) and meta in wrapper:
                            data[meta] = wrapper[meta]
                    break
                if "dataset" in wrapper or "summary" in wrapper:
                    for k, v in wrapper.items():
                        data.setdefault(k, v)
                    if "feature_explanations" in data:
                        break
        if "feature_explanations" not in data:
            _hoist_list(data, data)

    # --- Per-feature normalization ---
    for fh in data.get("feature_explanations", []):
        # field name aliases
        if "feature_name" not in fh:
            for alias in ("feature", "feature_raw", "feature_label", "name", "feature_id"):
                if alias in fh:
                    fh["feature_name"] = fh.pop(alias)
                    break

        # mechanisms: missing → empty list
        fh.setdefault("mechanisms", [])
        mechs = fh["mechanisms"]
        if mechs and isinstance(mechs[0], str):
            fh["mechanisms"] = [{"description": m} for m in mechs]
        elif mechs and isinstance(mechs[0], dict) and "description" not in mechs[0]:
            fh["mechanisms"] = [
                {"description": m.get("description") or m.get("explanation") or m.get("text") or str(m)}
                for m in mechs
            ]
    return data


def _extract_json_from_content(content: str, response_format: Type[_T]) -> _T:
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.DOTALL)
    if fence:
        content = fence.group(1).strip()
    obj_match = re.search(r"\{[\s\S]*\}", content, flags=re.DOTALL)
    if obj_match:
        content = obj_match.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(content)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return response_format.model_validate_json(content)
    if isinstance(data, dict):
        data = _normalize_cot_dict(data)
    return response_format.model_validate(data)


def _token_limit_kwarg(model_name: str, client: OpenAI, limit: int = 16384) -> dict:
    """Return the correct max-token parameter for the model/provider."""
    is_openrouter = getattr(client, '_base_url', None) and 'openrouter' in str(client._base_url)
    is_medgemma = getattr(client, '_is_medgemma', False)
    model_lower = model_name.lower()
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
    fields_hint = ", ".join(f'"{f}"' for f in top_fields)
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
# CoT generation
# ---------------------------------------------------------------------------

_COT_SYSTEM = (
    "You are a clinical research expert specializing in treatment effect heterogeneity.\n"
    "Your task is to reason step by step about which patient characteristics are most "
    "likely to predict differential responses to a given treatment, drawing on clinical "
    "knowledge and trial physiology.\n\n"
    "Be specific: name concrete features, propose plausible mechanisms, and identify "
    "the patient subgroups that might benefit more or less from treatment."
)


def _run_cot_reasoning(
    trial_name: str,
    treatment: str,
    outcome: str,
    population: str,
    n_features: int,
    n_explanations: int,
    client: OpenAI,
    model_name: str,
) -> str:
    """Step 1: Free-form CoT reasoning — returns the raw reasoning text."""
    prompt = (
        f"Trial: {trial_name.upper()}\n"
        f"Treatment: {treatment}\n"
        f"Outcome: {outcome}\n"
        f"Population: {population}\n\n"
        f"Think step by step:\n"
        f"1. What are the {n_features} most clinically important patient features that could "
        f"   predict who benefits more or less from '{treatment}'?\n"
        f"2. For each feature, explain the biological or clinical reason it might modify "
        f"   the treatment effect (not just prognosis).\n"
        f"3. For each feature, identify the patient subgroup likely to benefit more and "
        f"   the subgroup likely to benefit less (or be harmed).\n"
        f"4. Note any interactions or patterns across features.\n\n"
        f"Provide {n_explanations} explanation(es) per feature."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _COT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    # Strip thinking blocks — we want only the visible reasoning
    content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _extract_structured_explanations(
    cot_reasoning: str,
    trial_name: str,
    treatment: str,
    outcome: str,
    population: str,
    n_features: int,
    n_explanations: int,
    model_name: str,
    client: OpenAI,
) -> FeatureExplanationsSet:
    """Step 2: Convert CoT reasoning into structured FeatureExplanationsSet.

    Only extracts feature_name and mechanism description(s) — no extra
    augmentation fields — so the explanation text is preserved faithfully.
    """
    system = (
        "You are a precise scientific formatter. Convert the provided chain-of-thought "
        "reasoning into a structured JSON output containing ONLY the feature name and "
        "plain-text explanation(s). Do not add new ideas, rankings, effect "
        "directions, caveats, or any other fields — faithfully extract only the core "
        f"explanation statements. Each feature must have exactly {n_explanations} entry(ies) "
        "in its 'mechanisms' list."
    )
    user = (
        f"Trial: {trial_name.upper()} | Treatment: {treatment} | "
        f"Outcome: {outcome} | Population: {population}\n\n"
        f"Chain-of-thought reasoning:\n{cot_reasoning}\n\n"
        f"Extract exactly {n_features} features from the reasoning above. For each "
        f"feature provide: feature_name (string) and mechanisms (list of {n_explanations} "
        f"description(s)). Nothing else."
    )

    completion = _parse_structured(
        client, model_name,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        FeatureExplanationsSet,
    )
    return completion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_local_env()

    parser = argparse.ArgumentParser(
        description="CoT baseline: two-step chain-of-thought explanation generation"
    )
    parser.add_argument("--trial_name", required=True,
                        help="Trial name (ist3, crash_2, sprint, accord)")
    parser.add_argument("--out_json", required=True,
                        help="Output path for explanations JSON")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for output subfolder (default: 0)")
    parser.add_argument("--n_features", type=int, default=5,
                        help="Number of features to explain about (default: 5)")
    parser.add_argument("--n_explanations", type=int, default=1,
                        help="Number of explanations per feature (default: 1)")
    parser.add_argument("--model", default="gpt-5-mini",
                        help="Model name (default: gpt-5-mini)")
    parser.add_argument("--api_provider", default="openai",
                        choices=["openai", "openrouter", "medgemma"],
                        help="API provider (default: openai). Use 'medgemma' for local MedGemma model.")
    parser.add_argument("--api_base_url", default=None,
                        help="Optional API base URL override")
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
    args = parser.parse_args()

    args.out_json = resolve_seeded_output_path(args.out_json, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    client = get_model_client(
        args.api_provider, args.api_base_url,
        medgemma_model=args.medgemma_model,
        medgemma_device=args.medgemma_device,
    )
    meta = get_trial_metadata(args.trial_name)
    treatment = meta["treatment"]
    outcome = meta["outcome"]
    population = meta["population"]

    print(f"Trial:      {args.trial_name}")
    print(f"Treatment:  {treatment}")
    print(f"Outcome:    {outcome}")
    print(f"Features:   {args.n_features}  |  Explanations/feature: {args.n_explanations}")
    print(f"Model:      {args.model}")
    print()

    # Step 1: CoT reasoning
    print("Step 1/2: Running chain-of-thought reasoning...")
    reasoning = _run_cot_reasoning(
        trial_name=args.trial_name,
        treatment=treatment,
        outcome=outcome,
        population=population,
        n_features=args.n_features,
        n_explanations=args.n_explanations,
        client=client,
        model_name=args.model,
    )
    print(f"  Reasoning complete ({len(reasoning)} chars)")

    # Step 2: Structured extraction
    print("Step 2/2: Extracting structured explanations from reasoning...")
    result = _extract_structured_explanations(
        cot_reasoning=reasoning,
        trial_name=args.trial_name,
        treatment=treatment,
        outcome=outcome,
        population=population,
        n_features=args.n_features,
        n_explanations=args.n_explanations,
        model_name=args.model,
        client=client,
    )
    result.dataset = args.trial_name
    result.model = args.model

    output = result.model_dump()
    output["method"] = "CoT"
    output["cot_reasoning"] = reasoning  # preserve the reasoning trace

    write_json_file(args.out_json, output)
    print(f"\nWrote {len(result.feature_explanations)} feature explanations to: {args.out_json}")
    for i, fh in enumerate(result.feature_explanations, 1):
        desc = fh.mechanisms[0].description[:80] + "..." if fh.mechanisms else "(no mechanisms)"
        print(f"  {i}. {fh.feature_name}: {desc}")


if __name__ == "__main__":
    main()
