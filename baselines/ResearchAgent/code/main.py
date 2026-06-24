import argparse
import json
import os
import re
from tqdm import tqdm

from utils import data_io
from knowledge.store import KnowledgeStore
from models.openai import OpenAIClient
from pipelines.research_pipeline import ResearchPipeline


TRIAL_PAPERS = {
    'ist3': {
        'title': 'The benefits and harms of intravenous thrombolysis with recombinant tissue plasminogen activator within 6 h of acute ischemic stroke (IST-3): a randomised controlled trial',
        'abstract': (
            'IST-3 evaluated intravenous alteplase versus control in acute ischemic stroke patients treated within 6 hours of symptom onset. '
            'The trial examined functional outcome and mortality, and reported important subgroup heterogeneity analyses, including age, stroke severity, and time to treatment.'
        )
    },
    'crash_2': {
        'title': 'Effects of tranexamic acid on death, vascular occlusive events, and blood transfusion in trauma patients with significant haemorrhage (CRASH-2): a randomised, placebo-controlled trial',
        'abstract': (
            'CRASH-2 randomized bleeding trauma patients to tranexamic acid or placebo and assessed mortality and vascular outcomes. '
            'The study highlighted time-to-treatment effects and explored clinically relevant subgroups for treatment benefit and safety.'
        )
    },
    'sprint': {
        'title': 'A Randomized Trial of Intensive versus Standard Blood-Pressure Control (SPRINT)',
        'abstract': (
            'SPRINT compared intensive versus standard systolic blood pressure targets in high-risk adults without diabetes. '
            'It reported cardiovascular and mortality outcomes and included subgroup analyses across baseline risk characteristics.'
        )
    },
    'accord': {
        'title': 'Effects of Intensive Blood-Pressure Control in Type 2 Diabetes Mellitus (ACCORD BP)',
        'abstract': (
            'ACCORD BP compared intensive versus standard blood pressure targets in high-risk adults with type 2 diabetes. '
            'It evaluated major cardiovascular outcomes and mortality and reported subgroup-relevant heterogeneity patterns.'
        )
    },
    'accord_glycemia': {
        'title': 'Effects of Intensive Glucose Lowering in Type 2 Diabetes (ACCORD Glycemia)',
        'abstract': (
            'ACCORD Glycemia compared intensive versus standard glycemic control (HbA1c <6.0% vs 7.0-7.9%) in high-risk adults with type 2 diabetes. '
            'It evaluated major cardiovascular outcomes and mortality and reported treatment effect heterogeneity across baseline characteristics.'
        )
    }
}


DEFAULT_FEATURE_BY_TRIAL = {
    'ist3': 'age',
    'crash_2': 'ninjurytime',
    'sprint': 'sbp',
    'accord': 'sbp',
    'accord_glycemia': 'hba1c',
}


FEATURE_CUES_BY_TRIAL = {
    'ist3': {
        'age': ['age', 'elderly', 'older'],
        'nihss': ['nihss', 'stroke severity'],
        'time_to_treatment': ['time to treatment', 'onset to treatment', 'treatment delay'],
        'sbprand': ['systolic blood pressure', 'sbp', 'blood pressure'],
        'dbprand': ['diastolic blood pressure', 'dbp'],
        'glucose': ['glucose', 'hyperglycemia'],
        'stroketype': ['stroke type', 'stroke subtype', 'laci', 'taci', 'paci', 'poci'],
        'infarct': ['infarct', 'ct infarct'],
    },
    'crash_2': {
        'iage': ['age', 'elderly', 'older'],
        'ninjurytime': ['injury time', 'time from injury', 'time to treatment'],
        'isbp': ['systolic blood pressure', 'sbp', 'hypotension'],
        'ihr': ['heart rate'],
        'irr': ['respiratory rate'],
        'igcs': ['glasgow coma scale', 'gcs'],
        'iinjurytype': ['injury type', 'blunt', 'penetrating'],
    },
    'sprint': {
        'age': ['age', 'elderly', 'older'],
        'sbp': ['systolic blood pressure', 'sbp', 'blood pressure'],
        'dbp': ['diastolic blood pressure', 'dbp'],
        'gfr': ['gfr', 'egfr', 'renal function'],
        'glur': ['glucose', 'glycemia'],
        'trr': ['triglyceride', 'triglycerides'],
        'umalcr': ['uacr', 'albuminuria', 'albumin creatinine'],
        'cvd_hx_baseline': ['cvd history', 'cardiovascular history', 'prior cardiovascular'],
    },
    'accord': {
        'age': ['age', 'elderly', 'older'],
        'sbp': ['systolic blood pressure', 'sbp', 'blood pressure'],
        'dbp': ['diastolic blood pressure', 'dbp'],
        'hba1c': ['hba1c', 'glycated hemoglobin'],
        'egfr': ['egfr', 'gfr', 'renal function'],
        'uacr': ['uacr', 'albuminuria', 'albumin creatinine'],
        'trig': ['triglyceride', 'triglycerides'],
        'fpg': ['fasting plasma glucose', 'fpg'],
        'sub_cvd': ['cvd', 'cardiovascular disease'],
        'sub_ckd': ['ckd', 'chronic kidney disease'],
    },
    'accord_glycemia': {
        'hba1c': ['hba1c', 'glycated hemoglobin', 'glycemic control'],
        'baseline_age': ['age', 'elderly', 'older'],
        'bmi': ['bmi', 'body mass index', 'obesity'],
        'yrsdiab': ['diabetes duration', 'years of diabetes'],
        'sbp': ['systolic blood pressure', 'sbp', 'blood pressure'],
        'fpg': ['fasting plasma glucose', 'fpg', 'glucose'],
        'gfr': ['gfr', 'egfr', 'renal function'],
        'insulin': ['insulin', 'insulin therapy'],
        'dm_med': ['diabetes medication', 'oral hypoglycemic'],
        'cvd_hx_baseline': ['cvd history', 'cardiovascular history', 'prior cardiovascular'],
    },
}


CLINICAL_AGENT_JUDGE_SYSTEM = (
    "You are an independent scientific judge for mechanistic explanations.\n"
    "Evaluate each explanation on multiple dimensions using a 1-5 scale.\n"
    "Be objective, fair, and constructive.\n"
    "\n"
    "IMPORTANT: Each explanation includes an 'importance_rank' field (1=most important feature).\n"
    "When evaluating Evidence Alignment, consider whether the RELATIVE RANKING aligns with\n"
    "clinical knowledge. Features ranked higher should be more established effect modifiers\n"
    "according to literature. A mismatch between model ranking and clinical knowledge should\n"
    "lower the Evidence Alignment score.\n"
    "\n"
    "SCORING CRITERIA:\n"
    "\n"
    "1. Mechanism Plausibility (1-5):\n"
    "   - Biological/physiological coherence of proposed mechanisms\n"
    "   - Consistency with established pathophysiology and pharmacology\n"
    "   - Specificity to the treatment and outcome context\n"
    "   - Avoidance of generic mechanisms that could apply to any feature\n"
    "   Score 5: Strong biological basis, well-established pathways, specific to context\n"
    "   Score 3: Reasonable but speculative, some supporting literature, moderately specific\n"
    "   Score 1: Implausible, contradicts known biology, purely generic\n"
    "\n"
    "2. Evidence Alignment (1-5):\n"
    "   - Feature importance: Whether identified features are known effect modifiers from published trials\n"
    "   - Ranking accuracy: Does the feature's importance_rank align with clinical literature?\n"
    "     * Higher-ranked features (rank 1, 2, 3) should be well-established modifiers\n"
    "     * Lower-ranked features may be plausible but less critical\n"
    "     * Penalize if a weakly-supported feature ranks above a well-established one\n"
    "     * Penalize if proposed features were not measured/available in the original trial\n"
    "   - Correctness of feature selection/ranking based on existing clinical literature\n"
    "   - Consistency with meta-analyses and systematic reviews on treatment heterogeneity\n"
    "   - Mechanistic grounding: Citations and references to established treatment effect heterogeneity\n"
    "   - Connection to known biological markers and risk stratification literature\n"
    "   - Appropriate recognition when evidence is sparse or speculative\n"
    "   Score 5: Feature is a well-established modifier AND ranking position matches clinical importance\n"
    "   Score 3: Plausible modifier with some literature support, OR ranking doesn't match expected clinical priority\n"
    "   Score 1: Unlikely modifier, contradicts literature, OR inappropriate ranking (weak feature ranked too high)\n"
    "\n"
    "3. Subgroup Implications (1-5):\n"
    "   - Clarity and actionability of proposed subgroups\n"
    "   - Feasibility of defining subgroups in practice (available data, clear cutpoints)\n"
    "   - Clinical utility - would these subgroups inform treatment decisions?\n"
    "   - Avoidance of arbitrary or clinically meaningless stratifications\n"
    "   Score 5: Clear, actionable, clinically meaningful subgroups\n"
    "   Score 3: Reasonable but vague or difficult to operationalize\n"
    "   Score 1: Unclear, arbitrary, or clinically meaningless\n"
    "\n"
    "4. Caveat Awareness (1-5):\n"
    "   - Thoroughness in acknowledging limitations and alternative explanations\n"
    "   - Recognition of potential confounding, bias, measurement error\n"
    "   - Appropriate epistemic humility (avoiding overclaiming)\n"
    "   - Acknowledgment when evidence is weak or mechanisms speculative\n"
    "   Score 5: Comprehensive caveats, honest about limitations\n"
    "   Score 3: Some caveats but incomplete or superficial\n"
    "   Score 1: Overclaiming, ignoring limitations, false certainty\n"
    "\n"
    "5. Novelty (1-5):\n"
    "   - Originality of the explanation beyond existing clinical literature\n"
    "   - Potential to generate new insights or challenge existing paradigms\n"
    "   - Whether the explanation identifies underexplored treatment effect modifiers\n"
    "   - Balance between novelty and plausibility (novel but not implausible)\n"
    "   Score 5: Highly original, identifies underexplored mechanisms, potential paradigm shift\n"
    "   Score 3: Moderately novel, extends existing knowledge in meaningful ways\n"
    "   Score 1: Reiterates well-established findings, no new insights\n"
    "\n"
    "For EACH individual mechanism, also score (1-5 scale):\n"
    "- Plausibility (1-5): biological believability of this specific mechanism\n"
    "- Evidence support (1-5): how well clinical literature supports this mechanism\n"
    "- Specificity (1-5): how detailed and connected to clinical/biological reasoning\n"
    "- Testability (1-5): how testable/falsifiable with available data and methods\n"
    "- Novelty (1-5): originality and uniqueness of this mechanism explanation\n"
    "- Overall score (1-5): holistic assessment of this mechanism\n"
    "- Brief comments: strengths, weaknesses, any concerns\n"
    "\n"
    "If original trial article context is provided, use it to:\n"
    "- Assess mechanism plausibility based on trial physiology\n"
    "- Judge interpretation accuracy against population characteristics\n"
    "- Evaluate validation feasibility given trial design\n"
    "- Cross-check if proposed mechanisms contradict known trial findings\n"
    "\n"
    "Provide honest, rigorous critique in strengths/weaknesses. Use the full 1-10 range."
)


def fetch_resources(paper: dict, knowledge_store: KnowledgeStore):
    from utils import s2

    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities(
        [paper['corpusId']] + [reference['corpusId'] for reference in references]
    )
    return references, entities


def load_trial_paper(trial_name: str) -> dict:
    normalized = (trial_name or '').strip().lower().replace('-', '_')
    if normalized not in TRIAL_PAPERS:
        raise ValueError(
            f"Unknown trial '{trial_name}'. Supported trials: {', '.join(sorted(TRIAL_PAPERS.keys()))}."
        )
    return TRIAL_PAPERS[normalized]


def resolve_feature_name(trial_name: str, feature_name: str | None) -> str:
    if feature_name and feature_name.strip():
        return feature_name.strip()
    trial_key = (trial_name or '').strip().lower().replace('-', '_')
    return DEFAULT_FEATURE_BY_TRIAL.get(trial_key, 'feature')


def infer_feature_from_generated_explanation(
    context: dict,
    dataset: str,
    api_client: OpenAIClient,
    fallback_feature_name: str | None = None,
) -> str:
    trial_key = (dataset or '').strip().lower().replace('-', '_')
    cues = FEATURE_CUES_BY_TRIAL.get(trial_key, {})

    texts = [
        context.get('problem', ''),
        context.get('problem_rationale', ''),
        context.get('method', ''),
        context.get('method_rationale', ''),
        context.get('experiment', ''),
        context.get('experiment_rationale', ''),
    ]
    history = context.get('history', {}) or {}
    for key in ('problems', 'methods', 'experiments'):
        for item in history.get(key, []) or []:
            if isinstance(item, dict):
                texts.extend([item.get('problem', ''), item.get('method', ''), item.get('experiment', ''), item.get('rationale', '')])

    combined = "\n".join([t for t in texts if t]).lower()

    best_feature = None
    best_score = 0
    for feature_key, feature_cues in cues.items():
        score = sum(1 for cue in feature_cues if cue in combined)
        if score > best_score:
            best_feature = feature_key
            best_score = score
    if best_feature and best_score > 0:
        return best_feature

    if cues:
        prompt = (
            "Choose the single most likely effect-modifier feature key from the candidate list based on the generated trial explanations. "
            "Return only one exact key from the candidate list, no explanation.\n\n"
            f"Dataset: {trial_key}\n"
            f"Candidate feature keys: {', '.join(cues.keys())}\n\n"
            f"Generated text:\n{combined[:6000]}"
        )
        llm_choice = (api_client.call(messages=[
            {'role': 'system', 'content': 'Return one exact feature key only.'},
            {'role': 'user', 'content': prompt},
        ]) or '').strip().lower()
        if llm_choice in cues:
            return llm_choice

    return resolve_feature_name(trial_key, fallback_feature_name)


def infer_effect_direction(text: str) -> str:
    if not text:
        return 'unknown'
    lower_text = text.lower()
    if any(token in lower_text for token in ['increase', 'higher', 'worse', 'harm', 'risk']):
        return 'negative'
    if any(token in lower_text for token in ['decrease', 'lower', 'better', 'benefit', 'improve']):
        return 'positive'
    return 'unknown'


def _is_error_text(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return (
        'error code:' in low
        or 'invalid_request_error' in low
        or 'unsupported parameter' in low
        or 'traceback' in low
        or low.startswith('error:')
    )


def _expand_explanations_with_llm(
    context: dict,
    api_client: OpenAIClient,
    target_count: int,
) -> list[str]:
    prompt = (
        "Generate exactly "
        f"{target_count} distinct clinical mechanism explanations as a JSON array of strings. "
        "Ground them only in this trial context and the generated research outputs. "
        "Do not use external knowledge. Keep each item to one sentence.\n\n"
        f"Trial paper title: {context.get('paper', {}).get('title', '')}\n"
        f"Trial paper abstract: {context.get('paper', {}).get('abstract', '')}\n\n"
        f"Problem: {context.get('problem', '')}\n"
        f"Problem rationale: {context.get('problem_rationale', '')}\n\n"
        f"Method: {context.get('method', '')}\n"
        f"Method rationale: {context.get('method_rationale', '')}\n\n"
        f"Experiment: {context.get('experiment', '')}\n"
        f"Experiment rationale: {context.get('experiment_rationale', '')}\n"
    )

    raw = api_client.call(
        messages=[
            {'role': 'system', 'content': 'You are a precise clinical-trial explanation writer. Return valid JSON only.'},
            {'role': 'user', 'content': prompt},
        ]
    )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            cleaned = [str(item).strip() for item in parsed if str(item).strip()]
            return cleaned[:target_count]
    except Exception:
        pass

    line_candidates = []
    for line in raw.splitlines():
        normalized = re.sub(r'^\s*(?:[-*]|\d+[\.)])\s*', '', line).strip()
        if normalized:
            line_candidates.append(normalized)

    deduped = []
    seen = set()
    for text in line_candidates:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(text)

    return deduped[:target_count]


def _ensure_exactly_n_explanations(seed_texts: list[str], n: int) -> list[str]:
    seeds = [text.strip() for text in seed_texts if text and text.strip() and not _is_error_text(text)]
    if not seeds:
        seeds = ['No mechanism text generated from the current run.']

    output = []
    i = 0
    while len(output) < n:
        base = seeds[i % len(seeds)]
        variant = base if i < len(seeds) else f"{base} (variant {i + 1})"
        output.append(variant)
        i += 1
    return output[:n]


def _default_trial_explanations(context: dict, feature_name: str, n: int) -> list[str]:
    abstract = (context.get('paper', {}) or {}).get('abstract', '')
    trial = (context.get('paper', {}) or {}).get('title', 'the trial')
    base_texts = [
        f"In {trial}, baseline {feature_name} may modify treatment response through differential baseline cardiovascular risk.",
        f"Patients with different {feature_name} levels may show heterogeneous benefit due to variation in baseline disease burden.",
        f"Differences in {feature_name} may alter event risk trajectories, changing absolute treatment benefit.",
        f"The association between {feature_name} and outcome may interact with treatment intensity in the trial population.",
        f"Variation in {feature_name} may identify subgroups with different competing-risk profiles under treatment.",
    ]
    if abstract:
        base_texts.append(f"Trial abstract context suggests {feature_name} could act as an effect modifier for clinical outcomes.")
    return _ensure_exactly_n_explanations(base_texts, n)


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def score_feature_explanations_like_clinical_agent(
    feature_explanations: list[dict],
    study_context: dict,
    api_client: OpenAIClient,
) -> dict | None:
    judge_prompt = {
        "evidence": {
            "study_context": study_context,
        },
        "feature_explanations_to_score": feature_explanations,
        "scoring_instructions": {
            "be_objective": True,
            "score_range": "1-5 for each dimension",
            "provide_justification": True,
        },
    }

    raw = api_client.call(
        messages=[
            {"role": "system", "content": CLINICAL_AGENT_JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(judge_prompt, indent=2)},
        ]
    )

    if _is_error_text(raw):
        return None

    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed
    return {
        "summary": "Judge response was not strict JSON; raw response retained.",
        "raw_response": raw,
    }


def to_pubmed_explanations_format(
    context: dict,
    dataset: str,
    feature_name: str | None,
    api_client: OpenAIClient,
    num_explanations: int = 15,
) -> dict:
    inferred_feature_name = infer_feature_from_generated_explanation(
        context=context,
        dataset=dataset,
        api_client=api_client,
        fallback_feature_name=feature_name,
    )

    mechanisms = []
    candidates = [
        ('problem_mechanism', context.get('problem_rationale', '')),
        ('method_mechanism', context.get('method_rationale', '')),
        ('experiment_mechanism', context.get('experiment_rationale', '')),
    ]

    history = context.get('history', {}) or {}
    for item in history.get('problems', []) or []:
        text = item.get('rationale') or item.get('problem')
        if text:
            candidates.append(('problem_history_mechanism', text))
    for item in history.get('methods', []) or []:
        text = item.get('rationale') or item.get('method')
        if text:
            candidates.append(('method_history_mechanism', text))
    for item in history.get('experiments', []) or []:
        text = item.get('rationale') or item.get('experiment')
        if text:
            candidates.append(('experiment_history_mechanism', text))

    for mechanism_type, description in candidates:
        if description and not _is_error_text(description):
            mechanisms.append(
                {
                    'mechanism_type': mechanism_type,
                    'description': description,
                    'evidence_level': 'explanation_generating',
                    'effect_direction': infer_effect_direction(description),
                }
            )

    llm_needed = max(0, num_explanations - len(mechanisms))
    if llm_needed > 0:
        extra_descriptions = _expand_explanations_with_llm(
            context=context,
            api_client=api_client,
            target_count=llm_needed,
        )
        for description in extra_descriptions:
            mechanisms.append(
                {
                    'mechanism_type': 'llm_expanded_mechanism',
                    'description': description,
                    'evidence_level': 'explanation_generating',
                    'effect_direction': infer_effect_direction(description),
                }
            )

    if len(mechanisms) != num_explanations:
        seed_descriptions = [m.get('description', '') for m in mechanisms]
        if not any(text.strip() for text in seed_descriptions):
            repaired_descriptions = _default_trial_explanations(context, inferred_feature_name, num_explanations)
        else:
            repaired_descriptions = _ensure_exactly_n_explanations(seed_descriptions, n=num_explanations)
        mechanisms = [
            {
                'mechanism_type': 'explanation_mechanism',
                'description': desc,
                'evidence_level': 'explanation_generating',
                'effect_direction': infer_effect_direction(desc),
            }
            for desc in repaired_descriptions
        ]

    feature_explanation = {
        'feature_name': inferred_feature_name,
        'importance_rank': 1,
        'shap_value': 0.0,
        'effect_direction': mechanisms[0].get('effect_direction', 'unknown'),
        'clinical_interpretation': context.get('problem', ''),
        'why_important': 'Feature selected for trial-paper-only baseline mechanism validation.',
        'mechanisms': mechanisms,
        'subgroup_implications': context.get('method', ''),
        'validation_suggestions': [context.get('experiment', '')] if context.get('experiment') else [],
        'caveats': ['Generated without external retrieval; grounded only in the trial paper context.'],
    }

    return {
        'dataset': dataset,
        'model': 'ResearchAgent',
        'summary': f"{num_explanations} trial-paper-only explanations generated for {dataset} without external retrieval.",
        'feature_explanations': [feature_explanation],
        'cross_feature_patterns': None,
    }


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def resolve_seeded_output_path(output_path: str, seed: int) -> str:
    """Insert seed_<seed> before filename unless path already contains a seed folder."""
    normalized = (output_path or '').strip()
    if not normalized:
        return normalized

    path_obj = os.path.normpath(normalized)
    parts = path_obj.split(os.sep)
    if any(part.startswith('seed_') for part in parts):
        return normalized

    directory = os.path.dirname(normalized)
    filename = os.path.basename(normalized)
    seed_dir = f'seed_{seed}'

    if directory:
        return os.path.join(directory, seed_dir, filename)
    return os.path.join(seed_dir, filename)


def load_local_env(env_path: str = '.env') -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def run(
    papers: list,
    knowledge_store: KnowledgeStore,
    openai_client: OpenAIClient,
    ideas_output_path: str,
    use_external_retrieval: bool = False,
):
    results = []

    for paper in tqdm(papers):
        context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}

        references, entities = [], []
        if use_external_retrieval and knowledge_store is not None:
            references, entities = fetch_resources(paper, knowledge_store)
        context.update(references=references, entities=entities)

        research_pipeline = ResearchPipeline(api_client=openai_client)
        context = research_pipeline.run(context)

        results.append(context)
        data_io.save_result(ideas_output_path, context)

    return results


if __name__ == "__main__":
    load_local_env()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-path', '-d', default='./data/papers.jsonl')
    argparser.add_argument('--knowledge-path', '-k', default='./data/knowledge.jsonl')
    argparser.add_argument('--model-name', '-m', default='gpt-4o')
    argparser.add_argument('--api-provider', type=str, default='openai', choices=['openai', 'openrouter', 'medgemma'], help='LLM API provider.')
    argparser.add_argument('--api-key', type=str, default=None, help='Optional API key override. If omitted, uses env vars from .env.')
    argparser.add_argument('--api-base-url', type=str, default=None, help='Optional API base URL override (e.g., OpenRouter endpoint).')
    argparser.add_argument('--medgemma-model', type=str, default='google/medgemma-27b-text-it', help='HuggingFace model ID for MedGemma (default: google/medgemma-27b-text-it).')
    argparser.add_argument('--medgemma-device', type=str, default='cuda', help='Device for MedGemma inference (default: cuda).')
    argparser.add_argument('--trial-name', type=str, default='ist3', help='Cohort/trial name (e.g., ist3, crash_2, sprint, accord).')
    argparser.add_argument('--feature-name', type=str, default=None, help='Optional feature name used for PubMed mechanism validation format. If omitted, defaults by trial.')
    argparser.add_argument('--num-explanations', type=int, default=15, help='Number of mechanisms/explanations to export in PubMed format.')
    argparser.add_argument('--use-external-retrieval', action='store_true', help='Enable Semantic Scholar and entity retrieval. Default uses only trial paper context.')
    argparser.add_argument('--pubmed-output', type=str, default='', help='Optional output JSON path in pubmed_mechanism_validator.py input format.')
    argparser.add_argument('--ideas-output', type=str, default='./results/ideas.jsonl', help='Output JSONL path for generated ResearchAgent ideas.')
    argparser.add_argument('--seed', type=int, default=0, help='Seed index used to place outputs under seed_<seed> subfolder (default: 0).')
    args = argparser.parse_args()

    if args.num_explanations < 1:
        raise ValueError('--num-explanations must be >= 1')

    args.ideas_output = resolve_seeded_output_path(args.ideas_output, args.seed)
    if args.pubmed_output:
        args.pubmed_output = resolve_seeded_output_path(args.pubmed_output, args.seed)

    knowledge_store = KnowledgeStore(args.knowledge_path) if args.use_external_retrieval else None

    openai_client = OpenAIClient(
        model=args.model_name,
        api_provider=args.api_provider,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        medgemma_model=args.medgemma_model,
        medgemma_device=args.medgemma_device,
    )

    if args.use_external_retrieval:
        from utils import s2

        paper_ids = data_io.load_paper_ids(args.data_path)
        papers = s2.filter_papers(
            s2.get_papers(paper_ids),
            categories=['title', 'abstract', 'embedding']
        )
    else:
        papers = [load_trial_paper(args.trial_name)]

    results = run(
        papers=papers,
        knowledge_store=knowledge_store,
        openai_client=openai_client,
        ideas_output_path=args.ideas_output,
        use_external_retrieval=args.use_external_retrieval,
    )

    if args.pubmed_output and results:
        resolved_feature = resolve_feature_name(args.trial_name, args.feature_name)
        pubmed_format = to_pubmed_explanations_format(
            context=results[0],
            dataset=args.trial_name.lower().replace('-', '_'),
            feature_name=resolved_feature,
            api_client=openai_client,
            num_explanations=args.num_explanations,
        )
        save_json(args.pubmed_output, pubmed_format)
        print(f"Saved PubMed-validator explanations to: {args.pubmed_output}")
