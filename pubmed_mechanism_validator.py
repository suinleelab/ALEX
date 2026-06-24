"""
PubMed Mechanism Validator
===========================
This script extracts PubMed abstracts and analyzes whether they support or
conflict with proposed mechanisms from explanation files.

It uses a tiered search strategy:
1. Tier 1: Specific Mechanism (Treatment + Feature + Mechanism Keywords)
2. Tier 2: Strict Interaction (Treatment + Feature + Interaction Terms)
3. Tier 3: Broader Search (Treatment + Feature + No Interaction Filter)

Usage:
    python pubmed_mechanism_validator.py --input explanations_with_shap_XLearner.json --output validation_results.json
    python pubmed_mechanism_validator.py --cohort ist3 --max-abstracts 50
"""

import json
import argparse
import time
import os
import re
import hashlib
from typing import List, Dict, Any, Optional

from src.agent_utils import (
    _convert_hypogenic_to_feature_format,
    _is_hypogenic_format,
    load_json_file,
    load_local_env,
    write_json_file,
)

try:
    from Bio import Entrez
except ImportError:
    print("BioPython not found. Install with: pip install biopython")
    Entrez = None

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    print("OpenAI not found. Install with: pip install openai")
    OpenAI = None
    openai_available = False


class PubMedMechanismValidator:
    """Validates mechanisms against PubMed literature."""

    # PMIDs permanently excluded from all validation runs.
    # Use only for papers that cannot be fixed via prompt (e.g., metadata so ambiguous
    # that no gate catches it). Prefer fixing the evaluator prompt (Gate E) instead.
    PMID_BLACKLIST: set = {
        # IST3 / nihss feature
        "27507856",  # 9-RCT pooled analysis (Emberson 2016): abstract conclusion says
                     # "neither age nor stroke severity significantly influenced" the
                     # time-benefit slope, triggering NO_INTERACTION (-4 pts). The actual
                     # absolute risk data (22 vs 4 per 1000 by severity quintile) is buried
                     # in the full results section, not the abstract. Re-enable once
                     # --full-text is active.
        "40760234",  # ARAMIS secondary analysis (DAPT vs alteplase NIHSS 0-5):
                     # compares two active treatments (wrong control arm for alteplase
                     # vs placebo/no-treatment mechanism). Still retrieved by the
                     # nihss tier-1 query despite antiplat_rand fix.
        "35369376",  # SPRINT / DBP paper ("Baseline Diastolic Blood Pressure and
                     # Cardiovascular Outcomes"): a DBP-focused observational analysis
                     # that reports no interaction, but is wrongly retrieved for the
                     # sub_ckd and dbp queries. Not a CKD×BP-lowering RCT subgroup.
    }

    # Mapping from canonical feature key (matching dataset_config keys) to
    # clinical-concept synonyms. Used by _find_feature_query for cohort-aware
    # query construction; also exposed as a class attribute so subclasses
    # (e.g. V2) can resolve opaque variable names like "icc" or "sub_cvd" to
    # their actual clinical meaning before LLM strategy generation.
    FEATURE_ALIAS_CUES: tuple = (
        ('nihss', ('nihss', 'stroke severity')),
        ('time_to_treatment', ('time to treatment', 'onset to treatment', 'treatment delay')),
        ('ninjurytime', ('time from injury', 'injury to treatment', 'treatment delay')),
        ('age', ('age', 'elderly', 'older')),
        ('baseline_age', ('age', 'elderly', 'older', 'baseline age')),
        ('iage', ('age', 'elderly', 'older')),
        ('sbprand', ('systolic blood pressure',)),
        ('sbp', ('systolic blood pressure',)),
        ('isbp', ('systolic blood pressure', 'initial blood pressure', 'hypotension')),
        ('dbprand', ('diastolic blood pressure', 'dbprand', 'diastolic bp')),
        ('dbp', ('diastolic blood pressure',)),
        ('hba1c', ('hba1c', 'hemoglobin a1c', 'glycated hemoglobin')),
        ('gfr', ('gfr', 'egfr', 'renal function')),
        ('egfr', ('gfr', 'egfr', 'renal function')),
        ('glucose', ('glucose', 'hyperglycemia')),
        ('glur', ('glucose', 'fasting glucose', 'glycemia')),
        ('fpg', ('fasting plasma glucose', 'fpg', 'glucose')),
        ('trr', ('triglyceride', 'triglycerides')),
        ('trig', ('triglyceride', 'triglycerides')),
        ('vldl', ('vldl', 'very low density lipoprotein')),
        ('uacr', ('uacr', 'albumin creatinine ratio', 'albuminuria')),
        ('umalcr', ('umalcr', 'uacr', 'albumin creatinine ratio', 'albuminuria')),
        ('igcs', ('glasgow coma scale', 'gcs', 'consciousness')),
        ('irr', ('respiratory rate',)),
        ('hr', ('heart rate', 'tachycardia', 'bradycardia')),
        ('ihr', ('heart rate', 'tachycardia', 'bradycardia')),
        ('iinjurytype', ('injury type', 'blunt', 'penetrating', 'mechanism of injury')),
        ('icc', ('capillary refill', 'injury classification code', 'circulation code', 'peripheral perfusion')),
        ('antiplat_rand', ('antiplatelet', 'antiplat', 'prior antiplatelet', 'platelet inhibition', 'antithrombotic')),
        ('isex', ('sex', 'gender', 'male', 'female')),
        ('cvd_hx_baseline', ('history of cardiovascular disease', 'prior cardiovascular', 'cvd history')),
        ('prior stroke history', ('prior stroke', 'history of stroke', 'previous stroke')),
        ('sub_cvd', ('cardiovascular disease', 'cvd', 'heart disease')),
        ('sub_ckd', ('chronic kidney disease', 'ckd', 'renal impairment')),
        ('bp_med', ('bp med', 'blood pressure medication', 'antihypertensive')),
        ('dm_med', ('dm med', 'diabetes medication', 'oral hypoglycemic', 'antidiabetic')),
        ('anti_coag', ('anti coag', 'anticoagulant', 'anticoagulation', 'warfarin', 'heparin')),
        ('antiarrhythmic', ('antiarrhythmic', 'anti arrhythmic', 'rhythm control')),
        ('insulin', ('insulin', 'insulin therapy', 'insulin use')),
        ('yrsdiab', ('yrsdiab', 'diabetes duration', 'years of diabetes')),
        ('bmi', ('bmi', 'body mass index', 'obesity')),
        ('potassium', ('potassium', 'hyperkalemia', 'hypokalemia')),
        ('alt', ('alt', 'alanine aminotransferase', 'liver function')),
        ('cpk', ('cpk', 'creatine phosphokinase', 'creatine kinase')),
    )

    # PMIDs from each trial and its direct secondary analyses.
    # Excluded ONLY when validating explanations derived from that same trial
    # (prevents circular validation: a paper cannot corroborate an explanation
    # that was generated from the very same dataset).
    TRIAL_SOURCE_PMIDS: Dict[str, set] = {
        "crash_2": {
            "20554319",  # CRASH-2 main RCT (Lancet 2010)
            "23477634",  # CRASH-2 economic evaluation / extended trial report
            "21439633",  # CRASH-2: early treatment exploratory analysis
            "28143564",  # CRASH-2: exploration of benefits and harms (secondary)
        },
        "sprint": {
            "26551272",  # SPRINT main RCT (NEJM 2015)
            "31637971",  # SPRINT: albuminuria subgroup secondary analysis
            "33460256",  # SPRINT: heterogeneity of treatment effect by age
            "35254390",  # SPRINT MIND: cerebral blood flow secondary analysis
            "37105717",  # SPRINT: elderly patients secondary analysis
            "41159258",  # SPRINT: cerebral small vessel disease by age
        },
        "ist3": {
            "23859425",  # IST-3 main RCT (Lancet 2012)
        },
        "accord": {
            "20228401",  # ACCORD-BP main RCT (NEJM 2010)
            "39628286",  # ACCORD: HbA1c variability and intensive BP control
            "41159258",  # SPRINT+ACCORD bi-cohort: cerebral small vessel disease by age (contains ACCORD data)
        },
        "accord_glycemia": {
            "18539917",  # ACCORD glycemia main RCT (NEJM 2008)
            "25887355",  # ACCORD: hemoglobin glycation index secondary analysis
            "23114538",  # ACCORD: platelet function and tight glycemic control
        },
    }

    def __init__(self, email: str = "research@example.com", api_key: str = None, max_abstracts: int = 30, model: str = "gpt-5-mini", api_provider: str = "openai", api_base_url: str = None, llm_delay: float = 0.5, full_text: bool = False):
        if Entrez:
            Entrez.email = email
        self.api_key = api_key
        self.max_abstracts = max_abstracts
        self.model = model
        self.llm_delay = max(0.0, float(llm_delay))
        self.full_text = full_text
        self._search_cache: Dict[tuple, List[str]] = {}
        self._abstract_batch_cache: Dict[tuple, List[Dict[str, str]]] = {}
        self._llm_eval_cache: Dict[str, Dict[str, Any]] = {}
        self.openai_client = None
        if openai_available:
            if api_provider == "openrouter":
                resolved_key = api_key or os.environ.get('OPENROUTER_API_KEY')
                base_url = api_base_url or "https://openrouter.ai/api/v1"
                if resolved_key:
                    self.openai_client = OpenAI(api_key=resolved_key, base_url=base_url)
            elif api_key:
                kwargs = {"api_key": api_key}
                if api_base_url:
                    kwargs["base_url"] = api_base_url
                self.openai_client = OpenAI(**kwargs)

    def load_explanations(self, filepath: str) -> Dict[str, Any]:
        data = load_json_file(filepath)
        return _convert_hypogenic_to_feature_format(data) if _is_hypogenic_format(data) else data

    def get_cohort_trial_context(self, dataset: str) -> Dict[str, str]:
        """Return original cohort treatment/outcome context used for strict validation."""
        contexts = {
            'ist3': {
                'population': 'acute ischemic stroke patients',
                'treatment': 'rt-TPA / alteplase (intravenous thrombolysis)',
                'comparator': 'control/placebo or no rt-TPA',
                'outcome': 'functional outcome after stroke (e.g., mRS/dependency/death)'
            },
            'accord': {
                'population': 'type 2 diabetes patients at high cardiovascular risk',
                'treatment': 'intensive blood pressure control strategy',
                'comparator': 'standard blood pressure control strategy',
                'outcome': 'major cardiovascular outcomes and mortality'
            },
            'crash_2': {
                'population': 'adult trauma patients with or at risk of significant hemorrhage',
                'treatment': 'tranexamic acid (TXA)',
                'comparator': 'placebo/no TXA',
                'outcome': 'death due to bleeding / mortality outcomes'
            },
            'txa': {
                'population': 'adult pre-hospital trauma patients',
                'treatment': 'pre-hospital tranexamic acid (TXA)',
                'comparator': 'no pre-hospital TXA / usual care',
                'outcome': 'survival (in-hospital mortality status)'
            },
            'sprint': {
                'population': 'hypertensive adults at increased cardiovascular risk',
                'treatment': 'intensive systolic blood pressure target (<120 mmHg)',
                'comparator': 'standard systolic blood pressure target (<140 mmHg)',
                'outcome': 'major cardiovascular events and all-cause mortality'
            },
            'accord_glycemia': {
                'population': 'type 2 diabetes patients at high cardiovascular risk',
                'treatment': 'intensive glycemic control (target HbA1c <6.0%)',
                'comparator': 'standard glycemic control (target HbA1c 7.0-7.9%)',
                'outcome': 'major cardiovascular outcomes and mortality'
            }
        }
        return contexts.get(dataset, {
            'population': f'{dataset} cohort population',
            'treatment': 'cohort treatment strategy',
            'comparator': 'cohort comparator/control strategy',
            'outcome': 'cohort primary clinical outcome'
        })

    def _normalize_text(self, text: str) -> str:
        """Normalize text for robust fuzzy key matching."""
        return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()

    def _generated_name_to_feature_key(self, feature_name: str) -> str:
        """Map feature names generated in ensemble_shap.py to canonical keys."""
        normalized = self._normalize_text(feature_name)
        generated_aliases = {
            'stroke type taci total anterior circulation infarct': 'stroketype',
            'stroke type paci partial anterior circulation infarct': 'stroketype',
            'stroke type laci lacunar infarct': 'stroketype',
            'stroke type poci posterior circulation infarct': 'stroketype',
            'stroke type other': 'stroketype',
            'infarct visible on ct no': 'infarct_0',
            'infarct visible on ct possibly yes': 'infarct_1',
            'infarct visible on ct definitely yes': 'infarct_2',
            'injury type blunt': 'iinjurytype_1',
            'injury type penetrating': 'iinjurytype_2',
            'stroke type': 'stroketype',
            'iinjurytype': 'iinjurytype',
            'stroke severity nihss score': 'nihss'
        }
        return generated_aliases.get(normalized, '')

    def _extract_mechanism_keywords(self, description: str) -> str:
        """Extract high-value keywords from the mechanism description."""
        stop_words = {
            'the', 'and', 'with', 'that', 'this', 'for', 'from', 'are', 'may', 'can',
            'have', 'has', 'was', 'were', 'but', 'not', 'patients', 'clinical',
            'outcome', 'effect', 'treatment', 'associated', 'higher', 'lower',
            'increased', 'decreased', 'risk', 'benefit', 'efficacy', 'likely',
            'potential', 'potentially', 'due', 'because', 'mechanism', 'level',
            'study', 'trial', 'analysis', 'group', 'subgroup'
        }

        # Replace punctuation/hyphens with spaces so "treatment-response" → "treatment response"
        clean_desc = re.sub(r'[^a-zA-Z0-9]+', ' ', description.lower())
        tokens = clean_desc.split()

        # Keep specific words (4+ chars, not stop words)
        keywords = [t for t in tokens if len(t) > 3 and t not in stop_words]

        # Prioritize unique, longer words as they are usually more specific
        unique_keywords = sorted(list(set(keywords)), key=len, reverse=True)

        # Take top 5
        selected = unique_keywords[:5]

        if not selected:
            return ""

        return ' OR '.join([f'"{kw}"' for kw in selected])

    def _find_feature_query(self, feature_name: str, feature_map: Dict[str, str]) -> str:
        """Find best feature query from config using exact, alias, and fuzzy matching."""
        if not feature_name:
            return ""

        # 1. Exact/Lower match
        if feature_name in feature_map: return feature_map[feature_name]
        if feature_name.lower() in feature_map: return feature_map[feature_name.lower()]

        # 2. Mapped generated keys
        generated_key = self._generated_name_to_feature_key(feature_name)
        if generated_key and generated_key in feature_map: return feature_map[generated_key]

        # 2b. Extract raw key from parenthetical notation e.g. "Bp med (bp_med)" → "bp_med"
        paren_match = re.search(r'\(([a-z][a-z0-9_]*)\)\s*$', feature_name)
        if paren_match:
            raw_key = paren_match.group(1)
            if raw_key in feature_map:
                return feature_map[raw_key]

        # 3. Handle "feature:value"
        if ':' in feature_name:
            base_name = feature_name.split(':', 1)[0].strip().lower()
            if base_name in feature_map: return feature_map[base_name]

        # 4. Alias cues — read from class-level FEATURE_ALIAS_CUES so V2 can
        # reuse the same disambiguation map without duplication.
        normalized_feature = self._normalize_text(feature_name)
        for canonical_key, cues in self.FEATURE_ALIAS_CUES:
            if canonical_key in feature_map and any(cue in normalized_feature for cue in cues):
                return feature_map[canonical_key]

        return ""

    def construct_search_query(
        self,
        feature_name: str,
        mechanism: Dict[str, Any],
        dataset: str,
        require_interaction: bool = True,
        include_doc_types: bool = True,
        use_mechanism_keywords: bool = False
    ) -> str:
        """Construct a PubMed query optimized for effect-modifier evidence."""

        # Dataset Configuration
        dataset_config = {
            'ist3': {
                'context_terms': ['stroke', '"ischemic stroke"'],
                'treatment_terms': ['alteplase', '"intravenous thrombolysis"', '"tissue plasminogen activator"', 'rtPA', 'tPA'],
                'features': {
                    'stroketype': '"stroke subtype" OR lacunar OR cardioembolic OR "posterior circulation"',
                    'age': 'age OR elderly OR geriatric',
                    'nihss': 'NIHSS OR "stroke severity" OR "neurological deficit" OR "infarct volume"',
                    'sbprand': '"systolic blood pressure" OR "blood pressure" OR hypertension',
                    'dbprand': '"diastolic blood pressure" OR "blood pressure" OR hypertension OR "baseline blood pressure"',
                    'weight': 'weight OR BMI OR obesity OR "body mass"',
                    'glucose': 'glucose OR hyperglycemia OR "blood glucose"',
                    'gcs_score_rand': 'GCS OR "Glasgow Coma Scale" OR "consciousness level"',
                    'gender': 'sex OR gender OR male OR female',
                    'antiplat_rand': 'antiplatelet OR aspirin OR clopidogrel OR "prior antiplatelet" OR "antiplatelet therapy" OR "platelet inhibition" OR "antithrombotic pretreatment" OR "antiplatelet pretreatment" OR "pretreatment antiplatelet"',
                    'atrialfib_rand': '"atrial fibrillation" OR AF OR AFib',
                    'infarct': 'infarct OR ischemic OR "ischemic lesion"',
                    'stroketype_1': 'TACI OR "total anterior circulation infarct"',
                    'stroketype_2': 'PACI OR "partial anterior circulation infarct"',
                    'stroketype_3': 'LACI OR lacunar',
                    'stroketype_4': 'POCI OR "posterior circulation infarct"',
                    'stroketype_5': '"other ischemic stroke subtype"',
                    'diabetes mellitus': 'diabetes OR "diabetes mellitus" OR hyperglycemia OR glucose',
                    'prior stroke history': '"prior stroke" OR "previous stroke" OR "history of stroke"',
                    'time_to_treatment': '"time to treatment" OR "onset to treatment" OR "treatment delay" OR "door-to-needle" OR "symptom onset"'
                }
            },
            'accord': {
                'context_terms': ['diabetes', '"type 2 diabetes"', 'cardiovascular'],
                'treatment_terms': [
                    '"intensive blood pressure control"',
                    '"intensive systolic blood pressure"',
                    '"systolic blood pressure target"',
                    '"tight blood pressure control"',
                    '"aggressive blood pressure lowering"',
                    '"blood pressure management"',
                    '"intensive antihypertensive therapy"',
                    '"antihypertensive intensification"',
                ],
                'features': {
                    'hba1c': 'HbA1c OR "glycated hemoglobin" OR "glycemic control"',
                    'sbp': '"systolic blood pressure" OR hypertension',
                    'dbp': '"diastolic blood pressure" OR hypertension',
                    'age': 'age OR elderly OR geriatric',
                    'baseline_age': 'age OR elderly OR geriatric',
                    'bmi': 'BMI OR obesity OR "body mass index" OR "body mass" OR overweight OR adiposity OR "abdominal obesity" OR "waist circumference" OR "weight status"',
                    'duration': '"diabetes duration" OR "disease duration"',
                    'fpg': '"fasting plasma glucose" OR FPG OR glucose OR hyperglycemia',
                    'glur': '"fasting plasma glucose" OR FPG OR glucose OR glycemia OR hyperglycemia',
                    'gfr': 'GFR OR eGFR OR "renal function"',
                    'screat': 'creatinine OR "serum creatinine"',
                    'uacr': 'UACR OR albuminuria OR "albumin creatinine ratio"',
                    'umalcr': 'UACR OR albuminuria OR "albumin creatinine ratio"',
                    'chol': 'cholesterol OR "total cholesterol"',
                    'trig': 'triglyceride OR triglycerides',
                    'trr': 'triglyceride OR triglycerides',
                    'vldl': 'VLDL OR lipoprotein',
                    'ldl': 'LDL OR "low density lipoprotein"',
                    'hdl': 'HDL OR "high density lipoprotein"',
                    'hr': '"heart rate" OR pulse OR tachycardia OR bradycardia',
                    'bp_med': '"blood pressure medication" OR antihypertensive',
                    'female': 'female OR sex OR gender',
                    'raceclass': '"Continental Population Groups"[Mesh] OR "Black"[tiab] OR "White"[tiab] OR race[tiab]',
                    'race_black': '"Continental Population Groups"[Mesh] OR "Black"[tiab] OR "White"[tiab] OR race[tiab]',
                    'statin': 'statin OR lipid-lowering',
                    'aspirin': 'aspirin OR antiplatelet',
                    'x4smoke': 'smoking OR smoker OR tobacco',
                    'smoke_3cat': 'smoking OR smoker OR tobacco',
                    'cvd_hx_baseline': '"history of cardiovascular disease" OR "prior cardiovascular disease" OR "prior MI" OR "prior stroke"',
                    'sub_cvd': '"history of cardiovascular disease" OR "prior cardiovascular disease" OR "prior MI" OR "prior stroke"',
                    'anti_coag': 'anticoagulant OR anticoagulation OR warfarin OR heparin OR "blood thinner" OR coagulation OR "anticoagulant therapy" OR "oral anticoagulant" OR apixaban OR rivaroxaban OR dabigatran OR "thrombin inhibitor" OR "factor Xa inhibitor" OR "coagulation status" OR "thrombotic risk" OR "antithrombotic"'
                }
            },
            'crash_2': {
                'context_terms': ['trauma', 'bleeding', 'hemorrhage'],
                'treatment_terms': ['"tranexamic acid"', 'TXA', '"anti-fibrinolytic"', 'CRASH-2'],
                'features': {
                    'iage': 'age OR elderly OR geriatric',
                    'isbp': '"systolic blood pressure" OR "initial blood pressure" OR hypotension OR "hemorrhagic shock" OR "shock index"',
                    'irr': '"respiratory rate" OR breathing OR tachypnea',
                    'icc': '"capillary refill" OR perfusion OR shock OR "peripheral perfusion" OR "shock severity"',
                    'ihr': '"heart rate" OR pulse OR tachycardia OR bradycardia',
                    'ninjurytime': '"time from injury" OR "injury-to-treatment time" OR "treatment delay" OR "time to treatment" OR "early treatment" OR "delayed treatment" OR "treatment timing"',
                    'igcs': 'GCS OR "Glasgow Coma Scale" OR "consciousness level"',
                    'isex': 'sex OR gender OR male OR female',
                    'iinjurytype': '"injury type" OR "penetrating injury" OR "blunt trauma" OR "mechanism of injury" OR "penetrating trauma" OR "blunt injury" OR "injury mechanism" OR "injury pattern"',
                    'iinjurytype_1': '"blunt trauma" OR "blunt injury"',
                    'iinjurytype_2': '"penetrating injury" OR "penetrating trauma" OR gunshot OR stabbing'
                }
            },
            'txa': {
                'context_terms': [
                    'trauma',
                    'bleeding',
                    'hemorrhage',
                    'survival',
                    'mortality',
                    '"in-hospital mortality"',
                    '"pre-hospital"',
                    '"prehospital"',
                    'EMS',
                    '"emergency medical services"',
                ],
                'treatment_terms': [
                    '"tranexamic acid"',
                    'TXA',
                    '"anti-fibrinolytic"',
                    '"pre-hospital TXA"',
                    '"prehospital TXA"',
                    '"prehospital tranexamic acid"',
                ],
                # Reuse CRASH-2 feature query templates for harmonized TXA features.
                'features': {
                    'iage': 'age OR elderly OR geriatric',
                    'isbp': '"systolic blood pressure" OR "initial blood pressure" OR hypotension OR "hemorrhagic shock" OR "shock index"',
                    'irr': '"respiratory rate" OR breathing OR tachypnea',
                    'icc': '"capillary refill" OR perfusion OR shock OR "peripheral perfusion" OR "shock severity"',
                    'ihr': '"heart rate" OR pulse OR tachycardia OR bradycardia',
                    'ninjurytime': '"time from injury" OR "injury-to-treatment time" OR "treatment delay" OR "time to treatment" OR "early treatment" OR "delayed treatment" OR "treatment timing"',
                    'igcs': 'GCS OR "Glasgow Coma Scale" OR "consciousness level"',
                    'isex': 'sex OR gender OR male OR female',
                    'iinjurytype': '"injury type" OR "penetrating injury" OR "blunt trauma" OR "mechanism of injury" OR "penetrating trauma" OR "blunt injury" OR "injury mechanism" OR "injury pattern"',
                    'iinjurytype_1': '"blunt trauma" OR "blunt injury"',
                    'iinjurytype_2': '"penetrating injury" OR "penetrating trauma" OR gunshot OR stabbing'
                }
            },
            'accord_glycemia': {
                'context_terms': ['diabetes', '"type 2 diabetes"', 'cardiovascular'],
                'treatment_terms': [
                    '"intensive glycemic control"',
                    '"intensive glucose lowering"',
                    '"intensive glucose control"',
                    '"tight glycemic control"',
                    '"aggressive glycemic control"',
                    '"HbA1c target"',
                ],
                'features': {
                    'baseline_age': 'age OR elderly OR geriatric',
                    'bmi': 'BMI OR obesity OR "body mass index" OR overweight OR adiposity',
                    'hba1c': '"baseline HbA1c" OR "baseline A1c" OR "entry HbA1c" OR "initial HbA1c" OR "presenting HbA1c" OR "baseline glycated hemoglobin" OR HbA1c[tiab]',
                    'yrsdiab': '"diabetes duration" OR "disease duration" OR "years of diabetes"',
                    'sbp': '"systolic blood pressure" OR hypertension',
                    'dbp': '"diastolic blood pressure" OR hypertension',
                    'hr': '"heart rate" OR pulse OR tachycardia OR bradycardia',
                    'fpg': '"fasting plasma glucose" OR FPG OR glucose OR hyperglycemia',
                    'alt': 'ALT OR "alanine aminotransferase" OR "liver function"',
                    'cpk': 'CPK OR "creatine phosphokinase" OR "creatine kinase" OR rhabdomyolysis',
                    'potassium': 'potassium OR hyperkalemia OR hypokalemia OR electrolyte',
                    'gfr': 'GFR OR eGFR OR "renal function" OR "kidney function"',
                    'uacr': 'UACR OR albuminuria OR "albumin creatinine ratio"',
                    'trig': 'triglyceride OR triglycerides',
                    'ldl': 'LDL OR "low density lipoprotein"',
                    'hdl': 'HDL OR "high density lipoprotein"',
                    'bp_med': '"blood pressure medication" OR antihypertensive',
                    'dm_med': '"diabetes medication" OR "oral hypoglycemic" OR "antidiabetic"',
                    'female': 'female OR sex OR gender',
                    'raceclass': '"Continental Population Groups"[Mesh] OR "Black"[tiab] OR "White"[tiab] OR race[tiab]',
                    'cvd_hx_baseline': '"history of cardiovascular disease" OR "prior cardiovascular disease" OR "prior MI" OR "prior stroke"',
                    'insulin': 'insulin OR "insulin therapy" OR "insulin use" OR "exogenous insulin"',
                    'statin': 'statin OR lipid-lowering',
                    'aspirin': 'aspirin OR antiplatelet',
                    'antiarrhythmic': 'antiarrhythmic OR "anti-arrhythmic" OR "rhythm control"',
                    'anti_coag': 'anticoagulant OR anticoagulation OR warfarin OR heparin',
                    'x4smoke': 'smoking OR smoker OR tobacco',
                }
            },
            'sprint': {
                'context_terms': ['hypertension', '"blood pressure"', 'cardiovascular'],
                'treatment_terms': [
                    '"intensive blood pressure control"',
                    '"intensive systolic blood pressure"',
                    '"systolic blood pressure target"',
                    '"tight blood pressure control"',
                    '"aggressive blood pressure lowering"',
                    '"blood pressure management"',
                    '"intensive antihypertensive therapy"',
                    '"antihypertensive intensification"',
                    'SPRINT'
                ],
                'features': {
                    'age': 'age OR elderly OR geriatric OR "older adults" OR "older patients" OR frailty OR "frail older"',
                    'sbp': '"baseline systolic blood pressure" OR "pre-randomization SBP" OR "initial SBP" OR "J-curve" OR "SBP threshold"',
                    'dbp': '"baseline diastolic blood pressure" OR "low diastolic BP" OR "pulse pressure" OR "diastolic J-curve" OR "DBP threshold"',
                    'n_agents': '"number of antihypertensive agents" OR polypharmacy OR antihypertensive',
                    'egfr': 'eGFR OR GFR OR "renal function"',
                    'screat': 'creatinine OR "serum creatinine"',
                    'chr': 'cholesterol OR "total cholesterol"',
                    'glur': 'glucose OR "fasting glucose" OR glycemia OR hyperglycemia',
                    'hdl': 'HDL OR "high density lipoprotein"',
                    'trr': 'triglyceride OR triglycerides',
                    'umalcr': 'UACR OR albuminuria OR "albumin creatinine ratio"',
                    'bmi': 'BMI OR obesity OR "body mass index" OR "body mass" OR overweight OR adiposity OR "abdominal obesity" OR "waist circumference" OR "weight status"',
                    'female': 'female OR sex OR gender',
                    'race_black': '"African Americans"[Mesh] OR "Black"[tiab] OR "African American"[tiab]',
                    'smoke_3cat': 'smoking OR smoker OR tobacco',
                    'aspirin': 'aspirin OR antiplatelet',
                    'statin': 'statin OR lipid-lowering',
                    'sub_cvd': '"cardiovascular disease" OR CVD OR "heart disease"',
                    'sub_ckd': '"chronic kidney disease" OR CKD OR "renal impairment"'
                }
            }
        }

        config = dataset_config.get(dataset, {
            'context_terms': [], 'treatment_terms': [], 'features': {}
        })

        # Per-dataset competing-intervention exclusion terms
        negative_filters: Dict[str, List[str]] = {
            'ist3':    ['tenecteplase[tiab]', 'sonothrombolysis[tiab]',
                        '"mechanical thrombectomy"[tiab]', '"endovascular thrombectomy"[tiab]',
                        'thrombectomy[tiab]', 'urokinase[tiab]', 'desmoteplase[tiab]'],
            'crash_2': ['aminocaproic[tiab]', 'aprotinin[tiab]',
                        '"epsilon-aminocaproic"[tiab]', 'fibrinogen[tiab]'],
            'accord':  ['fenofibrate[tiab]', '"intensive glycemic"[tiab]',
                        '"glycemic arm"[tiab]'],
            'accord_glycemia': ['fenofibrate[tiab]', '"intensive blood pressure"[tiab]',
                        '"blood pressure arm"[tiab]'],
            'sprint':  [],
        }

        query_parts = []

        # 1. Clinical Context
        context_parts = []
        if config['treatment_terms']: context_parts.append(f"({' OR '.join(config['treatment_terms'])})")
        if config['context_terms']: context_parts.append(f"({' OR '.join(config['context_terms'])})")
        if context_parts: query_parts.append(f"({' AND '.join(context_parts)})")

        # 2. The Feature
        feature_query = self._find_feature_query(feature_name, config['features'])
        if feature_query:
            query_parts.append(f"({feature_query})")
        else:
            clean = self._normalize_text(feature_name)
            query_parts.append(f'"{clean.replace(" ", " AND ")}"')

        # 3. Mechanism Specifics (New)
        if use_mechanism_keywords:
            mech_desc = mechanism.get('description', '')
            mech_keywords = self._extract_mechanism_keywords(mech_desc)
            if mech_keywords:
                query_parts.append(f"({mech_keywords})")

        # 4. Interaction Terms
        if require_interaction:
            interaction_terms = [
                '"effect modification"[tiab]', '"treatment effect heterogeneity"[tiab]',
                '"heterogeneous treatment effect"[tiab]', '"treatment interaction"[tiab]',
                '"interaction effect"[tiab]', '"subgroup analysis"[tiab]',
                '"differential treatment effect"[tiab]', '"treatment-by"[tiab]',
                '"predictive factor"[tiab]', '"interaction term"[tiab]',
                '"forest plot"[tiab]', 'HTE[tiab]',
                '"prespecified subgroup"[tiab]', '"pre-specified subgroup"[tiab]',
                '"subgroup"[tiab]', '"effect modifier"[tiab]',
                '"interaction p"[tiab]', '"p for interaction"[tiab]',
                '"moderator"[tiab]', '"treatment-covariate"[tiab]'
            ]
            query_parts.append(f"({' OR '.join(interaction_terms)})")
        full_query = ' AND '.join(query_parts)

        # 5. Document Types
        if include_doc_types:
            doc_types = ['Clinical Trial[PT]', 'Randomized Controlled Trial[PT]', 'Meta-Analysis[PT]', 'Review[PT]']
            full_query += f" AND ({' OR '.join(doc_types)})"

        # 6. Exclude competing interventions (tier-1 and tier-2 only)
        if include_doc_types:  # proxy: tier-1 / tier-2 have doc_types; tier-3 doesn't
            excl = negative_filters.get(dataset, [])
            if excl:
                full_query += f" NOT ({' OR '.join(excl)})"

        return full_query

    def search_pubmed(self, query: str, max_results: int = None) -> List[str]:
        if not Entrez: return []
        max_results = max_results or self.max_abstracts

        cache_key = (query, int(max_results))
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        # Retry logic for transient PubMed errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                result = record.get("IdList", [])
                self._search_cache[cache_key] = result
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"PubMed error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error searching PubMed after {max_retries} attempts: {e}")
                    self._search_cache[cache_key] = []
                    return []
        return []

    def _fetch_pmc_full_text(self, pmids: List[str]) -> Dict[str, str]:
        """Return {pmid: full_text} for PMIDs available in PubMed Central (PMC-OA).
        Extracts title + abstract + results/discussion sections from JATS XML.
        Falls back silently for paywalled articles not in PMC."""
        if not Entrez or not pmids:
            return {}
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            return {}

        pmc_map: Dict[str, str] = {}  # pmid -> full text

        # Step 1: resolve PMIDs -> PMC IDs via elink
        pmid_to_pmcid: Dict[str, str] = {}
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmids, linkname="pubmed_pmc")
            linksets = Entrez.read(handle)
            handle.close()
            for ls in linksets:
                src_pmid = str(ls.get("IdList", ["?"])[0])
                for lsdb in ls.get("LinkSetDb", []):
                    if lsdb.get("LinkName") == "pubmed_pmc":
                        for link in lsdb.get("Link", []):
                            pmid_to_pmcid[src_pmid] = str(link["Id"])
                            break
            time.sleep(0.34)
        except Exception as e:
            print(f"  PMC elink error: {e}")
            return {}

        if not pmid_to_pmcid:
            return {}

        # Step 2: fetch JATS XML for each PMC ID and extract relevant sections
        RELEVANT_SECTIONS = {"results", "discussion", "conclusions", "abstract", "methods"}
        for pmid, pmcid in pmid_to_pmcid.items():
            try:
                handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
                raw_xml = handle.read()
                handle.close()
                time.sleep(0.34)

                root = ET.fromstring(raw_xml)
                parts: List[str] = []

                # Title
                for t in root.iter("article-title"):
                    parts.append(ET.tostring(t, encoding="unicode", method="text").strip())
                    break

                # Abstract
                for ab in root.iter("abstract"):
                    parts.append(ET.tostring(ab, encoding="unicode", method="text").strip())

                # Body sections: only keep results/discussion/conclusions
                for sec in root.iter("sec"):
                    sec_type = (sec.get("sec-type") or "").lower()
                    # also check first <title> child text
                    title_el = sec.find("title")
                    title_text = (ET.tostring(title_el, encoding="unicode", method="text").strip().lower()
                                  if title_el is not None else "")
                    if any(kw in sec_type or kw in title_text for kw in RELEVANT_SECTIONS):
                        parts.append(ET.tostring(sec, encoding="unicode", method="text").strip())

                full_text = "\n\n".join(parts)
                # Truncate to ~12 000 chars to stay within LLM context window
                pmc_map[pmid] = full_text[:12000]
                print(f"  [PMC full-text] PMID {pmid} -> PMC{pmcid}: {len(full_text)} chars fetched")
            except Exception as e:
                print(f"  [PMC full-text] PMID {pmid} fetch error: {e}")

        return pmc_map

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        if not Entrez or not pmids: return []

        cache_key = tuple(pmids)
        if cache_key in self._abstract_batch_cache:
            return self._abstract_batch_cache[cache_key]

        # Optional: pre-fetch PMC full text for articles that have it
        pmc_full_text: Dict[str, str] = {}
        if self.full_text:
            print(f"  [PMC full-text] resolving {len(pmids)} PMIDs via elink...")
            pmc_full_text = self._fetch_pmc_full_text(pmids)
            print(f"  [PMC full-text] {len(pmc_full_text)}/{len(pmids)} articles found in PMC")

        abstracts = []
        try:
            batch_size = 10
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]

                # Retry logic for transient errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="abstract", retmode="xml")
                        records = Entrez.read(handle)
                        handle.close()
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"  Fetch error (attempt {attempt+1}/{max_retries}): {e}")
                            time.sleep(wait_time)
                        else:
                            print(f"  Failed to fetch batch after {max_retries} attempts: {e}")
                            records = {'PubmedArticle': []}  # Empty result

                for record in records.get('PubmedArticle', []):
                    try:
                        article = record['MedlineCitation']['Article']
                        pmid = str(record['MedlineCitation']['PMID'])
                        title = article.get('ArticleTitle', '')
                        text = ''
                        if 'Abstract' in article:
                            parts = article['Abstract'].get('AbstractText', [])
                            text = ' '.join([str(p) for p in parts]) if isinstance(parts, list) else str(parts)
                        # Replace abstract with PMC full text if available
                        if pmid in pmc_full_text:
                            text = pmc_full_text[pmid]
                        abstracts.append({'pmid': pmid, 'title': title, 'abstract': text})
                    except Exception: continue
                time.sleep(0.34)
        except Exception as e:
            print(f"Error fetching abstracts: {e}")
        self._abstract_batch_cache[cache_key] = abstracts
        return abstracts

    def _llm_cache_key(self, abstract: Dict[str, str], mechanism: Dict[str, Any], feature_name: str, dataset: str) -> str:
        payload = {
            "dataset": dataset,
            "feature_name": feature_name,
            "mechanism_type": mechanism.get("mechanism_type", ""),
            "mechanism_description": mechanism.get("description", ""),
            "mechanism_direction": mechanism.get("effect_direction", "unknown"),
            "pmid": abstract.get("pmid", ""),
            "title": abstract.get("title", ""),
            "abstract": abstract.get("abstract", ""),
            "model": self.model,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def analyze_abstract_with_llm(self, abstract: Dict[str, str], mechanism: Dict[str, Any], feature_name: str, dataset: str) -> Dict[str, Any]:
        """
        Analyze abstract with Strict Feature Alignment to prevent false positives.
        """
        if not self.openai_client:
            return self.analyze_abstract_keyword(abstract, mechanism)

        cache_key = self._llm_cache_key(abstract, mechanism, feature_name, dataset)
        cached = self._llm_eval_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        cohort_context = self.get_cohort_trial_context(dataset)
        mech_desc = mechanism.get('description', '')
        mech_dir = mechanism.get('effect_direction', 'unknown')
        feature_concept = feature_name.replace("_", " ").title()

        prompt = f"""
        You are an expert Biostatistician and Medical Researcher.

        YOUR TASK: Evaluate if the Clinical Abstract provides evidence supporting the proposed Mechanism.
        ---
        1. THE EXPLANATION:
           - Trial Context: {dataset.upper()} ({cohort_context['treatment']} vs {cohort_context['comparator']})
           - Outcome: {cohort_context['outcome']}
           - TARGET FEATURE: "{feature_concept}"
           - Mechanism: {mech_desc}
           - Expected Direction: {mech_dir}

        2. THE ABSTRACT:
           - Title: {abstract.get('title', '')}
           - Text: {abstract.get('abstract', '')}

        ---
        3. STEP 1: RELEVANCE GATES (Pass/Fail)
        (A) Treatment Check: Is {cohort_context['treatment']} (or class equivalent) evaluated?
        (B) Outcome Alignment: Must evaluate '{cohort_context['outcome']}' or surrogate.
        (C) Feature Check: Is {feature_concept} explicitly analyzed for outcome association?

            - REJECT if feature is missing.
            - REJECT if feature is only a covariate/baseline stat, but not linked to outcome.

        (D) Intervention Match: The abstract must test {cohort_context['treatment']} (or a direct class equivalent) as a PRIMARY arm — not merely as a comparator arm when the study is actually evaluating a different agent.
            - REJECT if the study primarily evaluates a DIFFERENT intervention (e.g., a competing drug, device, or technique) that happens to use {cohort_context['treatment']} as control.
            - Example REJECT: A meta-analysis of tenecteplase vs alteplase should be IRRELEVANT for an alteplase vs placebo mechanism, even if it reports NIHSS subgroups.
            - Example PASS: A pooled analysis of alteplase RCTs, or a secondary analysis of a trial where alteplase was the active arm.

        (E) Population Context: The study population must be reasonably comparable to: {cohort_context['population']}.
            - REJECT if the abstract studies a fundamentally DIFFERENT disease entity, even if the same drug is used.
            - Different severity, age range, or geography within the same disease is acceptable.
            - Example REJECT (crash_2): A TXA study in aneurysmal subarachnoid hemorrhage (SAH) — a neurological condition — is IRRELEVANT for a trauma-hemorrhage mechanism because the underlying coagulation pathophysiology and clinical context differ fundamentally from traumatic bleeding.
            - Example REJECT (ist3): A thrombolysis study exclusively in hemorrhagic stroke patients is IRRELEVANT for an ischemic stroke mechanism.
            - Example PASS: A TXA study in post-partum hemorrhage or surgical bleeding may be relevant to a crash_2 trauma mechanism if the coagulation dynamics are directly analogous and the authors explicitly discuss generalizability.

        -> If (A) or (B) or (C) or (D) or (E) fails, output: [F] IRRELEVANT.
        ---
        4. STEP 2: EVIDENCE EVALUATION (The "Mechanism Test")
            If the abstract passes Step 1, compare the REPORTED RESULTS against the EXPECTED CLINICAL OUTCOME.

            [A] SUPPORT_INTERACTION (Mechanism Supported):
                - Core Requirement: Explicit statement that the magnitude of treatment benefit is SIGNIFICANTLY DIFFERENT between subgroups defined by {feature_concept}.
                - Directionality: The difference must match the expected direction (e.g., "Greater benefit in High-Risk group" if that was your explanation).
                - Statistical Evidence:
                    * Includes explicit numeric evidence (e.g., "Interaction P < 0.05").
                    * Includes strong textual claims (e.g., "Treatment efficacy was significantly superior in [subgroup] compared to [other subgroup]").
                    * Note: A statement like "Patients with X had longer survival" is insufficient unless it adds "...specifically in the treatment arm" or "...compared to placebo."
                - ABSOLUTE vs RELATIVE SCALE: Both count. A paper reporting consistent relative benefit (similar OR/RR across subgroups) but explicitly noting greater ABSOLUTE benefit (larger ARR, lower NNT) in the predicted subgroup is SUPPORT_INTERACTION for an absolute-benefit mechanism. SHAP captures absolute treatment effect — so evidence of greater absolute risk reduction in the predicted subgroup qualifies as SUPPORT_INTERACTION even if the relative risk ratio is consistent across subgroups.
                * Mechanism Independence: The abstract does NOT need to explain the biological "why." If the clinical numbers match your prediction, it counts.

            [B] SUPPORT_WEAK (Mechanism Consistent):
                - Core Requirement: The abstract describes a numerical trend or subgroup observation that favors the mechanism, but explicitly notes it is not statistically significant or is explanation-generating only.
                - Key Indicators:
                   * P-values for interaction are > 0.05 (e.g., $p=0.09$, $p=0.12$).
                   * Language like "numeric trend," "suggests a benefit," "exploratory analysis," or "promising signal."
                   * Post-hoc analyses not originally powered for significance.
                - Differentiation from [D] (Null):
                  * [B] says "There is a signal, but we can't prove it yet." (e.g., HR 0.7 vs HR 0.9, p=0.15).
                  * [D] says "There is NO signal." (e.g., HR 0.8 vs HR 0.8, p=0.90).
                - Differentiation from [F] (Irrelevant):
                  * [B] must still show data relevant to the specific feature/drug pair. A general statement that "more research is needed" without data is [F].

            [C] PROGNOSTIC_MAIN_EFFECT (Mechanism Unclear):
                - Core Rule: Evidence that the feature predicts the Outcome (e.g., Survival) generally, but NOT the response to the specific drug.
                  * Scenario 1: The feature is a general risk factor (e.g., "Old age predicted higher mortality in both arms").
                  * Scenario 2: The abstract mentions the feature's effect on survival but is silent on whether it changed the drug's efficacy.

            [D] NO_INTERACTION (Mechanism Inactive):
                - Core Rule: Explicit statement that the feature does NOT modify the treatment effect.
                  * Key Phrases: "Outcomes were similar regardless of status," "Interaction p > 0.05," "Consistent benefit across subgroups."
                  * Note: A non-significant trend (p=0.15) often falls here unless the author explicitly calls it "promising" (which moves it to [B]).
                - SCALE CAVEAT: If the mechanism is framed as an ABSOLUTE benefit interaction, a paper reporting only consistent relative benefit (same OR/RR across strata) should be [C] PROGNOSTIC_MAIN_EFFECT rather than [D] — absolute benefit still differs by baseline risk even when relative benefit is constant. Only classify [D] if the paper explicitly states absolute benefit was also equivalent across the relevant subgroups.

            [E] CONFLICT (Mechanism Contradicted):
                - Core Rule: The abstract reports a Significant Interaction in the OPPOSITE direction of the explanation.
                  * Example: You predicted the feature would enhance drug efficacy, but the data shows it reduces efficacy or causes harm relative to the control group.
                  * Crucial Distinction: The drug must perform worse than the comparator in this subgroup (or significantly worse than in the other subgroup). If the subgroup just has a poor baseline prognosis but the drug still helps them a little, that is [C], not [E].
                  * Intervention Requirement: The conflicting result must come from a trial/analysis where {cohort_context['treatment']} is the active treatment arm — not a related drug being compared against {cohort_context['treatment']}. A study showing a competing agent outperforms {cohort_context['treatment']} in high-NIHSS patients is NOT a CONFLICT for {cohort_context['treatment']}; it is IRRELEVANT.

            5. STEP 3: STUDY DESIGN CLASSIFICATION
            Classify the study based solely on the abstract:

            "RCT"                       — Primary randomized controlled trial (original allocation)
            "RCT_secondary"             — Secondary / post-hoc / subgroup analysis of an RCT
            "systematic_review_meta_analysis" — Pooled evidence synthesis (SR or MA)
            "prospective_cohort"        — Prospective observational cohort
            "retrospective_cohort"      — Retrospective observational cohort or registry
            "case_control"              — Case-control study
            "cross_sectional"           — Cross-sectional or survey study
            "case_series"               — Case series or case report
            "narrative_review"          — Expert opinion, narrative or scoping review
            "other"                     — Cannot determine or does not fit above

            6. OUTPUT FORMAT:
            Return a valid JSON object with the following fields:
            {{
                "classification": "SUPPORT_INTERACTION | SUPPORT_WEAK | PROGNOSTIC_MAIN_EFFECT | NO_INTERACTION | CONFLICT | IRRELEVANT",
                "confidence": "high | medium | low",
                "study_design": "RCT | RCT_secondary | systematic_review_meta_analysis | prospective_cohort | retrospective_cohort | case_control | cross_sectional | case_series | narrative_review | other",
                "reasoning": "Explain why it matches the TARGET FEATURE and whether it supports the specific mechanism claim.",
                "evidence_quote": "Quote proving the interaction involves {feature_concept}."
            }}
            """

        try:
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an evidence-based medicine evaluator. Be calibrated and avoid overly strict judgments. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }

            model_lc = str(self.model).lower()
            if not model_lc.startswith("gpt-5"):
                request_kwargs["temperature"] = 0.0

            response = self.openai_client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"): content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"): content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Normalize classification to canonical labels
            classification_raw = str(result.get('classification', 'IRRELEVANT')).strip()
            classification_key = classification_raw.upper()
            letter_to_label = {
                'A': 'SUPPORT_INTERACTION',
                'B': 'SUPPORT_WEAK',
                'C': 'PROGNOSTIC_MAIN_EFFECT',
                'D': 'NO_INTERACTION',
                'E': 'CONFLICT',
                'F': 'IRRELEVANT',
            }
            allowed_labels = {
                'SUPPORT_INTERACTION',
                'SUPPORT_WEAK',
                'PROGNOSTIC_MAIN_EFFECT',
                'NO_INTERACTION',
                'CONFLICT',
                'IRRELEVANT',
            }

            if classification_key in letter_to_label:
                classification = letter_to_label[classification_key]
            elif classification_key == 'PROGNOSTIC_ONLY':
                classification = 'PROGNOSTIC_MAIN_EFFECT'
            elif classification_key in allowed_labels:
                classification = classification_key
            else:
                classification = 'IRRELEVANT'

            result['classification'] = classification

            # Map to stance
            cls = result.get('classification', 'IRRELEVANT')
            if cls in ['SUPPORT_INTERACTION', 'SUPPORT_WEAK']: stance = 'support'
            elif cls == 'CONFLICT': stance = 'conflict'
            else: stance = 'neutral'

            result['stance'] = stance
            result['pmid'] = abstract.get('pmid', '')
            result['title'] = abstract.get('title', '')
            result['analysis_method'] = 'llm_strict_v2'

            # Normalise study_design to allowed values
            valid_study_designs = {
                'rct', 'rct_secondary', 'systematic_review_meta_analysis',
                'prospective_cohort', 'retrospective_cohort', 'case_control',
                'cross_sectional', 'case_series', 'narrative_review', 'other',
            }
            raw_design = str(result.get('study_design', 'other')).lower().strip()
            result['study_design'] = raw_design if raw_design in valid_study_designs else 'other'

            self._llm_eval_cache[cache_key] = dict(result)
            return result
        except Exception as e:
            print(f"Error with LLM analysis: {e}")
            return self.analyze_abstract_keyword(abstract, mechanism)

    def analyze_abstract_keyword(self, abstract: Dict[str, str], mechanism: Dict[str, Any]) -> Dict[str, Any]:
        """Simple keyword fallback."""
        text = (abstract.get('title', '') + ' ' + abstract.get('abstract', '')).lower()
        description = mechanism.get('description', '').lower()
        mech_terms = [w for w in re.findall(r'\b[a-z]{4,}\b', description) if w not in {'with', 'that', 'from'}]

        matches = sum(1 for t in mech_terms if t in text)
        ratio = matches / len(mech_terms) if mech_terms else 0

        support_kws = ['interaction', 'modifies', 'subgroup', 'heterogeneity', 'differential', 'predictive']
        has_support = any(k in text for k in support_kws)

        stance = 'support' if has_support and ratio > 0.2 else 'neutral'
        return {
            'pmid': abstract.get('pmid', ''), 'title': abstract.get('title', ''),
            'stance': stance, 'confidence': 'low', 'analysis_method': 'keyword',
            'reasoning': f"Keyword match ratio: {ratio:.2f}"
        }

    def validate_mechanism(self, feature_name: str, mechanism: Dict[str, Any], dataset: str, use_llm: bool = True) -> Dict[str, Any]:
        print(f"\nValidating mechanism for {feature_name} ({mechanism.get('mechanism_type', 'unknown')})")

        # Tiered Search Strategy
        tiers = [
            ("tier_1_mechanism_specific", self.construct_search_query(feature_name, mechanism, dataset, True, True, True)),
            ("tier_2_strict_interaction", self.construct_search_query(feature_name, mechanism, dataset, True, True, False)),
            ("tier_3_broad_fallback", self.construct_search_query(feature_name, mechanism, dataset, True, False, False))
        ]

        pmids = []
        used_tier = ""
        query = ""

        for tier_name, tier_query in tiers:
            print(f"  Search [{tier_name}]: {tier_query[:100]}...")
            # Tier 1 is very specific, so we accept fewer results to avoid noise
            limit = 15 if "mechanism" in tier_name else self.max_abstracts
            found = self.search_pubmed(tier_query, max_results=limit)
            print(f"    Found {len(found)} articles.")
            # Require minimum 3 abstracts except for the last tier (broad fallback)
            min_threshold = 5 if tier_name != "tier_3_broad_fallback" else 1
            if len(found) >= min_threshold:
                pmids = found
                used_tier = tier_name
                query = tier_query
                break

        # Remove permanently blacklisted PMIDs and source-trial PMIDs before fetching
        _trial_pmids = self.TRIAL_SOURCE_PMIDS.get(dataset, set())
        pmids = [p for p in pmids if p not in self.PMID_BLACKLIST and p not in _trial_pmids]

        if not pmids:
            return {'feature_name': feature_name, 'mechanism': mechanism, 'total_abstracts': 0,
                    'support_count': 0, 'conflict_count': 0, 'neutral_count': 0, 'abstracts_analyzed': []}

        abstracts = self.fetch_abstracts(pmids)
        analyses = []

        for abs_data in abstracts:
            if use_llm:
                # Pass feature_name explicitly
                res = self.analyze_abstract_with_llm(abs_data, mechanism, feature_name, dataset)
                if self.llm_delay > 0:
                    time.sleep(self.llm_delay)
            else:
                res = self.analyze_abstract_keyword(abs_data, mechanism)
            analyses.append(res)

        # ── Adaptive budget ───────────────────────────────────────────────
        # If fewer than 2 abstracts passed relevance gates, expand to next tiers
        MIN_RELEVANT = 2
        relevant_count = sum(
            1 for a in analyses
            if a.get('classification', 'IRRELEVANT') != 'IRRELEVANT'
        )
        if relevant_count < MIN_RELEVANT:
            tier_idx = next(
                (i for i, t in enumerate(tiers) if t[0] == used_tier), -1
            )
            seen_pmids = set(pmids)
            for next_tier_name, next_tier_query in tiers[tier_idx + 1:]:
                print(f"  Adaptive expansion [{next_tier_name}]: "
                      f"only {relevant_count} relevant found, expanding...")
                extra_found = self.search_pubmed(
                    next_tier_query, max_results=self.max_abstracts
                )
                extra_pmids = [p for p in extra_found if p not in seen_pmids and p not in self.PMID_BLACKLIST and p not in _trial_pmids]
                if not extra_pmids:
                    continue
                seen_pmids.update(extra_pmids)
                extra_abstracts = self.fetch_abstracts(extra_pmids)
                for abs_data in extra_abstracts:
                    if use_llm:
                        res = self.analyze_abstract_with_llm(
                            abs_data, mechanism, feature_name, dataset
                        )
                        if self.llm_delay > 0:
                            time.sleep(self.llm_delay)
                    else:
                        res = self.analyze_abstract_keyword(abs_data, mechanism)
                    analyses.append(res)
                relevant_count = sum(
                    1 for a in analyses
                    if a.get('classification', 'IRRELEVANT') != 'IRRELEVANT'
                )
                used_tier = next_tier_name
                query = next_tier_query
                if relevant_count >= MIN_RELEVANT:
                    break

        support = sum(1 for a in analyses if a['stance'] == 'support')
        conflict = sum(1 for a in analyses if a['stance'] == 'conflict')
        neutral = len(analyses) - support - conflict

        return {
            'feature_name': feature_name,
            'mechanism': mechanism,
            'query': query,
            'query_tier_used': used_tier,
            'total_abstracts': len(analyses),
            'support_count': support,
            'conflict_count': conflict,
            'neutral_count': neutral,
            'abstracts_analyzed': analyses
        }

    def validate_all_mechanisms(self, explanations_file: str, use_llm: bool = True) -> Dict[str, Any]:
        explanations = self.load_explanations(explanations_file)
        dataset = explanations.get('dataset', 'unknown').lower().replace('-', '_')
        all_results = []

        for feat_hyp in explanations.get('feature_explanations', []):
            feat_name = feat_hyp.get('feature_name', 'unknown')
            for mech in feat_hyp.get('mechanisms', []):
                all_results.append(self.validate_mechanism(feat_name, mech, dataset, use_llm))

        return {
            'dataset': dataset,
            'total_mechanisms_analyzed': len(all_results),
            'total_abstracts_retrieved': sum(r['total_abstracts'] for r in all_results),
            'overall_support_count': sum(r['support_count'] for r in all_results),
            'overall_conflict_count': sum(r['conflict_count'] for r in all_results),
            'evaluator_model': self.model,
            'mechanism_results': all_results
        }

    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        print("\n" + "="*60)
        print(f"VALIDATION REPORT: {results['dataset'].upper()}")
        print(f"Support: {results['overall_support_count']} | Conflict: {results['overall_conflict_count']}")
        print("="*60)

        for r in results['mechanism_results']:
            if r['total_abstracts'] > 0:
                print(f"\nFeature: {r['feature_name']}")
                print(f"Tier Used: {r.get('query_tier_used', 'N/A')}")
                print(f"Stance: {r['support_count']} Support / {r['neutral_count']} Neutral / {r['conflict_count']} Conflict")

                # Study design breakdown
                designs: Dict[str, int] = {}
                for a in r.get('abstracts_analyzed', []):
                    d = a.get('study_design', 'other')
                    designs[d] = designs.get(d, 0) + 1
                if designs:
                    design_str = ', '.join(f"{k}: {v}" for k, v in sorted(designs.items()))
                    print(f"Study Designs: {design_str}")

        if output_file:
            write_json_file(output_file, results)
            print(f"\nSaved to: {output_file}")

def main():
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str)
    parser.add_argument('--dataset', type=str, default='ist3')
    parser.add_argument('--email', type=str, default='research@example.com')
    parser.add_argument('--api-key', type=str)
    parser.add_argument('--max-abstracts', type=int, default=30)
    parser.add_argument('--model', type=str, default='gpt-5-mini')
    parser.add_argument('--api-provider', type=str, default='openai', choices=['openai', 'openrouter'])
    parser.add_argument('--api-base-url', type=str, default=None)
    parser.add_argument('--llm-delay', type=float, default=0.5, help='Seconds to sleep between LLM abstract evaluations (set 0 for max speed).')
    parser.add_argument('--full-text', action='store_true', default=False, help='Fetch full article text from PubMed Central (PMC-OA) where available, falling back to abstract.')
    args = parser.parse_args()

    if args.api_provider == 'openrouter':
        api_key = args.api_key  # constructor will resolve OPENROUTER_API_KEY from env
    else:
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    validator = PubMedMechanismValidator(
        email=args.email,
        api_key=api_key,
        max_abstracts=args.max_abstracts,
        model=args.model,
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        llm_delay=args.llm_delay,
        full_text=args.full_text,
    )

    results = validator.validate_all_mechanisms(args.input, use_llm=True)

    # Default output path: same directory as input file.
    # When a non-default judge model is used, the model name is appended so
    # results from different judge models coexist without overwriting each other.
    judge_suffix = f"__{args.model}" if args.model != "gpt-5-mini" else ""
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)),
        f"explanations_pubmed_validation{judge_suffix}.json"
    )
    validator.generate_report(results, output_path)

if __name__ == '__main__':
    main()