from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ArticleMetadata(BaseModel):
    title: str = Field(..., description="Article title")
    authors: Optional[str] = Field(None, description="Authors (if found)")
    journal: Optional[str] = Field(None, description="Journal name")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="DOI")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    url: str = Field(..., description="URL to the article")


class TrialCharacteristics(BaseModel):
    sample_size: Optional[int] = Field(None, description="Total sample size")
    intervention_description: str = Field(..., description="Description of intervention")
    control_description: str = Field(..., description="Description of control/comparator")
    primary_outcome: str = Field(..., description="Primary outcome measure")
    inclusion_criteria: List[str] = Field(..., description="Key inclusion criteria")
    exclusion_criteria: Optional[List[str]] = Field(
        None, description="Key exclusion criteria"
    )
    baseline_characteristics: Optional[str] = Field(
        None, description="Summary of baseline characteristics"
    )
    randomization_method: Optional[str] = Field(
        None, description="Randomization method if described"
    )


class TrialResults(BaseModel):
    primary_outcome_result: str = Field(
        ..., description="Result for primary outcome (effect size, p-value, CI)"
    )
    subgroup_analyses_reported: bool = Field(
        ..., description="Whether subgroup analyses were reported"
    )
    subgroups_analyzed: Optional[List[str]] = Field(
        None, description="Which subgroups were analyzed (if any)"
    )
    heterogeneity_findings: Optional[str] = Field(
        None, description="Any reported treatment effect heterogeneity"
    )
    adverse_events: Optional[str] = Field(
        None, description="Key adverse events or safety findings"
    )


class ArticleExtraction(BaseModel):
    metadata: ArticleMetadata
    trial_characteristics: TrialCharacteristics
    results: TrialResults
    study_limitations: List[str] = Field(
        ..., description="Limitations noted in the article or evident from design"
    )
    relevant_to_explanations: str = Field(
        ..., description="How this trial context relates to the generated explanations"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in the extraction accuracy"
    )
