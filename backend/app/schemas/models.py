"""Pydantic API models."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ProcessorType = Literal["auto", "cuda", "mps", "cpu"]
PipelineMode = Literal["classify", "ner", "normalize", "full"]
NerMethod = Literal["transformers", "bent"]


class ArticleInput(BaseModel):
    pmid: str
    text: str
    title: Optional[str] = Field(
        None,
        description="Optional title; when provided (DKDM-style), title+abstract pair-encoding is used.",
    )
    label: Optional[int] = Field(None, description="0 irrelevant, 1 relevant (for training/eval)")


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    disease_key: str = Field(
        "custom",
        description="Identifier for disease/domain (e.g. dkd, alzheimer). Used for organization only.",
    )
    description: str = ""


class ProjectOut(BaseModel):
    id: str
    name: str
    disease_key: str
    description: str


class TrainingConfig(BaseModel):
    processor: ProcessorType = "auto"
    base_model: str = "dmis-lab/biobert-v1.1"
    learning_rate: float = 2e-5
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 42
    max_length: int = 512
    fp16: Optional[bool] = None


class KFoldTrainingConfig(TrainingConfig):
    n_splits: int = Field(5, ge=2, le=10)


class TrainJobCreate(BaseModel):
    articles: List[ArticleInput]
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    validation_split: float = Field(0.2, ge=0.1, le=0.4)
    dedupe_by_pmid: bool = Field(True, description="Keep first row per PMID (DKDM-style deduplication)")


class KFoldTrainJobCreate(BaseModel):
    articles: List[ArticleInput]
    config: KFoldTrainingConfig = Field(default_factory=KFoldTrainingConfig)
    dedupe_by_pmid: bool = Field(True, description="Keep first row per PMID (DKDM-style deduplication)")


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None


class JobSummary(JobStatus):
    project_id: Optional[str] = None


class PipelineRunRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., description="Trained relevance model folder name under project")
    articles: List[ArticleInput]
    mode: PipelineMode = "full"
    processor: ProcessorType = "auto"
    ner_model: str = "pruas/BENT-PubMedBERT-NER-Gene"
    ner_method: NerMethod = "transformers"
    bent_service_url: Optional[str] = None
    batch_size: int = 16
    use_wikipedia_fallback: bool = True
    # For normalize-only: pass prior mentions as rows
    mentions_json: Optional[List[Dict[str, Any]]] = None


class DeviceInfo(BaseModel):
    available: Dict[str, bool]
    recommended: str


class BaseModelDownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1, description="Hugging Face model id")


ModelTaskKind = Literal["classification", "token_classification"]


class ModelCompatibilityResult(BaseModel):
    model_id: str
    expected_task: ModelTaskKind
    compatible: bool
    detected_tasks: List[str]
    message: str


class ProjectModelInfo(BaseModel):
    model_id: str
    path: str


class ProjectModelCatalog(BaseModel):
    models: List[ProjectModelInfo]


class PubMedFetchRequest(BaseModel):
    """NCBI E-utilities require a valid email (https://www.ncbi.nlm.nih.gov/books/NBK25497/)."""

    email: str = Field(..., min_length=4, description="Contact email for NCBI Entrez")
    query: Optional[str] = Field(
        None,
        description="PubMed query (esearch). Ignored if `pmids` is non-empty.",
    )
    pmids: Optional[List[str]] = Field(
        None,
        description="Explicit PMID list to fetch via efetch (skips esearch when non-empty).",
    )
    max_results: int = Field(200, ge=1, le=10_000, description="Max IDs from search or cap on explicit pmids")
    retstart: int = Field(0, ge=0, description="esearch offset when using query")
    min_abstract_chars: int = Field(
        0,
        ge=0,
        le=50_000,
        description="If > 0, drops rows with shorter abstracts (DKDM-style filter).",
    )
    sleep_between_batches: float = Field(
        0.12,
        ge=0.05,
        le=3.0,
        description="Pause between batched efetch calls (seconds).",
    )


class LitSuggestCompareRequest(BaseModel):
    """Rows must share PMID keys. Primary: binary 0/1 labels; LitSuggest: continuous scores."""

    primary: List[Dict[str, Any]] = Field(
        ...,
        description="Rows with pmid and label (or relevant/y) from L2G or your model.",
    )
    litsuggest: List[Dict[str, Any]] = Field(
        ...,
        description="Rows with pmid and score (or prediction/prob) from LitSuggest export.",
    )
    score_threshold: float = Field(
        0.5,
        description="Scores >= threshold are treated as label 1 (DKDM-style 0.5 cutoff).",
    )
