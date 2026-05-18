"""Run pipeline steps (sync) for API."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from app.config import get_settings
from app.schemas.models import ArticleInput, PipelineRunRequest
from app.services.project_service import project_dir

from l2g_core.pipeline import (
    run_classify_only,
    run_full_pipeline,
    run_normalize_only,
    run_ner_only,
)
from l2g_core.schemas import ArticleRow


def articles_to_rows(articles: List[ArticleInput]) -> List[ArticleRow]:
    return [
        ArticleRow(
            pmid=str(a.pmid),
            text=str(a.text),
            label=a.label,
            title=(a.title.strip() if a.title and a.title.strip() else None),
        )
        for a in articles
    ]


def execute_pipeline(req: PipelineRunRequest) -> Dict[str, Any]:
    base = project_dir(req.project_id)
    model_path = base / "models" / req.model_id
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rows = articles_to_rows(req.articles)
    mode = req.mode
    out_dir = base / "outputs" / "last_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "classify":
        df = run_classify_only(
            rows,
            model_path,
            processor=req.processor,
            batch_size=req.batch_size,
        )
        df.to_csv(out_dir / "classification.csv", index=False)
        payload = _df_payload(df, "classification")
        payload["saved_to"] = str(out_dir.relative_to(get_settings().data_dir))
        return payload

    if mode == "ner":
        df = run_ner_only(
            rows,
            ner_model=req.ner_model,
            processor=req.processor,
            ner_method=req.ner_method,
            bent_service_url=req.bent_service_url or get_settings().bent_service_url,
        )
        df.to_csv(out_dir / "mentions.csv", index=False)
        payload = _df_payload(df, "ner")
        payload["saved_to"] = str(out_dir.relative_to(get_settings().data_dir))
        return payload

    if mode == "normalize":
        if not req.mentions_json:
            raise ValueError("normalize mode requires mentions_json")
        mdf = pd.DataFrame(req.mentions_json)
        df = run_normalize_only(mdf)
        df.to_csv(out_dir / "normalized.csv", index=False)
        payload = _df_payload(df, "normalized")
        payload["saved_to"] = str(out_dir.relative_to(get_settings().data_dir))
        return payload

    if mode == "full":
        cls_df, ner_df, norm_df = run_full_pipeline(
            rows,
            model_path,
            ner_model=req.ner_model,
            processor=req.processor,
            ner_method=req.ner_method,
            bent_service_url=req.bent_service_url or get_settings().bent_service_url,
            batch_size=req.batch_size,
            use_wikipedia_fallback=req.use_wikipedia_fallback,
        )
        cls_df.to_csv(out_dir / "classification.csv", index=False)
        ner_df.to_csv(out_dir / "mentions.csv", index=False)
        norm_df.to_csv(out_dir / "normalized.csv", index=False)
        return {
            "kind": "full",
            "classification": cls_df.to_dict(orient="records"),
            "mentions": ner_df.to_dict(orient="records"),
            "normalized": norm_df.to_dict(orient="records"),
            "saved_to": str(out_dir.relative_to(get_settings().data_dir)),
        }

    raise ValueError(f"Unknown mode {mode}")


def _df_payload(df: pd.DataFrame, kind: str) -> Dict[str, Any]:
    charts: Dict[str, Any] = {}
    if kind == "classification" and "relevant" in df.columns:
        vc = df["relevant"].value_counts().to_dict()
        charts["label_counts"] = {str(k): int(v) for k, v in vc.items()}
    if kind == "normalized" and "source" in df.columns:
        vc = df["source"].value_counts().to_dict()
        charts["source_counts"] = {str(k): int(v) for k, v in vc.items()}
    return {
        "kind": kind,
        "rows": df.to_dict(orient="records"),
        "row_count": len(df),
        "charts": charts,
    }
