"""Orchestrate classify → ner → normalize (full or single steps)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from geneminer_core.ner import NerMethod, extract_entities_with_method, load_ner_pipeline
from geneminer_core.normalization import normalize_mentions_df
from geneminer_core.relevance import predict_relevance, relevance_model_input_mode
from geneminer_core.schemas import ArticleRow


class PipelineStep(str, Enum):
    CLASSIFY = "classify"
    NER = "ner"
    NORMALIZE = "normalize"
    FULL = "full"


def run_classify_only(
    articles: List[ArticleRow],
    relevance_model_dir: str | Path,
    processor: str = "auto",
    batch_size: int = 16,
) -> pd.DataFrame:
    model_path = Path(relevance_model_dir)
    texts = [a.text for a in articles]
    pmids = [a.pmid for a in articles]
    mode = relevance_model_input_mode(model_path)
    if mode == "pair":
        titles = [a.title or "" for a in articles]
        preds, probs = predict_relevance(
            texts,
            model_path,
            device_kind=processor,
            batch_size=batch_size,
            texts_b=titles,
        )
    else:
        preds, probs = predict_relevance(
            texts,
            model_path,
            device_kind=processor,
            batch_size=batch_size,
        )
    row: Dict[str, Any] = {
        "pmid": pmids,
        "text": texts,
        "relevant": preds,
        "relevance_prob": probs,
    }
    if mode == "pair" or any(a.title for a in articles):
        row["title"] = [a.title or "" for a in articles]
    return pd.DataFrame(row)


def run_ner_only(
    articles: List[ArticleRow],
    ner_model: str,
    processor: Optional[str],
    ner_method: NerMethod = "transformers",
    bent_service_url: str | None = None,
) -> pd.DataFrame:
    pairs = [(a.pmid, a.text) for a in articles]
    ner = None
    if ner_method == "transformers":
        ner = load_ner_pipeline(model_name=ner_model, processor=processor)
    mentions = extract_entities_with_method(
        pairs,
        ner_pipeline=ner,
        method=ner_method,
        bent_service_url=bent_service_url,
    )
    if not mentions:
        return pd.DataFrame(columns=["pmid", "mention", "start", "end", "score"])
    return pd.DataFrame(
        [
            {
                "pmid": m.pmid,
                "mention": m.mention,
                "start": m.start,
                "end": m.end,
                "score": m.score,
            }
            for m in mentions
        ]
    )


def run_normalize_only(mentions_df: pd.DataFrame, **norm_kw) -> pd.DataFrame:
    if mentions_df.empty:
        return mentions_df
    return normalize_mentions_df(mentions_df, **norm_kw)


def run_full_pipeline(
    articles: List[ArticleRow],
    relevance_model_dir: str | Path,
    ner_model: str = "pruas/BENT-PubMedBERT-NER-Gene",
    processor: str = "auto",
    ner_method: NerMethod = "transformers",
    bent_service_url: str | None = None,
    relevant_label: int = 1,
    batch_size: int = 16,
    use_wikipedia_fallback: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (abstracts_with_scores, mentions_raw, mentions_normalized).
    """
    df_cls = run_classify_only(
        articles,
        relevance_model_dir,
        processor=processor,
        batch_size=batch_size,
    )
    rel = df_cls[df_cls["relevant"] == relevant_label]
    if rel.empty:
        return (
            df_cls,
            pd.DataFrame(columns=["pmid", "mention", "start", "end", "score"]),
            pd.DataFrame(),
        )

    sub = []
    for r in rel.itertuples(index=False):
        raw_title = getattr(r, "title", None)
        title = None
        if raw_title is not None and str(raw_title).strip() != "":
            title = str(raw_title).strip()
        sub.append(
            ArticleRow(pmid=str(r.pmid), text=str(r.text), title=title)
        )
    ner_df = run_ner_only(
        sub,
        ner_model=ner_model,
        processor=processor,
        ner_method=ner_method,
        bent_service_url=bent_service_url,
    )
    if ner_df.empty:
        return df_cls, ner_df, pd.DataFrame()

    norm_df = normalize_mentions_df(ner_df, use_wikipedia_fallback=use_wikipedia_fallback)
    return df_cls, ner_df, norm_df
