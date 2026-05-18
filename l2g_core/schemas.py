"""Shared data shapes (plain dicts / dataclasses for pipeline I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArticleRow:
    pmid: str
    text: str
    label: Optional[int] = None  # 0/1 relevance, optional for inference-only
    title: Optional[str] = None  # optional; pair-encoded with text (abstract) when training/inference use pair mode


@dataclass
class MentionRecord:
    pmid: str
    mention: str
    start: int
    end: int
    score: Optional[float] = None


@dataclass
class NormalizedRecord:
    pmid: str
    mention: str
    normalized_symbol: Optional[str]
    source: str
    flag_manual_review: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def rows_from_records(records: List[Dict[str, Any]]) -> List[ArticleRow]:
    out: List[ArticleRow] = []
    for r in records:
        lbl = r.get("label")
        if lbl is not None and lbl != "":
            lbl = int(lbl)
        else:
            lbl = None
        raw_title = r.get("title")
        title = None
        if raw_title is not None and str(raw_title).strip() != "":
            title = str(raw_title).strip()
        out.append(
            ArticleRow(
                pmid=str(r["pmid"]),
                text=str(r["text"]),
                label=lbl,
                title=title,
            )
        )
    return out
