"""Compare binary labels (e.g. L2G vs LitSuggest-style scores)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _norm_pmid(value: Any) -> str:
    return str(value).strip()


def score_to_binary_label(score: float, threshold: float) -> int:
    return 1 if float(score) >= threshold else 0


def compare_pmids_labels_vs_scores(
    primary_rows: List[Dict[str, Any]],
    litsuggest_rows: List[Dict[str, Any]],
    *,
    score_threshold: float = 0.5,
    primary_label_keys: Tuple[str, ...] = ("label", "relevant", "y"),
    score_keys: Tuple[str, ...] = ("score", "prediction", "prob", "probability", "p"),
) -> Dict[str, Any]:
    """
    Inner-join on PMID: primary supplies binary label; LitSuggest row supplies score.

    Primary row must include PMID and one of primary_label_keys (0/1).
    LitSuggest row must include PMID and one of score_keys.
    """
    primary_map: Dict[str, int] = {}
    for row in primary_rows:
        pmid = _norm_pmid(row.get("pmid"))
        if not pmid:
            continue
        lbl: int | None = None
        for k in primary_label_keys:
            if k in row and row[k] is not None and str(row[k]).strip() != "":
                try:
                    lbl = int(row[k])
                except (TypeError, ValueError):
                    continue
                break
        if lbl is None or lbl not in (0, 1):
            continue
        primary_map[pmid] = lbl

    score_map: Dict[str, float] = {}
    for row in litsuggest_rows:
        pmid = _norm_pmid(row.get("pmid"))
        if not pmid:
            continue
        sc: float | None = None
        for k in score_keys:
            if k in row and row[k] is not None and str(row[k]).strip() != "":
                try:
                    sc = float(row[k])
                except (TypeError, ValueError):
                    continue
                break
        if sc is None:
            continue
        score_map[pmid] = sc

    keys = sorted(set(primary_map.keys()) & set(score_map.keys()))
    if not keys:
        return {
            "intersection_count": 0,
            "metrics": None,
            "confusion": None,
            "classification_report": None,
            "rows": [],
            "mismatches": [],
            "note": "No overlapping PMIDs with valid primary labels and LitSuggest scores.",
        }

    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    rows_out: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []

    for pmid in keys:
        pl = primary_map[pmid]
        sc = score_map[pmid]
        ls_lbl = score_to_binary_label(sc, score_threshold)
        y_true.append(pl)
        y_pred.append(ls_lbl)
        scores.append(sc)
        agree = pl == ls_lbl
        rec = {
            "pmid": pmid,
            "primary_label": pl,
            "litsuggest_score": sc,
            "litsuggest_label_at_threshold": ls_lbl,
            "agreement": agree,
        }
        rows_out.append(rec)
        if not agree:
            mismatches.append(rec)

    yt = np.array(y_true)
    yp = np.array(y_pred)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    rep = classification_report(yt, yp, labels=[0, 1], output_dict=True, zero_division=0)

    return {
        "intersection_count": len(keys),
        "score_threshold": score_threshold,
        "metrics": {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        },
        "confusion": {
            "labels_order": [0, 1],
            "matrix": cm.tolist(),
        },
        "classification_report": rep,
        "mismatch_count": len(mismatches),
        "agreement_count": len(keys) - len(mismatches),
        "rows": rows_out,
        "mismatches": mismatches,
    }
