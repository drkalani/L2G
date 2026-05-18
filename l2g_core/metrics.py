"""Lightweight metrics for API reporting and charts (JSON-serializable)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: List[int], y_pred: List[int], y_prob: np.ndarray | None = None
) -> Dict[str, float]:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(yt)) > 1:
        try:
            # probability of positive class
            if y_prob.ndim == 2 and y_prob.shape[1] > 1:
                p = y_prob[:, 1]
            else:
                p = y_prob.flatten()
            out["roc_auc"] = float(roc_auc_score(yt, p))
        except Exception:
            out["roc_auc"] = 0.0
    return out


def confusion_binary(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def histogram_counts(values: List[float], bins: int = 10) -> Tuple[List[float], List[int]]:
    if not values:
        return [], []
    arr = np.array(values, dtype=float)
    hist, edges = np.histogram(arr, bins=bins)
    return edges.tolist(), hist.astype(int).tolist()
