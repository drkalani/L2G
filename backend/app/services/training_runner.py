"""Background training jobs."""

from __future__ import annotations

import json
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.config import get_settings
from app.schemas.models import ArticleInput, KFoldTrainJobCreate, TrainJobCreate, TrainingConfig
from app.services import job_store
from app.services.project_service import project_dir

from l2g_core.metrics import binary_classification_metrics, confusion_binary
from l2g_core.relevance import predict_relevance, train_relevance_classifier


def _json_safe(obj: Any) -> Any:
    """Convert numpy / sklearn outputs for JSON job results."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _dedupe_articles(articles: List[ArticleInput]) -> List[ArticleInput]:
    seen: set[str] = set()
    out: List[ArticleInput] = []
    for a in articles:
        k = str(a.pmid).strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def _articles_to_training(
    articles: List[ArticleInput],
) -> Tuple[List[str], List[int], List[str], Optional[List[str]]]:
    texts: List[str] = []
    labels: List[int] = []
    pmids: List[str] = []
    raw_titles: List[str] = []
    for a in articles:
        if a.label is None:
            raise ValueError("Each article must have label (0 or 1) for training")
        texts.append(a.text)
        labels.append(int(a.label))
        pmids.append(str(a.pmid))
        raw_titles.append((a.title or "").strip())
    pair = any(raw_titles)
    titles_b: Optional[List[str]] = raw_titles if pair else None
    return texts, labels, pmids, titles_b


def run_single_train_job(
    job_id: str,
    project_id: str,
    body: TrainJobCreate,
) -> None:
    def _run() -> None:
        try:
            job_store.set_job(job_id, "running", "Preparing data…", progress=0.05)
            arts = _dedupe_articles(body.articles) if body.dedupe_by_pmid else list(body.articles)
            texts, labels, _pmids, titles_b = _articles_to_training(arts)
            cfg: TrainingConfig = body.config
            job_store.set_job(job_id, "running", "Splitting train/validation data…", progress=0.2)

            if len(set(labels)) < 2:
                raise ValueError("Need at least one sample per class (0 and 1) to train")

            if titles_b is not None:
                train_t, val_t, train_y, val_y, train_tb, val_tb = train_test_split(
                    texts,
                    labels,
                    titles_b,
                    test_size=body.validation_split,
                    random_state=cfg.seed,
                    stratify=labels,
                )
            else:
                train_t, val_t, train_y, val_y = train_test_split(
                    texts,
                    labels,
                    test_size=body.validation_split,
                    random_state=cfg.seed,
                    stratify=labels,
                )
                train_tb = val_tb = None
            job_store.set_job(job_id, "running", "Fine-tuning BioBERT…", progress=0.3)

            out_name = f"relevance_{job_id[:8]}"
            out_dir = project_dir(project_id) / "models" / out_name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            job_store.set_job(job_id, "running", "Fine-tuning BioBERT…", progress=0.35)
            kw_train: Dict[str, Any] = dict(
                train_texts=train_t,
                train_labels=train_y,
                val_texts=val_t,
                val_labels=val_y,
                output_dir=out_dir,
                device_kind=cfg.processor,
                base_model=cfg.base_model,
                learning_rate=cfg.learning_rate,
                num_train_epochs=cfg.num_train_epochs,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                weight_decay=cfg.weight_decay,
                seed=cfg.seed,
                max_length=cfg.max_length,
                fp16=cfg.fp16,
            )
            if train_tb is not None and val_tb is not None:
                kw_train["train_texts_b"] = train_tb
                kw_train["val_texts_b"] = val_tb
            _, _, eval_metrics = train_relevance_classifier(**kw_train)
            job_store.set_job(job_id, "running", "Evaluating validation set…", progress=0.95)

            preds, probs = predict_relevance(
                val_t,
                out_dir,
                device_kind=cfg.processor,
                batch_size=cfg.per_device_eval_batch_size,
                max_length=cfg.max_length,
                texts_b=val_tb,
            )
            pr = (
                np.stack([1 - probs, probs], axis=1)
                if probs is not None and len(probs) == len(val_y)
                else None
            )
            metrics = binary_classification_metrics(val_y, preds.tolist(), pr)

            cm = confusion_binary(val_y, preds.tolist())
            report_dict = classification_report(val_y, preds.tolist(), output_dict=True, zero_division=0)

            result = {
                "model_id": out_name,
                "path": str(out_dir.relative_to(get_settings().data_dir)),
                "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                "metrics": metrics,
                "confusion": cm,
                "classification_report": _json_safe(report_dict),
                "eval_sklearn": {
                    "accuracy": float(accuracy_score(val_y, preds)),
                    "precision": float(precision_score(val_y, preds, zero_division=0)),
                    "recall": float(recall_score(val_y, preds, zero_division=0)),
                    "f1": float(f1_score(val_y, preds, zero_division=0)),
                },
            }
            (out_dir / "validation_report.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            job_store.set_job(job_id, "completed", "Done", result=result, progress=1.0)
        except Exception as exc:  # noqa: BLE001
            job_store.set_job(job_id, "failed", message=str(exc), result=None)

    threading.Thread(target=_run, daemon=True).start()


def run_kfold_train_job(
    job_id: str,
    project_id: str,
    body: KFoldTrainJobCreate,
) -> None:
    def _run() -> None:
        try:
            job_store.set_job(job_id, "running", "K-fold cross-validation…", progress=0.05)
            arts = _dedupe_articles(body.articles) if body.dedupe_by_pmid else list(body.articles)
            texts, labels, _, titles_b = _articles_to_training(arts)
            cfg = body.config

            if len(set(labels)) < 2:
                raise ValueError("Need both classes in the dataset for stratified k-fold")

            skf = StratifiedKFold(
                n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
            )
            fold_metrics: List[Dict[str, Any]] = []
            out_root = project_dir(project_id) / "models" / f"kfold_{job_id[:8]}"
            out_root.mkdir(parents=True, exist_ok=True)
            job_store.set_job(
                job_id,
                "running",
                f"Running {cfg.n_splits}-fold cross-validation…",
                progress=0.1,
            )

            for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(texts, labels)
            ):
                job_store.update_job(
                    job_id,
                    message=f"Training fold {fold_idx + 1}/{cfg.n_splits}…",
                    progress=min(
                        0.95,
                        0.1 + 0.85 * ((fold_idx + 1) / max(1, cfg.n_splits)),
                    ),
                )
                train_t = [texts[i] for i in train_idx]
                train_y = [labels[i] for i in train_idx]
                val_t = [texts[i] for i in val_idx]
                val_y = [labels[i] for i in val_idx]
                train_tb: Optional[List[str]] = (
                    [titles_b[i] for i in train_idx] if titles_b is not None else None
                )
                val_tb: Optional[List[str]] = (
                    [titles_b[i] for i in val_idx] if titles_b is not None else None
                )

                fold_dir = out_root / f"fold_{fold_idx}"
                if fold_dir.exists():
                    shutil.rmtree(fold_dir)
                fold_dir.mkdir(parents=True)

                kw_fold: Dict[str, Any] = dict(
                    train_texts=train_t,
                    train_labels=train_y,
                    val_texts=val_t,
                    val_labels=val_y,
                    output_dir=fold_dir,
                    device_kind=cfg.processor,
                    base_model=cfg.base_model,
                    learning_rate=cfg.learning_rate,
                    num_train_epochs=cfg.num_train_epochs,
                    per_device_train_batch_size=cfg.per_device_train_batch_size,
                    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                    weight_decay=cfg.weight_decay,
                    seed=cfg.seed + fold_idx,
                    max_length=cfg.max_length,
                    fp16=cfg.fp16,
                )
                if train_tb is not None and val_tb is not None:
                    kw_fold["train_texts_b"] = train_tb
                    kw_fold["val_texts_b"] = val_tb
                _, _, eval_metrics = train_relevance_classifier(**kw_fold)
                preds, probs = predict_relevance(
                    val_t,
                    fold_dir,
                    device_kind=cfg.processor,
                    batch_size=cfg.per_device_eval_batch_size,
                    max_length=cfg.max_length,
                    texts_b=val_tb,
                )
                pr = (
                    np.stack([1 - probs, probs], axis=1)
                    if probs is not None and len(probs) == len(val_y)
                    else None
                )
                m = binary_classification_metrics(val_y, preds.tolist(), pr)
                report_dict = classification_report(
                    val_y, preds.tolist(), output_dict=True, zero_division=0
                )
                fold_block = {
                    "fold": fold_idx,
                    "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                    "metrics": m,
                    "confusion": confusion_binary(val_y, preds.tolist()),
                    "classification_report": _json_safe(report_dict),
                    "eval_sklearn": {
                        "accuracy": float(accuracy_score(val_y, preds)),
                        "precision": float(precision_score(val_y, preds, zero_division=0)),
                        "recall": float(recall_score(val_y, preds, zero_division=0)),
                        "f1": float(f1_score(val_y, preds, zero_division=0)),
                    },
                }
                fold_metrics.append(fold_block)
                (fold_dir / "fold_report.json").write_text(
                    json.dumps(fold_block, indent=2), encoding="utf-8"
                )

            # aggregate
            f1s = [float(f["metrics"].get("f1", 0.0)) for f in fold_metrics]
            accs = [float(f["metrics"].get("accuracy", 0.0)) for f in fold_metrics]
            rocs = [
                float(f["metrics"].get("roc_auc", 0.0))
                for f in fold_metrics
                if "roc_auc" in f.get("metrics", {})
            ]
            summary = {
                "model_bundle": out_root.name,
                "path": str(out_root.relative_to(get_settings().data_dir)),
                "n_splits": cfg.n_splits,
                "folds": fold_metrics,
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
            }
            if rocs:
                summary["mean_roc_auc"] = float(np.mean(rocs))
                summary["std_roc_auc"] = float(np.std(rocs))
            (out_root / "kfold_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            job_store.set_job(
                job_id,
                "completed",
                "K-fold complete",
                result=summary,
                progress=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            job_store.set_job(job_id, "failed", message=str(exc), result=None)

    threading.Thread(target=_run, daemon=True).start()
