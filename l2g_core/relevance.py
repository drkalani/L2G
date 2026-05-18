"""BioBERT-based relevance classification (binary; disease-agnostic labels)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from geneminer_core.devices import ProcessorKind, resolve_torch_device
from geneminer_core.text_cleaning import clean_text


class AbstractDataset(Dataset):
    """Sequence-pair mode uses (title_or_sentence_a, abstract_or_sentence_b) when texts_b is set."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_len: int = 512,
        texts_b: Optional[List[str]] = None,
    ):
        if texts_b is not None:
            if len(texts_b) != len(texts):
                raise ValueError("texts and texts_b must have the same length")
            self.encodings = tokenizer(
                texts_b,
                texts,
                truncation=True,
                padding=True,
                max_length=max_len,
            )
        else:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_len,
            )
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


DEFAULT_MODEL = "dmis-lab/biobert-v1.1"


def train_relevance_classifier(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    output_dir: str | Path,
    device_kind: str | ProcessorKind = ProcessorKind.AUTO,
    base_model: str = DEFAULT_MODEL,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    weight_decay: float = 0.01,
    seed: int = 42,
    max_length: int = 512,
    fp16: Optional[bool] = None,
    train_texts_b: Optional[List[str]] = None,
    val_texts_b: Optional[List[str]] = None,
) -> Tuple[Any, Any, Dict[str, float]]:
    """
    Fine-tune a sequence classifier. Returns (model, tokenizer, metrics_dict).
    fp16: if None, enable only when CUDA is used (MPS/CPU often work better without fp16).
    """
    device = resolve_torch_device(device_kind)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts = [clean_text(t) for t in train_texts]
    val_texts = [clean_text(t) for t in val_texts]
    if train_texts_b is not None and val_texts_b is not None:
        train_texts_b = [clean_text(t or "") for t in train_texts_b]
        val_texts_b = [clean_text(t or "") for t in val_texts_b]
        if len(train_texts_b) != len(train_texts) or len(val_texts_b) != len(val_texts):
            raise ValueError("train/val pair lists must align with primary texts")
        input_mode = "pair"
    elif train_texts_b is not None or val_texts_b is not None:
        raise ValueError("Provide both train_texts_b and val_texts_b for pair mode, or neither")
    else:
        input_mode = "single"
        train_texts_b = None
        val_texts_b = None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    )
    model.to(device)

    train_ds = AbstractDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_len=max_length,
        texts_b=train_texts_b,
    )
    val_ds = AbstractDataset(
        val_texts,
        val_labels,
        tokenizer,
        max_len=max_length,
        texts_b=val_texts_b,
    )

    use_fp16 = fp16 if fp16 is not None else device.type == "cuda"

    common_kw: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "load_best_model_at_end": True,
        "fp16": use_fp16,
        "seed": seed,
        "logging_steps": 50,
        "report_to": "none",
    }
    try:
        args = TrainingArguments(
            **common_kw,
            evaluation_strategy="epoch",
        )
    except TypeError:
        args = TrainingArguments(
            **common_kw,
            eval_strategy="epoch",
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "train_loss": float(train_result.training_loss) if train_result.training_loss else 0.0,
        "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
        "base_model": base_model,
        "device": str(device),
        "input_mode": input_mode,
    }
    with open(output_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return model, tokenizer, eval_metrics


def _read_input_mode(model_dir: Path) -> str:
    p = model_dir / "train_manifest.json"
    if not p.is_file():
        return "single"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return str(data.get("input_mode", "single"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "single"


def relevance_model_input_mode(model_dir: str | Path) -> str:
    """Whether a saved classifier expects single-field text or (title, abstract) pairs."""
    return _read_input_mode(Path(model_dir))


def predict_relevance(
    texts: List[str],
    model_dir: str | Path,
    device_kind: str | ProcessorKind = ProcessorKind.AUTO,
    batch_size: int = 16,
    max_length: int = 512,
    texts_b: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_labels, positive-class probability).

    When the saved manifest has input_mode \"pair\", passes (titles, abstracts) to the tokenizer;
    if texts_b is omitted, empty titles are used.
    """
    device = resolve_torch_device(device_kind)
    model_dir = Path(model_dir)
    input_mode = _read_input_mode(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    texts = [clean_text(t) for t in texts]
    if input_mode == "pair":
        if texts_b is None:
            texts_b = [""] * len(texts)
        else:
            if len(texts_b) != len(texts):
                raise ValueError("texts_b must match texts length for pair-mode models")
        texts_b = [clean_text(t or "") for t in texts_b]
    else:
        texts_b = None
    labels: List[int] = []
    probs: List[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_a = texts[i : i + batch_size]
            if texts_b is not None:
                batch_b = texts_b[i : i + batch_size]
                enc = tokenizer(
                    batch_b,
                    batch_a,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
            else:
                enc = tokenizer(
                    batch_a,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
            logits = model(**enc).logits
            pr = torch.softmax(logits, dim=-1)
            pred = torch.argmax(pr, dim=-1)
            # probability of class 1 (relevant), for binary metrics / ROC
            if pr.shape[-1] >= 2:
                pos_p = pr[:, 1]
            else:
                pos_p = pr[:, 0]
            labels.extend(pred.cpu().numpy().tolist())
            probs.extend(pos_p.cpu().numpy().tolist())

    return np.array(labels, dtype=np.int64), np.array(probs, dtype=np.float64)
