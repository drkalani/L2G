"""Processor selection: CUDA, Apple Metal (MPS), or CPU."""

from __future__ import annotations

import enum
from typing import Optional

import torch


class ProcessorKind(str, enum.Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def resolve_torch_device(kind: str | ProcessorKind) -> torch.device:
    """
    Resolve training/inference device from user preference.

    - cuda: requires CUDA availability
    - mps: Apple Silicon Metal Performance Shaders (PyTorch MPS backend)
    - cpu: always available
    - auto: prefers cuda, then mps, then cpu
    """
    if isinstance(kind, str):
        try:
            kind = ProcessorKind(kind.lower())
        except ValueError:
            kind = ProcessorKind.AUTO

    if kind == ProcessorKind.CPU:
        return torch.device("cpu")

    if kind == ProcessorKind.CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if kind == ProcessorKind.MPS:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS (Metal) requested but torch.backends.mps.is_available() is False. "
                "Use CPU or CUDA, or run on Apple Silicon with PyTorch MPS support."
            )
        return torch.device("mps")

    # AUTO
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_for_pipeline(kind: Optional[str] = None) -> torch.device:
    """Default device for Hugging Face pipelines (-1 = CPU in pipeline API)."""
    if kind is None or kind == ProcessorKind.AUTO or str(kind).lower() == "auto":
        d = resolve_torch_device(ProcessorKind.AUTO)
    else:
        d = resolve_torch_device(kind)
    return d


def pipeline_device_index(device: torch.device) -> int:
    """transformers pipeline uses -1 for CPU, 0 for first GPU/MPS."""
    if device.type == "cpu":
        return -1
    return 0
