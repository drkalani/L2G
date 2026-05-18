"""GeneMiner core: relevance classification, NER, normalization, pipeline."""

__version__ = "1.0.0"


def __getattr__(name: str):
    if name in {"ProcessorKind", "resolve_torch_device"}:
        from geneminer_core.devices import ProcessorKind, resolve_torch_device

        return {"ProcessorKind": ProcessorKind, "resolve_torch_device": resolve_torch_device}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProcessorKind", "resolve_torch_device", "__version__"]
