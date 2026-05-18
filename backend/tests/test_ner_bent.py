"""Tests for Bent-backed NER adapter."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Callable, Iterator, List, Tuple
import unittest


@contextlib.contextmanager
def fake_bent_module(annotate_fn: Callable[..., None]) -> Iterator[None]:
    original_pkg = sys.modules.get("bent")
    original_sub = sys.modules.get("bent.annotate")

    fake_pkg = types.ModuleType("bent")
    fake_annotate = types.ModuleType("bent.annotate")
    fake_annotate.annotate = annotate_fn
    fake_pkg.annotate = fake_annotate

    sys.modules["bent"] = fake_pkg
    sys.modules["bent.annotate"] = fake_annotate
    try:
        yield
    finally:
        if original_pkg is None:
            sys.modules.pop("bent", None)
        else:
            sys.modules["bent"] = original_pkg
        if original_sub is None:
            sys.modules.pop("bent.annotate", None)
        else:
            sys.modules["bent.annotate"] = original_sub


class TestNerBentMethod(unittest.TestCase):
    def test_extract_entities_with_bent_uses_ann_outputs(self) -> None:
        from l2g_core.ner import extract_entities_with_bent

        calls: List[tuple[tuple, dict]] = []

        def annotate(**kwargs) -> None:
            calls.append(((), kwargs))
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # emulate bent output per input row
            texts = kwargs.get("input_text") or kwargs.get("texts") or kwargs.get("text")
            if isinstance(texts, list):
                count = len(texts)
            elif isinstance(texts, tuple):
                count = len(texts)
            else:
                count = 0
            for idx, _ in enumerate(range(count), start=1):
                content = f"T{idx}\tGene 0 5\tGENE{idx}\n"
                (out_dir / f"doc_{idx}.ann").write_text(content, encoding="utf-8")

        with fake_bent_module(annotate):
            rows = extract_entities_with_bent([("pmid-1", "abc"), ("pmid-2", "def")])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].pmid, "pmid-1")
        self.assertEqual(rows[1].mention, "GENE2")
        self.assertEqual(len(calls), 1)

    def test_fallback_signature_tries_variants_until_success(self) -> None:
        from l2g_core.ner import _invoke_bent_annotate
        from tempfile import TemporaryDirectory

        calls: List[dict] = []

        def annotate(**kwargs) -> None:
            calls.append(kwargs)
            if "input_text" in kwargs:
                raise TypeError("unexpected input_text for this version")
            if "texts" in kwargs and "task" in kwargs:
                raise TypeError("unexpected task variant")
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # minimal success on text keyword
            with (out_dir / "doc_1.ann").open("w", encoding="utf-8") as handle:
                handle.write("T1\tGene 0 3\tTP53\n")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ner"
            with fake_bent_module(annotate):
                _invoke_bent_annotate(annotate, ["row one"], out_dir)

        # First two attempts should fail and the third should be text/texts branch
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(any("texts" in call or "text" in call for call in calls))


if __name__ == "__main__":
    unittest.main()

