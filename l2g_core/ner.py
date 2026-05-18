"""Gene/protein NER helpers.

Current supported backends:
- transformers token-classification pipeline (default, production path)
- optional legacy bent annotator parsing
"""

from __future__ import annotations

import json
import re
import tempfile
import os
from pathlib import Path
from importlib import import_module
from typing import Any, Callable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from typing import Literal

from geneminer_core.schemas import MentionRecord

DEFAULT_NER_MODEL = "pruas/BENT-PubMedBERT-NER-Gene"
NerMethod = Literal["transformers", "bent"]
BENT_SERVICE_DEFAULT_TIMEOUT = 30


def load_ner_pipeline(
    model_name: str = DEFAULT_NER_MODEL,
    processor: Optional[str] = None,
):
    from geneminer_core.devices import device_for_pipeline, pipeline_device_index

    transformers_pipeline = import_module("transformers").pipeline
    device = device_for_pipeline(processor)
    dev_idx = pipeline_device_index(device)
    return transformers_pipeline(
        "token-classification",
        model=model_name,
        tokenizer=model_name,
        aggregation_strategy="simple",
        device=dev_idx,
    )


def extract_entities(
    pairs: List[Tuple[str, str]],
    ner_pipeline,
) -> List[MentionRecord]:
    """pairs: list of (pmid, text)."""
    records: List[MentionRecord] = []
    for pmid, text in pairs:
        if not text or not str(text).strip():
            continue
        try:
            entities = ner_pipeline(text)
        except Exception:
            continue
        for ent in entities:
            score = ent.get("score")
            records.append(
                MentionRecord(
                    pmid=str(pmid),
                    mention=str(ent.get("word", "")).strip(),
                    start=int(ent.get("start", 0)),
                    end=int(ent.get("end", 0)),
                    score=float(score) if score is not None else None,
                )
            )
    return records


def _parse_bent_ann_lines(lines: List[str], pmid: str) -> List[MentionRecord]:
    records: List[MentionRecord] = []
    for line in lines:
        if not line.startswith("T"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue

        span = parts[1]
        nums = [int(n) for n in re.findall(r"\d+", span)]
        start = nums[0] if nums else 0
        end = nums[1] if len(nums) > 1 else 0
        mention = parts[2].strip()
        records.append(
            MentionRecord(
                pmid=str(pmid),
                mention=mention,
                start=start,
                end=end,
                score=None,
            )
        )
    return records


def _collect_ann_files(out_dir: Path, expected_count: int) -> List[Path]:
    def _sort_key(path: Path) -> tuple[object, ...]:
        parts = re.findall(r"\d+|[A-Za-z]+", path.stem)
        keys: List[object] = []
        for part in parts:
            if part.isdigit():
                keys.append(int(part))
            else:
                keys.append(part.lower())
        if not keys:
            return (path.name,)
        return tuple(keys)

    files = sorted(out_dir.rglob("*.ann"), key=_sort_key)
    if not files:
        return []
    if len(files) >= expected_count:
        return files[:expected_count]
    return files


def _invoke_bent_annotate(annotate_fn: Callable[..., Any], texts: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_arg = f"{str(out_dir).rstrip('/')}/"
    attempts: List[dict[str, Any]] = [
        {
            "input_text": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": out_dir_arg,
        },
        {
            "input_text": texts,
            "types": {"gene": ""},
            "task": "gene",
            "recognize": True,
            "out_dir": out_dir_arg,
        },
        {
            "texts": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": out_dir_arg,
        },
        {
            "text": texts,
            "types": {"gene": ""},
            "recognize": True,
            "out_dir": out_dir_arg,
        },
        {
            "input_text": texts,
            "types": ["gene"],
            "recognize": True,
            "out_dir": out_dir_arg,
        },
    ]

    last_error: Exception | None = None
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            annotate_fn(**kwargs)
            return
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            raise RuntimeError(f"Bent annotation failed: {exc}") from exc

    try:
        annotate_fn(texts, out_dir=out_dir_arg, recognize=True)
        return
    except Exception as exc:
        raise RuntimeError(
            "Bent annotation call signature did not match known variants."
        ) from last_error or exc


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def extract_entities_with_bent_partitioned(
    pairs: List[Tuple[str, str]]
) -> List[tuple[str, List[MentionRecord]]]:
    non_empty = [(pmid, text) for pmid, text in pairs if text and str(text).strip()]
    if not non_empty:
        return []

    try:
        bt = import_module("bent.annotate")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bent method requires `bent` package. Install the latest supported version explicitly: "
            "`pip install bent==0.0.80` (Python 3.10.x, <=3.10.13). "
            "You can use `scripts/setup_bent_runtime.sh` for guided local setup."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to import bent annotator: {exc}") from exc
    if not hasattr(bt, "annotate"):
        raise RuntimeError("Bent package does not expose annotate() entry point.")

    texts = [text for _, text in non_empty]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "output" / "ner"

        try:
            _invoke_bent_annotate(bt.annotate, texts, out_dir)
        except Exception as exc:
            raise RuntimeError(f"Bent annotation failed: {exc}") from exc

        ann_files = _collect_ann_files(out_dir, len(non_empty))
        if not ann_files:
            raise RuntimeError(f"Bent annotation produced no .ann files in {out_dir}.")
        chunks: List[tuple[str, List[MentionRecord]]] = []
        for ann_file, (pmid, _) in zip(ann_files, non_empty):
            lines = ann_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            chunks.append((str(pmid), _parse_bent_ann_lines(lines, pmid)))

    return chunks


def extract_entities_with_bent_local(pairs: List[Tuple[str, str]]) -> List[MentionRecord]:
    records: List[MentionRecord] = []
    for _, mentions in extract_entities_with_bent_partitioned(pairs):
        records.extend(mentions)
    return records


def _extract_entities_with_bent_service(
    pairs: List[Tuple[str, str]], service_url: str
) -> List[MentionRecord]:
    timeout_raw = os.getenv("BENT_SERVICE_TIMEOUT_SECONDS", str(BENT_SERVICE_DEFAULT_TIMEOUT))
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = float(BENT_SERVICE_DEFAULT_TIMEOUT)
    request_url = service_url.rstrip("/")
    if not request_url.endswith("/annotate"):
        request_url = request_url + "/annotate"
    payload_pairs = [
        {"pmid": str(pmid), "text": str(text), "text_index": idx}
        for idx, (pmid, text) in enumerate(pairs)
        if text and str(text).strip()
    ]
    payload = {
        "pairs": [
            {"pmid": pair["pmid"], "text": pair["text"], "text_index": pair["text_index"]}
            for pair in payload_pairs
        ]
    }

    request = Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if response.status >= 400:
                raise RuntimeError(f"Bent service returned status {response.status}: {body}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Bent service request failed: HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bent service request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Bent service request failed: {exc}") from exc

    try:
        result = json.loads(body or "{}")
    except ValueError as exc:
        raise RuntimeError(f"Bent service response was not valid JSON: {body}") from exc

    results = result.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Bent service response missing `results` list.")

    index_map: dict[int, List[MentionRecord]] = {}
    fallback_results: List[List[MentionRecord]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        mentions = item.get("mentions")
        if not isinstance(mentions, list):
            continue
        mentions_list: List[MentionRecord] = []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            mention_text = str(mention.get("mention", "")).strip()
            start = _safe_int(mention.get("start"), 0)
            end = _safe_int(mention.get("end"), 0)
            score_raw = mention.get("score")
            score = float(score_raw) if score_raw is not None else None
            mention_pmid = str(item.get("pmid", ""))
            mentions_list.append(
                MentionRecord(
                    pmid=mention_pmid,
                    mention=mention_text,
                    start=start,
                    end=end,
                    score=score,
                )
            )
        if "text_index" in item:
            index = _safe_int(item.get("text_index"), len(index_map))
            index_map[index] = mentions_list
        else:
            fallback_results.append(mentions_list)

    ordered_results: List[List[MentionRecord]] = []
    for idx, pair in enumerate(payload_pairs):
        pair_index = _safe_int(pair["text_index"], idx)
        mentions = index_map.get(pair_index)
        if mentions is None and fallback_results:
            mentions = fallback_results.pop(0)
        else:
            mentions = mentions or []
        if mentions:
            has_pmid = any(m.pmid for m in mentions)
            if not has_pmid:
                for m in mentions:
                    m.pmid = pair["pmid"]
        ordered_results.append(mentions)

    rows: List[MentionRecord] = []
    for mentions in ordered_results:
        rows.extend(mentions)
    return rows


def extract_entities_with_bent(
    pairs: List[Tuple[str, str]],
    bent_service_url: Optional[str] = None,
) -> List[MentionRecord]:
    non_empty = [(pmid, text) for pmid, text in pairs if text and str(text).strip()]
    if not non_empty:
        return []

    service_url = bent_service_url or os.getenv("BENT_SERVICE_URL", "").strip()
    if service_url:
        return _extract_entities_with_bent_service(non_empty, service_url)
    return extract_entities_with_bent_local(non_empty)


def extract_entities_with_method(
    pairs: List[Tuple[str, str]],
    ner_pipeline=None,
    method: NerMethod = "transformers",
    bent_service_url: Optional[str] = None,
) -> List[MentionRecord]:
    if method == "bent":
        return extract_entities_with_bent(pairs, bent_service_url=bent_service_url)
    return extract_entities(pairs, ner_pipeline=ner_pipeline)
