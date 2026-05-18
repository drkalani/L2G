"""Entity normalization: character rules + MyGene (HGNC symbol resolution)."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

GREEK_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "κ": "kappa",
    "μ": "mu",
    "τ": "tau",
    "ω": "omega",
}

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def normalize_characters(symbol: str) -> str:
    s = symbol
    for g, latin in GREEK_MAP.items():
        s = s.replace(g, latin)
    s = re.sub(r"[-_/]", "", s)
    return s.upper()


def _mygene_query_symbol(term: str) -> Optional[str]:
    try:
        import mygene  # type: ignore

        mg = mygene.MyGeneInfo()
        q = mg.query(term, fields="symbol", species=9606)
        if q and q.get("hits"):
            return str(q["hits"][0].get("symbol") or "").strip() or None
    except Exception:
        pass
    return None


def wikipedia_lookup(term: str, timeout: float = 5.0) -> Dict[str, Any]:
    try:
        r = requests.get(WIKI_API.format(term), timeout=timeout)
        if r.status_code != 200:
            return {}
        data = r.json()
        desc = (data.get("description") or "").lower()
        if "gene" in desc:
            return {
                "wiki_title": data.get("title"),
                "wiki_description": data.get("description"),
                "source": "Wikipedia",
            }
    except Exception:
        pass
    return {}


def normalize_mentions_df(
    entity_df: pd.DataFrame,
    use_wikipedia_fallback: bool = True,
    rate_limit_sleep: float = 0.15,
) -> pd.DataFrame:
    """
    Rows need columns: pmid, mention, (optional) start, end.
    Returns dataframe with normalized_symbol, source, flag_manual_review.
    """
    rows: List[Dict[str, Any]] = []

    for _, row in entity_df.iterrows():
        raw = str(row.get("mention", ""))
        mention = normalize_characters(raw)

        sym = _mygene_query_symbol(mention)
        if sym:
            rec = dict(row)
            rec["normalized_symbol"] = sym
            rec["source"] = "MyGene"
            rec["flag_manual_review"] = False
            rows.append(rec)
            time.sleep(rate_limit_sleep)
            continue

        # try raw mention without aggressive char strip (short symbols)
        if raw != mention:
            sym2 = _mygene_query_symbol(raw.strip())
            if sym2:
                rec = dict(row)
                rec["normalized_symbol"] = sym2
                rec["source"] = "MyGene"
                rec["flag_manual_review"] = False
                rows.append(rec)
                time.sleep(rate_limit_sleep)
                continue

        if use_wikipedia_fallback:
            wiki = wikipedia_lookup(mention)
            if wiki:
                rec = dict(row)
                rec["normalized_symbol"] = None
                rec["source"] = "Wikipedia"
                rec["flag_manual_review"] = True
                rec["wiki_title"] = wiki.get("wiki_title")
                rows.append(rec)
                time.sleep(rate_limit_sleep)
                continue

        rec = dict(row)
        rec["normalized_symbol"] = None
        rec["source"] = "unresolved"
        rec["flag_manual_review"] = True
        rows.append(rec)
        time.sleep(rate_limit_sleep)

    return pd.DataFrame(rows)
