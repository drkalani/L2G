"""Colab smoke checks for newly added L2G endpoints.

This script validates:
- API health
- project create
- LitSuggest comparison endpoint
- PubMed Entrez fetch endpoint (optional)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


def _base_url(value: str) -> str:
    return (value or "").rstrip("/")


def _ngrok_headers_for(url: str) -> dict[str, str]:
    if "ngrok" in url.lower():
        return {"ngrok-skip-browser-warning": "true"}
    return {}


def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
    headers = {**_ngrok_headers_for(url), **(kwargs.pop("headers", None) or {})}
    r = requests.request(method, url, timeout=60, headers=headers, **kwargs)
    return r


def _fail(message: str, response: requests.Response | None = None) -> int:
    print(f"ERROR: {message}")
    if response is not None:
        try:
            print("body:", response.text)
        except Exception:
            pass
        print("status:", response.status_code)
    return 1


def run_health(base: str) -> int:
    r = _request("GET", f"{base}/health")
    if r.status_code != 200:
        return _fail("Health check failed", r)
    print("[OK] /health:", r.json())
    return 0


def run_compare(base: str, project_id: str) -> int:
    payload = {
        "primary": [{"pmid": "10", "label": 1}, {"pmid": "11", "label": 0}],
        "litsuggest": [{"pmid": "10", "score": 0.87}, {"pmid": "11", "score": 0.15}],
        "score_threshold": 0.5,
    }
    r = _request(
        "POST",
        f"{base}/projects/{project_id}/data/compare/litsuggest",
        json=payload,
    )
    if r.status_code != 200:
        return _fail("LitSuggest comparison failed", r)
    body = r.json()
    print(
        "[OK] /compare/litsuggest:",
        f"intersection={body.get('intersection_count')}",
        f"mismatch={body.get('mismatch_count')}",
        f"accuracy={body.get('metrics', {}).get('accuracy')}",
    )
    return 0


def run_pubmed_fetch(base: str, project_id: str, email: str) -> int:
    payload = {
        "email": email,
        "query": '\"diabetic kidney disease\"[Title/Abstract]',
        "max_results": 10,
        "min_abstract_chars": 0,
        "sleep_between_batches": 0.12,
    }
    r = _request("POST", f"{base}/projects/{project_id}/data/pubmed/fetch", json=payload)
    if r.status_code != 200:
        return _fail("PubMed fetch failed", r)
    body = r.json()
    print(
        "[OK] /pubmed/fetch:",
        f"queried={body.get('queried_id_count')}",
        f"returned={body.get('row_count')}",
    )
    return 0


def create_project(base: str) -> str:
    payload = {
        "name": "Colab Smoke Test Project",
        "disease_key": "colab-smoke",
        "description": "Created by colab_smoke_check.py",
    }
    r = _request("POST", f"{base}/projects", json=payload)
    if r.status_code != 200:
        raise RuntimeError(f"Create project failed: {r.status_code} {r.text}")
    return r.json()["id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Base URL for backend")
    parser.add_argument("--email", default="", help="Required when running pubmed fetch")
    parser.add_argument(
        "--require-pubmed",
        action="store_true",
        help="Fail the script if PubMed fetch endpoint is not successful",
    )
    parser.add_argument(
        "--skip-pubmed",
        action="store_true",
        help="Skip PubMed fetch smoke check",
    )
    args = parser.parse_args()

    base = _base_url(args.base_url)
    if not base:
        print("ERROR: --base-url is required")
        return 2

    project_id = create_project(base)
    print("project_id:", project_id)

    if run_health(base) != 0:
        return 1
    if run_compare(base, project_id) != 0:
        return 1

    if args.skip_pubmed:
        print("[SKIP] PubMed fetch check disabled with --skip-pubmed")
        return 0

    if not args.email:
        if args.require_pubmed:
            return _fail("Email required for PubMed fetch (--email)")
        print("[SKIP] Missing --email, skipping PubMed fetch check")
        return 0

    try:
        code = run_pubmed_fetch(base, project_id, args.email)
    except Exception as exc:
        if args.require_pubmed:
            return _fail(f"PubMed fetch raised: {exc}")
        print("WARN: PubMed check error (ignored):", exc)
        return 0

    if code != 0 and not args.require_pubmed:
        print("WARN: PubMed check failed but continuing.")
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())
