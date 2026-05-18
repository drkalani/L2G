"""Lightweight Bent microservice."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple

from l2g_core.ner import extract_entities_with_bent_partitioned


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_pairs(payload: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("Payload must include `pairs` as a list.")

    out: List[Tuple[int, str, str]] = []
    for idx, item in enumerate(raw_pairs):
        if isinstance(item, dict):
            text = item.get("text")
            pmid = item.get("pmid", "")
            pair_index = item.get("text_index", idx)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            pmid, text = item[0], item[1]
            pair_index = idx
        else:
            continue

        if text is None or not str(text).strip():
            continue

        out.append((_safe_int(pair_index, idx), str(pmid), str(text)))
    return out


def _run_bent_for_pairs(pairs: List[Tuple[int, str, str]]) -> List[Dict[str, Any]]:
    if not pairs:
        return []

    chunks = extract_entities_with_bent_partitioned([(pmid, text) for _, pmid, text in pairs])
    results: List[Dict[str, Any]] = []

    # zip keeps the response aligned to the original non-empty pair order.
    for (pair_index, pmid, _), (_, mentions) in zip(pairs, chunks):
        results.append(
            {
                "text_index": pair_index,
                "pmid": pmid,
                "mentions": [
                    {
                        "mention": mention.mention,
                        "start": mention.start,
                        "end": mention.end,
                        "score": mention.score,
                    }
                    for mention in mentions
                ],
            }
        )
    return results


def _error_response(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class BentRequestHandler(BaseHTTPRequestHandler):
    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/annotate":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            _error_response(self, 400, "Request body is required.")
            return

        body = self.rfile.read(content_length).decode("utf-8", errors="ignore")
        try:
            payload = json.loads(body or "{}")
        except ValueError as exc:
            _error_response(self, 400, f"Invalid JSON body: {exc}")
            return

        try:
            pairs = _parse_pairs(payload)
            results = _run_bent_for_pairs(pairs)
        except ValueError as exc:
            _error_response(self, 400, str(exc))
            return
        except Exception as exc:
            _error_response(self, 500, f"Bent execution failed: {exc}")
            return

        self._json_response(200, {"results": results})

    def log_message(self, format: str, *args: Any) -> None:
        # Keep logs compact in production containers.
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bent microservice")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), BentRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
