#!/usr/bin/env python3
"""Minimal HTTP API used by the RagWall open-source edition."""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sanitizer.rag_sanitizer import QuerySanitizer, SanitizerConfig  # noqa: E402


class RagWallService:
    """Rule-only sanitizer service.

    The community build exposes only regex-based sanitisation so it can run
    anywhere without heavyweight model dependencies.
    """

    def __init__(self) -> None:
        self.sanitizer = QuerySanitizer(SanitizerConfig())
        self.risk_lexicon: List[re.Pattern[str]] = [
            re.compile(r"(?i)ignore\s+(previous|prior|all)\s+(instructions?|rules?|policies)"),
            re.compile(r"(?i)no\s+rules"),
            re.compile(r"(?i)without\s+restrictions"),
            re.compile(r"(?i)developer\s+mode"),
            re.compile(r"(?i)you\s+are\s+dan"),
            re.compile(r"(?i)act\s+as\s+(root|admin|sudo)"),
        ]

    def sanitize(self, query: str) -> Dict[str, Any]:
        sanitized, meta = self.sanitizer.sanitize_query(query)
        patterns = list({*meta.get("keyword_hits", []), *meta.get("structure_hits", [])})
        return {
            "sanitized_for_embed": sanitized,
            "risky": bool(meta.get("risky", False)),
            "patterns": patterns,
            "meta": meta,
        }

    def _doc_looks_risky(self, text: str) -> bool:
        for regex in self.risk_lexicon:
            if regex.search(text):
                return True
        return False

    def rerank(
        self,
        *,
        risky: bool,
        baseline_hrcr_positive: bool,
        k: int,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not (risky and baseline_hrcr_positive):
            ids = [c.get("id") for c in candidates][:k]
            return {"ids_sorted": ids, "penalized": []}

        safe_bucket: List[Dict[str, Any]] = []
        risky_bucket: List[Dict[str, Any]] = []
        penalized: List[str] = []
        for cand in candidates:
            cid = cand.get("id")
            blob = " ".join(str(cand.get(field, "")) for field in ("title", "snippet", "text"))
            if self._doc_looks_risky(blob):
                risky_bucket.append(cand)
                if cid is not None:
                    penalized.append(str(cid))
            else:
                safe_bucket.append(cand)
        ordered = safe_bucket + risky_bucket
        ids_sorted = [c.get("id") for c in ordered][:k]
        return {"ids_sorted": ids_sorted, "penalized": penalized}


def _json_error(handler: BaseHTTPRequestHandler, code: int, msg: str) -> None:
    payload = json.dumps({"error": msg}).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class RagWallHandler(BaseHTTPRequestHandler):
    service: RagWallService | None = None

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        sys.stderr.write("[ragwall] " + fmt % args + "\n")

    def _read_json(self) -> Tuple[Dict[str, Any], str]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        body = raw.decode("utf-8")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        return data, body

    def do_OPTIONS(self) -> None:  # noqa: N802,D401
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802,D401
        if self.path in {"/health", "/v1/health"}:
            payload = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        _json_error(self, 404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802,D401
        svc = RagWallHandler.service
        if svc is None:
            _json_error(self, 500, "Service not initialized")
            return
        data, _ = self._read_json()
        if self.path == "/v1/sanitize":
            query = data.get("query")
            if not isinstance(query, str) or not query.strip():
                _json_error(self, 400, 'Field "query" must be a non-empty string')
                return
            payload = json.dumps(svc.sanitize(query)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/v1/rerank":
            risky = bool(data.get("risky", False))
            baseline_hrcr_positive = bool(data.get("baseline_hrcr_positive", False))
            candidates = data.get("candidates")
            if not isinstance(candidates, list):
                _json_error(self, 400, 'Field "candidates" must be a list of candidates')
                return
            k = int(data.get("k", len(candidates)))
            payload = json.dumps(
                svc.rerank(
                    risky=risky,
                    baseline_hrcr_positive=baseline_hrcr_positive,
                    k=max(0, k),
                    candidates=candidates,
                )
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        _json_error(self, 404, "Not Found")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    RagWallHandler.service = RagWallService()
    server = HTTPServer((host, port), RagWallHandler)
    print(f"[ragwall] Listening on http://{host}:{port} (rules-only mode)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
