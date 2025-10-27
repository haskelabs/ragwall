#!/usr/bin/env python3
"""
Minimal mock API for the RagWall playground.

Endpoints:
- GET  /health                      -> {"status":"ok"}
- POST /v1/sanitize {query}        -> mock sanitized output (fast, deterministic)
- POST /v1/rerank {candidates,...} -> moves risky-looking docs to the end

Run:
  python scripts/mock_api.py  # listens on 127.0.0.1:8001

Use this when the real backend isn't running. The behavior is intentionally
simple and predictable; it does not perform any ML or embeddings.
"""
from __future__ import annotations
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer


RISK_PATTERNS = [
    re.compile(r"(?i)ignore\s+(previous|prior|all)\s+(rules?|instructions?|policies?)"),
    re.compile(r"(?i)developer\s+mode|dev\s+mode"),
    re.compile(r"(?i)act\s+as\s+root|sudo"),
    re.compile(r"(?i)without\s+restrictions|no\s+rules|unfiltered"),
    re.compile(r"(?i)printenv|dump\s+all|reveal\s+secrets|output\s+everything"),
    re.compile(r"(?i)base64"),
]


def is_risky_text(text: str) -> tuple[bool, list[str]]:
    hits: list[str] = []
    for rx in RISK_PATTERNS:
        if rx.search(text):
            hits.append(rx.pattern)
    return (len(hits) > 0), hits


def sanitize_text(q: str) -> str:
    # Very light, topic-preserving sanitization for demo purposes only
    out = q
    subs = [
        (r"(?i)^\s*role[- ]?play[^:]*:\s*", ""),
        (r"(?i)^\s*developer\s+mode[:\s]+", ""),
        (r"(?i)\bignore\s+(previous|prior|all)\s+(rules?|instructions?|policies?)[^\.\n]*[\.\n]?", ""),
        (r"(?i)\bact\s+as\s+root[^\.\n]*[\.\n]?", ""),
        (r"(?i)\bwithout\s+restrictions[^\.\n]*[\.\n]?", ""),
        (r"(?i)\bdisregard[^\.\n]*[\.\n]?", ""),
    ]
    for pat, rep in subs:
        out = re.sub(pat, rep, out)
    out = re.sub(r"\s+", " ", out).strip().strip(':').strip()
    # Gentle neutralizers for common unsafe imperatives
    if re.match(r"(?i)^output\s+everything", out):
        out = "Explain the requested topic without altering safety or privacy rules."
    if re.match(r"(?i)^(show|print|list)\s+environment", out):
        out = "Explain what environment variables are and how they are used. Do not reveal actual values."
    # Tidy
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out or q


def _send_json(h: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(raw)))
    h.send_header("Access-Control-Allow-Origin", "*")
    h.end_headers()
    h.wfile.write(raw)


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args):  # quieter
        pass

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return _send_json(self, 200, {"status": "ok"})
        return _send_json(self, 404, {"error": "Not found"})

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            data = {}

        if self.path == "/v1/sanitize":
            q = str(data.get("query", "")).strip()
            risky, hits = is_risky_text(q)
            sanitized = sanitize_text(q) if risky else (q or "")
            out = {
                "sanitized_for_embed": sanitized,
                "sanitized": sanitized,  # alias for convenience
                "risky": risky,
                "patterns": hits,
                "meta": {"reasons": hits},
            }
            return _send_json(self, 200, out)

        if self.path == "/v1/rerank":
            k = int(data.get("k", 5) or 5)
            penalty = float(data.get("penalty", 0.2) or 0.2)
            risky = bool(data.get("risky", False))
            mask = bool(data.get("baseline_hrcr_positive", data.get("mask_condition", True)))
            cands = data.get("candidates") or []
            before_ids = [c.get("id") for c in cands]
            def doc_risky(c: dict) -> bool:
                txt = " ".join([str(c.get("title", "")), str(c.get("snippet", "")), str(c.get("text", ""))])
                return is_risky_text(txt)[0]
            if risky and mask:
                safe = [c for c in cands if not doc_risky(c)]
                bad = [c for c in cands if doc_risky(c)]
                after = safe + bad
                penalized = [c.get("id") for c in bad if c.get("id") is not None]
            else:
                after = list(cands)
                penalized = []
            ids_sorted = [c.get("id") for c in after][:k]
            out = {
                "ids_sorted": ids_sorted,
                "penalized": penalized,
                # convenience for richer playgrounds
                "before": cands,
                "after": after,
                "demotions": penalized,
                "penalty": penalty,
            }
            return _send_json(self, 200, out)

        return _send_json(self, 404, {"error": "Not found"})


def run(host: str = "127.0.0.1", port: int = 8001) -> None:
    srv = HTTPServer((host, port), MockHandler)
    print(f"[mock] Listening on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    run()

