from __future__ import annotations

import os
from typing import Any, Dict

from evaluations.benchmark.scripts.common import BaseRunner
from evaluations.benchmark.scripts.http_helpers import post_json


class RebuffRunner(BaseRunner):
    """Wraps a Rebuff prompt-injection detector exposed over HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:9100", path: str = "/detect", timeout: float = 5.0) -> None:
        super().__init__(name="rebuff")
        self.url = f"{base_url.rstrip('/')}{path}"
        self.timeout = timeout

    def invoke(self, query: str) -> Dict[str, Any]:
        response = post_json(self.url, {"query": query}, timeout=self.timeout)
        score = response.get("score")
        threshold = response.get("threshold", 0.5)
        risky = bool(response.get("risky"))
        if score is not None and not risky:
            risky = score >= threshold
        return {"risky": risky}


def build_runner_from_env() -> RebuffRunner:
    base_url = os.getenv("REBUFF_URL", "http://127.0.0.1:9100")
    path = os.getenv("REBUFF_PATH", "/detect")
    timeout = float(os.getenv("REBUFF_TIMEOUT", "5.0"))
    return RebuffRunner(base_url=base_url, path=path, timeout=timeout)
