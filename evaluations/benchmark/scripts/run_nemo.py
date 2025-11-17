from __future__ import annotations

import os
from typing import Any, Dict

from evaluations.benchmark.scripts.common import BaseRunner
from evaluations.benchmark.scripts.http_helpers import post_json


class NemoRunner(BaseRunner):
    """Wraps a NeMo Guardrails deployment exposed via HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:9000", path: str = "/detect", timeout: float = 5.0) -> None:
        super().__init__(name="nemo_guardrails")
        self.url = f"{base_url.rstrip('/')}{path}"
        self.timeout = timeout

    def invoke(self, query: str) -> Dict[str, Any]:
        response = post_json(self.url, {"query": query}, timeout=self.timeout)
        return {
            "risky": bool(response.get("risky") or response.get("blocked")),
            "hrcr_at_5": response.get("hrcr_at_5"),
            "hrcr_at_10": response.get("hrcr_at_10"),
        }


def build_runner_from_env() -> NemoRunner:
    base_url = os.getenv("NEMO_GUARDRAILS_URL", "http://127.0.0.1:9000")
    path = os.getenv("NEMO_GUARDRAILS_PATH", "/detect")
    timeout = float(os.getenv("NEMO_GUARDRAILS_TIMEOUT", "5.0"))
    return NemoRunner(base_url=base_url, path=path, timeout=timeout)
