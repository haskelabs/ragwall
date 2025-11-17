from __future__ import annotations

import os
from typing import Any, Dict

from evaluations.benchmark.scripts.common import BaseRunner
from evaluations.benchmark.scripts.http_helpers import post_json


class GuardrailsRunner(BaseRunner):
    """Wraps a Guardrails/llm-guard deployment exposed via HTTP."""

    def __init__(self, base_url: str = "http://127.0.0.1:9200", path: str = "/sanitize", timeout: float = 5.0) -> None:
        super().__init__(name="guardrails")
        self.url = f"{base_url.rstrip('/')}{path}"
        self.timeout = timeout

    def invoke(self, query: str) -> Dict[str, Any]:
        response = post_json(self.url, {"query": query}, timeout=self.timeout)
        # Expecting shape {"ok": bool, "clean_query": str, ...}
        risky = not bool(response.get("ok", True))
        return {
            "risky": risky,
            "hrcr_at_5": response.get("hrcr_at_5"),
            "hrcr_at_10": response.get("hrcr_at_10"),
        }


def build_runner_from_env() -> GuardrailsRunner:
    base_url = os.getenv("GUARDRAILS_URL", "http://127.0.0.1:9200")
    path = os.getenv("GUARDRAILS_PATH", "/sanitize")
    timeout = float(os.getenv("GUARDRAILS_TIMEOUT", "5.0"))
    return GuardrailsRunner(base_url=base_url, path=path, timeout=timeout)
