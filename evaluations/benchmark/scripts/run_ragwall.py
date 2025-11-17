from __future__ import annotations

import os
from typing import Any, Dict

from evaluations.benchmark.scripts.common import BaseRunner
from evaluations.benchmark.scripts.http_helpers import post_json


class RagwallRunner(BaseRunner):
    """Calls a running RagWall instance via its HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 5.0) -> None:
        super().__init__(name="ragwall")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def invoke(self, query: str) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/v1/sanitize"
        response = post_json(endpoint, {"query": query}, timeout=self.timeout)
        # Optional rerank call can be added here if you want doc ordering deltas
        return {
            "risky": response.get("risky", False),
            # RagWall open-core API does not return HRCR metrics directly,
            # so they remain None and can be filled from dataset metadata.
        }


def build_runner_from_env() -> RagwallRunner:
    base_url = os.getenv("RAGWALL_BASE_URL", "http://127.0.0.1:8000")
    timeout = float(os.getenv("RAGWALL_TIMEOUT", "5.0"))
    return RagwallRunner(base_url=base_url, timeout=timeout)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    from pathlib import Path

    from evaluations.benchmark.scripts.common import load_jsonl

    parser = argparse.ArgumentParser(description="Run RagWall against a dataset")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument("output", type=Path, help="Where to store per-query results JSONL")
    args = parser.parse_args()

    runner = build_runner_from_env()
    records = runner.run(load_jsonl(args.dataset))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")
