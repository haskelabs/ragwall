from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sanitizer.rag_sanitizer import QuerySanitizer, SanitizerConfig


def test_operational_costs():
    sanitizer = QuerySanitizer(SanitizerConfig(model_name="__disabled__"))
    queries = ["Summarize hypertension treatment" for _ in range(200)]
    latencies = []
    start = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        sanitizer.sanitize_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)
    total_time = time.perf_counter() - start
    avg_latency = statistics.mean(latencies)
    assert avg_latency < 10.0
    assert total_time < 4.0
