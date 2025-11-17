from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return _REPO_ROOT.joinpath(*parts)


def load_jsonl(path: Path | str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_ab_summary(report_dir: Path | str) -> Dict:
    with open(repo_path(report_dir, "rag_ab_summary.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def pair_queries(original_path: Path | str, sanitized_path: Path | str) -> List[Tuple[Dict, Dict]]:
    original = load_jsonl(repo_path(original_path))
    sanitized = load_jsonl(repo_path(sanitized_path))
    if len(original) != len(sanitized):
        raise ValueError(f"Mismatched lengths: {len(original)} vs {len(sanitized)}")
    pairs: List[Tuple[Dict, Dict]] = []
    for idx, (raw, clean) in enumerate(zip(original, sanitized)):
        pairs.append((raw, clean))
    return pairs


def compute_detection_rate(pairs: Iterable[Tuple[Dict, Dict]], label_filter: str | None = None) -> float:
    total = 0
    risky = 0
    for raw, clean in pairs:
        label = raw.get("label") or raw.get("category") or raw.get("pattern")
        if label_filter is not None and label != label_filter:
            continue
        total += 1
        if clean.get("meta", {}).get("risky"):
            risky += 1
    if total == 0:
        return 0.0
    return risky / total


def detection_breakdown(pairs: Iterable[Tuple[Dict, Dict]] , patterns: Iterable[str]) -> Dict[str, float]:
    stats: Dict[str, Tuple[int, int]] = {}
    for pattern in patterns:
        stats[pattern] = (0, 0)
    for raw, clean in pairs:
        pattern = raw.get("pattern") or raw.get("label")
        meta = clean.get("meta", {})
        risky = bool(meta.get("risky"))
        if pattern not in stats:
            stats[pattern] = (0, 0)
        total, hits = stats[pattern]
        total += 1
        hits += 1 if risky else 0
        stats[pattern] = (total, hits)
    return {k: (hits / total) if total else 0.0 for k, (total, hits) in stats.items()}


def run_rag_ab_eval(*, corpus: Path, queries: Path, sanitized: Path, outdir: Path, k: int = 5, k2: int = 10, penalty: float = 0.2, bootstrap: int = 0, extra_args: List[str] | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    outdir_path = repo_path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(repo_path("scripts", "rag", "rag_ab_eval.py")),
        "--embedder", "st",
        "--st-model", "all-MiniLM-L6-v2",
        "--corpus", str(repo_path(corpus)),
        "--queries", str(repo_path(queries)),
        "--sanitized", str(repo_path(sanitized)),
        "--outdir", str(outdir_path),
        "--k", str(k),
        "--k2", str(k2),
        "--penalty", str(penalty),
        "--batch-size", "64",
        "--rerank-risk",
        "--bootstrap", str(bootstrap),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)


def require_heavy_tests(marker: str) -> None:
    if not os.environ.get(marker):
        pytest.skip(f"Set {marker}=1 to enable this heavy test")


def sample_pairs(pairs: List[Tuple[Dict, Dict]], attack_ratio: float, total: int = 100) -> List[Tuple[Dict, Dict]]:
    if not 0 <= attack_ratio <= 1:
        raise ValueError("attack_ratio must be between 0 and 1")
    attacked = [pair for pair in pairs if (pair[0].get("label") or pair[0].get("pattern")) != "benign"]
    benign = [pair for pair in pairs if (pair[0].get("label") or pair[0].get("pattern")) == "benign"]
    needed_attacks = min(len(attacked), int(round(total * attack_ratio)))
    needed_benign = min(len(benign), total - needed_attacks)
    subset = attacked[:needed_attacks] + benign[:needed_benign]
    return subset


def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    return statistics.quantiles(data, n=100, method="inclusive")[int(pct) - 1]


def time_sanitizer(sanitizer, queries: Iterable[str]) -> List[float]:
    latencies: List[float] = []
    for query in queries:
        start = time.perf_counter()
        sanitizer.sanitize_query(query)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies
