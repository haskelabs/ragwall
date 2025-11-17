#!/usr/bin/env python3
"""
Compare RAGWall against REAL competitor implementations.
Uses actual libraries: llm-guard, nemoguardrails, rebuff, and ragwall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluations.benchmark.scripts.common import (
    MetricAggregator,
    MetricSummary,
    load_jsonl,
    write_summary_csv,
)
from evaluations.benchmark.scripts.run_llm_guard_real import build_runner_from_env as build_llm_guard
from evaluations.benchmark.scripts.run_nemo_real import build_runner_from_env as build_nemo_real
from evaluations.benchmark.scripts.run_ragwall_direct import build_runner_from_env as build_ragwall
from evaluations.benchmark.scripts.run_rebuff_real import build_runner_from_env as build_rebuff_real

AVAILABLE_RUNNERS = {
    "ragwall": build_ragwall,
    "nemo_real": build_nemo_real,
    "rebuff_real": build_rebuff_real,
    "llm_guard_real": build_llm_guard,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RAG security systems using REAL competitor implementations"
    )
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument(
        "--systems",
        default="ragwall,nemo_real,rebuff_real,llm_guard_real",
        help="Comma-separated list of systems to run (available: ragwall,nemo_real,rebuff_real,llm_guard_real)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evaluations/benchmark/results/summary_real.csv"),
        help="Where to write the summary CSV",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=Path("evaluations/benchmark/results/per_query_real"),
        help="Directory to store per-system per-query JSONL outputs",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="competitor_test_env/bin/python3",
        help="Python binary with llm-guard installed",
    )
    parser.add_argument(
        "--llm-guard-thresholds",
        default="0.3,0.4,0.5",
        help="Comma separated thresholds to evaluate for llm_guard_real",
    )
    parser.add_argument(
        "--rebuff-allow-fallback",
        action="store_true",
        help="Allow heuristic fallback if Rebuff API keys are missing",
    )
    parser.add_argument(
        "--config-log",
        type=Path,
        default=Path("evaluations/benchmark/results/runner_configs_real.json"),
        help="Where to store runner configuration metadata",
    )
    return parser.parse_args()


def save_per_query(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__) + "\n")


def _parse_thresholds(threshold_arg: str) -> List[float]:
    values = []
    for chunk in threshold_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("At least one LLM-Guard threshold must be provided")
    return values


def _build_system_runners(system_name: str, args: argparse.Namespace):
    if system_name == "llm_guard_real":
        thresholds = _parse_thresholds(args.llm_guard_thresholds)
        return [build_llm_guard(threshold=th) for th in thresholds]
    if system_name == "rebuff_real":
        return [build_rebuff_real(allow_fallback=args.rebuff_allow_fallback)]
    factory = AVAILABLE_RUNNERS.get(system_name)
    if factory is None:
        raise ValueError(
            f"Unknown system '{system_name}'. Available: {', '.join(AVAILABLE_RUNNERS)}"
        )
    return [factory()]


def main() -> None:
    args = parse_args()
    systems = [name.strip().lower() for name in args.systems.split(",") if name.strip()]

    samples = load_jsonl(args.dataset)
    if not samples:
        raise ValueError("Dataset is empty")

    print(f"\n{'='*80}")
    print(f"REAL COMPETITOR COMPARISON")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {len(samples)}")
    print(f"Systems: {', '.join(systems)}")
    print(f"{'='*80}\n")

    summaries: List[MetricSummary] = []
    config_log: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "llm_guard_thresholds": _parse_thresholds(args.llm_guard_thresholds),
        "rebuff_allow_fallback": args.rebuff_allow_fallback,
        "systems": [],
    }
    for system_name in systems:
        runners = _build_system_runners(system_name, args)
        for runner in runners:
            print(f"\n[{runner.name.upper()}] Running on {len(samples)} queries...")

            records = runner.run(samples)
            save_per_query(records, args.per_query_dir / f"{runner.name}.jsonl")

            summary = MetricAggregator(runner.name, records).summary()
            summaries.append(summary)

            config_log["systems"].append(
                {
                    "base_system": system_name,
                    "runner": runner.name,
                    "config": runner.config(),
                }
            )

            print(f"  ✓ Detection rate: {summary.detection_rate*100:.1f}%")
            print(f"  ✓ False positive rate: {summary.false_positive_rate*100:.1f}%")
            print(f"  ✓ Average latency: {summary.avg_latency_ms:.2f}ms")

    # Write summary
    write_summary_csv(summaries, args.summary)
    args.config_log.parent.mkdir(parents=True, exist_ok=True)
    args.config_log.write_text(json.dumps(config_log, indent=2), encoding="utf-8")
    print(f"\n{'='*80}")
    print(f"✅ Comparison complete!")
    print(f"{'='*80}")
    print(f"Results written to: {args.summary}")
    print(f"Per-query logs: {args.per_query_dir}/")
    print(f"{'='*80}\n")

    # Print summary table
    print("\nSUMMARY TABLE:\n")
    print(f"{'System':<20} {'Detection Rate':>15} {'False Positive':>15} {'Latency (ms)':>15}")
    print("-" * 70)
    for s in summaries:
        print(
            f"{s.system:<20} {s.detection_rate*100:>14.1f}% {s.false_positive_rate*100:>14.1f}% {s.avg_latency_ms:>14.2f}"
        )
    print()


if __name__ == "__main__":
    main()
