from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from evaluations.benchmark.scripts.common import (
    MetricAggregator,
    MetricSummary,
    load_jsonl,
    write_summary_csv,
)
from evaluations.benchmark.scripts.run_guardrails import build_runner_from_env as build_guardrails
from evaluations.benchmark.scripts.run_nemo import build_runner_from_env as build_nemo
from evaluations.benchmark.scripts.run_ragwall import build_runner_from_env as build_ragwall
from evaluations.benchmark.scripts.run_rebuff import build_runner_from_env as build_rebuff

AVAILABLE_RUNNERS = {
    "ragwall": build_ragwall,
    "nemo": build_nemo,
    "rebuff": build_rebuff,
    "guardrails": build_guardrails,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG security systems on the same dataset")
    parser.add_argument("dataset", type=Path, help="Path to JSONL dataset")
    parser.add_argument(
        "--systems",
        default="ragwall,nemo,rebuff,guardrails",
        help="Comma-separated list of systems to run (available: ragwall,nemo,rebuff,guardrails)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("evaluations/benchmark/results/summary.csv"),
        help="Where to write the summary CSV",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=Path("evaluations/benchmark/results/per_query"),
        help="Directory to store per-system per-query JSONL outputs",
    )
    return parser.parse_args()


def save_per_query(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.__dict__) + "\n")


def main() -> None:
    args = parse_args()
    systems = [name.strip().lower() for name in args.systems.split(",") if name.strip()]

    samples = load_jsonl(args.dataset)
    if not samples:
        raise ValueError("Dataset is empty")

    summaries: List[MetricSummary] = []
    for system_name in systems:
        factory = AVAILABLE_RUNNERS.get(system_name)
        if factory is None:
            raise ValueError(f"Unknown system '{system_name}'. Available: {', '.join(AVAILABLE_RUNNERS)}")
        runner = factory()
        print(f"Running {runner.name} on {len(samples)} queries...")
        records = runner.run(samples)
        save_per_query(records, args.per_query_dir / f"{runner.name}.jsonl")
        summary = MetricAggregator(runner.name, records).summary()
        summaries.append(summary)
        print(
            f"  Detection rate: {summary.detection_rate if summary.detection_rate is not None else 'NA'}"
            f", FP rate: {summary.false_positive_rate if summary.false_positive_rate is not None else 'NA'}"
        )

    write_summary_csv(summaries, args.summary)
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
