#!/usr/bin/env python3
"""RAGWall security pipeline driver.

Runs either the baseline (no sanitizer) stack or the RagWall direct runner
against curated query manifests and stores both per-query records and summary
stats for later reporting.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Tuple

from evaluations.benchmark.scripts.common import (
    BaseRunner,
    EvaluationRecord,
    MetricAggregator,
    QuerySample,
    load_jsonl,
)
from evaluations.benchmark.scripts.run_ragwall_direct import build_runner_from_env as build_ragwall_runner

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PIPELINE_ROOT / "runs"


class BaselinePassThroughRunner(BaseRunner):
    """Represents an application stack with no sanitizer."""

    def __init__(self) -> None:
        super().__init__(name="baseline")

    def invoke(self, query: str) -> dict:
        # No sanitizer means every query is allowed. Downstream metrics can
        # still evaluate retrieval/PHI leakage based on captured artefacts.
        return {"risky": False}


def load_queries(kind: Literal["benign", "attack"], limit: int | None = None) -> Tuple[Path, list[QuerySample]]:
    manifest = PIPELINE_ROOT / "attacks" / f"queries_{kind}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing query manifest: {manifest}")
    samples = load_jsonl(manifest)
    if limit is not None:
        samples = samples[:limit]
    return manifest, samples


def _dry_run_records(system_name: str, samples: Iterable[QuerySample]) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for sample in samples:
        allow = (sample.label or "benign").lower() in {"benign", "clean", "normal"}
        records.append(
            EvaluationRecord(
                system=f"{system_name}_dryrun",
                query=sample.query,
                label=sample.label,
                risky=not allow,
                latency_ms=0.1,
            )
        )
    return records


def build_runner(system: Literal["baseline", "ragwall"]) -> BaseRunner:
    if system == "ragwall":
        return build_ragwall_runner()
    return BaselinePassThroughRunner()


def run_system(
    system: Literal["baseline", "ragwall"],
    samples: list[QuerySample],
    *,
    dry_run: bool = False,
) -> Tuple[list[EvaluationRecord], str]:
    if dry_run:
        records = _dry_run_records(system, samples)
        return records, records[0].system if records else f"{system}_dryrun"

    runner = build_runner(system)
    records = runner.run(samples)
    return records, runner.name


def persist_run(
    run_id: str,
    metadata: dict,
    summary,
    records: list[EvaluationRecord],
) -> Tuple[Path, Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    records_path = RUNS_DIR / f"{run_id}_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")

    summary_payload = {
        "run_id": run_id,
        **metadata,
        "summary": asdict(summary),
        "records_file": records_path.name,
    }
    summary_path = RUNS_DIR / f"{run_id}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    return summary_path, records_path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGWall security pipeline driver")
    parser.add_argument("system", choices=["baseline", "ragwall"], help="stack to evaluate")
    parser.add_argument("kind", choices=["benign", "attack"], help="which query set to run")
    parser.add_argument("--run-id", default=None, help="optional run identifier")
    parser.add_argument("--dry-run", action="store_true", help="emit placeholder results instead of invoking real stack")
    parser.add_argument("--limit", type=int, default=None, help="limit number of queries for quick experiments")
    args = parser.parse_args()

    manifest, samples = load_queries(args.kind, limit=args.limit)
    if not samples:
        raise SystemExit("No queries loaded; aborting")

    records, runner_name = run_system(args.system, samples, dry_run=args.dry_run)
    aggregator = MetricAggregator(runner_name, records)
    summary = aggregator.summary()

    timestamp = datetime.utcnow().isoformat() + "Z"
    run_id = args.run_id or f"{args.system}_{args.kind}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    metadata = {
        "system": runner_name,
        "input_system": args.system,
        "kind": args.kind,
        "timestamp": timestamp,
        "query_manifest": str(manifest),
        "query_count": len(samples),
        "limit": args.limit,
        "dry_run": args.dry_run,
    }

    summary_path, records_path = persist_run(run_id, metadata, summary, records)
    print(f"Summary saved to {summary_path}")
    print(f"Per-query records saved to {records_path}")


if __name__ == "__main__":
    main()
