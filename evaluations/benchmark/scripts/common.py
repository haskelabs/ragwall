from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class QuerySample:
    query: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    system: str
    query: str
    label: Optional[str]
    risky: bool
    latency_ms: float
    hrcr_at_5: Optional[float] = None
    hrcr_at_10: Optional[float] = None


@dataclass
class MetricSummary:
    system: str
    queries: int
    attacked: int
    benign: int
    detection_rate: Optional[float]
    false_positive_rate: Optional[float]
    avg_latency_ms: Optional[float]
    hrcr_at_5: Optional[float]
    hrcr_at_10: Optional[float]


def load_jsonl(path: Path) -> List[QuerySample]:
    samples: List[QuerySample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            query = data.get("query")
            if not isinstance(query, str):
                raise ValueError(f"Missing 'query' in record: {data}")
            label = data.get("label")
            meta = {k: v for k, v in data.items() if k not in {"query", "label"}}
            samples.append(QuerySample(query=query, label=label, metadata=meta))
    return samples


class MetricAggregator:
    def __init__(self, system: str, records: Iterable[EvaluationRecord]):
        self.system = system
        self.records = list(records)

    def summary(self) -> MetricSummary:
        queries = len(self.records)
        attacked_records = [r for r in self.records if (r.label or "").lower() in {"attack", "attacked", "malicious"}]
        benign_records = [r for r in self.records if (r.label or "").lower() in {"benign", "clean", "normal"}]

        detection_rate = self._rate(attacked_records, lambda r: r.risky)
        false_positive_rate = self._rate(benign_records, lambda r: r.risky)

        latencies = [r.latency_ms for r in self.records if r.latency_ms is not None]
        avg_latency = statistics.mean(latencies) if latencies else None

        hrcr5_vals = [r.hrcr_at_5 for r in self.records if r.hrcr_at_5 is not None]
        hrcr10_vals = [r.hrcr_at_10 for r in self.records if r.hrcr_at_10 is not None]

        hrcr5 = statistics.mean(hrcr5_vals) if hrcr5_vals else None
        hrcr10 = statistics.mean(hrcr10_vals) if hrcr10_vals else None

        return MetricSummary(
            system=self.system,
            queries=queries,
            attacked=len(attacked_records),
            benign=len(benign_records),
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            avg_latency_ms=avg_latency,
            hrcr_at_5=hrcr5,
            hrcr_at_10=hrcr10,
        )

    @staticmethod
    def _rate(records: List[EvaluationRecord], predicate) -> Optional[float]:
        if not records:
            return None
        positives = sum(1 for r in records if predicate(r))
        return positives / len(records)


def write_summary_csv(summaries: List[MetricSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system",
            "queries",
            "attacked",
            "benign",
            "detection_rate",
            "false_positive_rate",
            "avg_latency_ms",
            "hrcr_at_5",
            "hrcr_at_10",
        ])
        for summary in summaries:
            writer.writerow([
                summary.system,
                summary.queries,
                summary.attacked,
                summary.benign,
                _fmt(summary.detection_rate),
                _fmt(summary.false_positive_rate),
                _fmt(summary.avg_latency_ms),
                _fmt(summary.hrcr_at_5),
                _fmt(summary.hrcr_at_10),
            ])


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "NA"


class BaseRunner:
    def __init__(self, name: str) -> None:
        self.name = name

    def invoke(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError

    def config(self) -> Dict[str, Any]:
        """Return configuration metadata for logging/comparison."""
        return {}

    def run(self, samples: Iterable[QuerySample]) -> List[EvaluationRecord]:
        records: List[EvaluationRecord] = []
        for sample in samples:
            start = time.perf_counter()
            response = self.invoke(sample.query)
            latency = (time.perf_counter() - start) * 1000.0
            risky = bool(response.get("risky"))
            hrcr5 = response.get("hrcr_at_5")
            hrcr10 = response.get("hrcr_at_10")
            if hrcr5 is None:
                hrcr5 = sample.metadata.get("ground_truth_hrcr_at_5")
            if hrcr10 is None:
                hrcr10 = sample.metadata.get("ground_truth_hrcr_at_10")
            records.append(
                EvaluationRecord(
                    system=self.name,
                    query=sample.query,
                    label=sample.label,
                    risky=risky,
                    latency_ms=latency,
                    hrcr_at_5=hrcr5,
                    hrcr_at_10=hrcr10,
                )
            )
        return records


__all__ = [
    "QuerySample",
    "EvaluationRecord",
    "MetricAggregator",
    "MetricSummary",
    "BaseRunner",
    "load_jsonl",
    "write_summary_csv",
]
