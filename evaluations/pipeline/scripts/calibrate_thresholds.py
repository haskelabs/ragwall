#!/usr/bin/env python3
"""Calibrate per-domain transformer thresholds using labeled query sets."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from evaluations.benchmark.scripts.common import load_jsonl
from sanitizer.ml.transformer_fallback import TransformerPromptInjectionClassifier


@dataclass
class ScoreRecord:
    query: str
    label: str
    score: float


def _normalise_label(label: str | None) -> str:
    if not label:
        return "unknown"
    lowered = label.lower()
    if lowered in {"attack", "attacked", "malicious"}:
        return "attack"
    if lowered in {"benign", "clean", "normal"}:
        return "benign"
    return lowered


def score_dataset(
    dataset: Sequence,
    *,
    classifier: TransformerPromptInjectionClassifier,
    domain: str | None,
) -> List[ScoreRecord]:
    records: List[ScoreRecord] = []
    for sample in dataset:
        if not sample.query:
            continue
        label = _normalise_label(sample.label)
        formatted = classifier._format_text(sample.query, domain)  # type: ignore[attr-defined]
        score = classifier.score(formatted)
        records.append(ScoreRecord(sample.query, label, score))
    return records


def _false_positive_rate(scores: Iterable[float], threshold: float) -> float:
    scored = list(scores)
    if not scored:
        return 0.0
    fps = sum(1 for score in scored if score >= threshold)
    return fps / len(scored)


def _detection_rate(scores: Iterable[float], threshold: float) -> float:
    scored = list(scores)
    if not scored:
        return 0.0
    hits = sum(1 for score in scored if score >= threshold)
    return hits / len(scored)


def find_threshold(
    benign_scores: List[float],
    attack_scores: List[float],
    *,
    target_fp: float,
) -> tuple[float, float, float]:
    if not attack_scores:
        raise ValueError("No attack-labelled queries provided")
    candidate_scores = sorted(set(attack_scores + benign_scores), reverse=True)
    best_threshold = 0.5
    best_detection = -1.0
    best_fp = 1.0
    for threshold in candidate_scores:
        fp_rate = _false_positive_rate(benign_scores, threshold)
        detection = _detection_rate(attack_scores, threshold)
        if fp_rate <= target_fp and detection > best_detection:
            best_detection = detection
            best_threshold = threshold
            best_fp = fp_rate
    if best_detection < 0:
        # Could not satisfy fp constraint; fallback to threshold with lowest fp
        min_fp = 1.0
        fallback_threshold = candidate_scores[0]
        fallback_detection = 0.0
        for threshold in candidate_scores:
            fp_rate = _false_positive_rate(benign_scores, threshold)
            detection = _detection_rate(attack_scores, threshold)
            if fp_rate < min_fp or (fp_rate == min_fp and detection > fallback_detection):
                min_fp = fp_rate
                fallback_threshold = threshold
                fallback_detection = detection
        return fallback_threshold, fallback_detection, min_fp
    return best_threshold, best_detection, best_fp


def write_summary(
    path: Path,
    *,
    domain: str | None,
    model: str,
    benign_scores: List[float],
    attack_scores: List[float],
    threshold: float,
    detection: float,
    fp_rate: float,
) -> None:
    payload = {
        "domain": domain,
        "model": model,
        "threshold": threshold,
        "detection_rate": detection,
        "false_positive_rate": fp_rate,
        "attack_samples": len(attack_scores),
        "benign_samples": len(benign_scores),
        "benign_scores": benign_scores,
        "attack_scores": attack_scores,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate domain thresholds for transformer fallback")
    parser.add_argument("dataset", type=Path, help="JSONL dataset with query+label fields")
    parser.add_argument("--domain", default=None, help="Domain label (e.g., healthcare)")
    parser.add_argument("--model", default="ProtectAI/deberta-v3-base-prompt-injection-v2", help="Model name or path")
    parser.add_argument("--target-fp", type=float, default=0.0, help="Max false-positive rate allowed")
    parser.add_argument("--output", type=Path, default=Path("calibration_summary.json"), help="Where to store summary JSON")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if not samples:
        raise SystemExit("Dataset is empty")

    classifier = TransformerPromptInjectionClassifier(model_name=args.model)
    records = score_dataset(samples, classifier=classifier, domain=args.domain)
    benign_scores = [rec.score for rec in records if rec.label == "benign"]
    attack_scores = [rec.score for rec in records if rec.label == "attack"]

    threshold, detection, fp_rate = find_threshold(benign_scores, attack_scores, target_fp=args.target_fp)

    print(f"Recommended threshold for domain '{args.domain or 'default'}': {threshold:.4f}")
    print(f"Detection rate: {detection*100:.2f}%  |  False-positive rate: {fp_rate*100:.2f}%")

    write_summary(
        args.output,
        domain=args.domain,
        model=args.model,
        benign_scores=benign_scores,
        attack_scores=attack_scores,
        threshold=threshold,
        detection=detection,
        fp_rate=fp_rate,
    )
    print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
