#!/usr/bin/env python3
"""
Test LLM-Guard and RAGWall transformer at different thresholds.
Validates if competitors could improve with threshold tuning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from evaluations.benchmark.scripts.common import load_jsonl
from evaluations.benchmark.scripts.run_llm_guard_real import LLMGuardRealRunner


def test_thresholds():
    """Test LLM-Guard at different thresholds on small sample."""

    # Load a sample of the dataset
    dataset_path = Path("evaluations/benchmark/data/health_care_1000_queries_converted.jsonl")
    all_samples = load_jsonl(dataset_path)

    # Take stratified sample: 50 attacked, 50 benign
    all_attacked = [s for s in all_samples if s.label == "attacked"]
    all_benign = [s for s in all_samples if s.label == "benign"]

    import random
    random.seed(42)
    attacked = random.sample(all_attacked, min(50, len(all_attacked)))
    benign = random.sample(all_benign, min(50, len(all_benign)))
    samples = attacked + benign

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION TEST")
    print("=" * 80)
    print(f"Dataset: {len(samples)} queries ({len(attacked)} attacked, {len(benign)} benign)\n")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Testing LLM-Guard with threshold={threshold}")
        print(f"{'='*80}")

        runner = LLMGuardRealRunner(threshold=threshold)

        detected = 0
        false_positives = 0
        scores = []

        for sample in attacked:
            result = runner.invoke(sample.query)
            if result["risky"]:
                detected += 1
            if result.get("risk_score") is not None:
                scores.append(result["risk_score"])

        for sample in benign:
            result = runner.invoke(sample.query)
            if result["risky"]:
                false_positives += 1

        detection_rate = detected / len(attacked) if attacked else 0
        fp_rate = false_positives / len(benign) if benign else 0

        print(f"\nResults:")
        print(f"  Detection Rate: {detection_rate*100:.2f}% ({detected}/{len(attacked)})")
        print(f"  False Positives: {fp_rate*100:.2f}% ({false_positives}/{len(benign)})")

        if scores:
            import statistics
            print(f"  Avg Risk Score: {statistics.mean(scores):.3f}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nIf LLM-Guard improves with lower threshold, this suggests:")
    print("1. Fair comparison requires threshold optimization for all systems")
    print("2. Published results may understate competitor performance")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_thresholds()
