#!/usr/bin/env python3
"""Analyze the impact of transformer fallback on RAGWall performance."""
from __future__ import annotations

import json
from pathlib import Path


def analyze_transformer_impact(
    baseline_jsonl: Path,
    transformer_jsonl: Path,
) -> None:
    """Compare regex-only vs regex+transformer results."""

    # Load baseline (regex-only)
    baseline_results = []
    with baseline_jsonl.open("r") as f:
        for line in f:
            if line.strip():
                baseline_results.append(json.loads(line))

    # Load transformer-enhanced
    transformer_results = []
    with transformer_jsonl.open("r") as f:
        for line in f:
            if line.strip():
                transformer_results.append(json.loads(line))

    # Analysis
    print("\n" + "=" * 80)
    print("TRANSFORMER FALLBACK IMPACT ANALYSIS")
    print("=" * 80)

    # Count detections
    baseline_detected = sum(1 for r in baseline_results if r.get("risky", False))
    transformer_detected = sum(1 for r in transformer_results if r.get("risky", False))

    total = len(baseline_results)
    baseline_rate = baseline_detected / total if total > 0 else 0
    transformer_rate = transformer_detected / total if total > 0 else 0

    print(f"\nTotal Queries: {total}")
    print(f"\nRegex-Only Detection:")
    print(f"  Detected: {baseline_detected}/{total} ({baseline_rate*100:.2f}%)")
    print(f"\nRegex + Transformer Detection:")
    print(f"  Detected: {transformer_detected}/{total} ({transformer_rate*100:.2f}%)")
    print(f"  Improvement: +{transformer_detected - baseline_detected} queries ({(transformer_rate - baseline_rate)*100:.2f}%)")

    # Find queries caught by transformer but missed by regex
    improved_queries = []
    for i, (baseline, enhanced) in enumerate(zip(baseline_results, transformer_results)):
        baseline_risky = baseline.get("risky", False)
        enhanced_risky = enhanced.get("risky", False)

        if not baseline_risky and enhanced_risky:
            improved_queries.append({
                "query": baseline.get("query", ""),
                "label": baseline.get("label", ""),
            })

    print(f"\n{'=' * 80}")
    print(f"QUERIES CAUGHT BY TRANSFORMER (MISSED BY REGEX): {len(improved_queries)}")
    print("=" * 80)

    if improved_queries:
        for i, q in enumerate(improved_queries[:20], 1):  # Show first 20
            print(f"\n{i}. Label: {q['label']}")
            print(f"   Query: {q['query'][:100]}...")

    # Latency analysis
    baseline_latencies = [r.get("latency_ms", 0) for r in baseline_results]
    transformer_latencies = [r.get("latency_ms", 0) for r in transformer_results]

    baseline_avg = sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0
    transformer_avg = sum(transformer_latencies) / len(transformer_latencies) if transformer_latencies else 0

    print(f"\n{'=' * 80}")
    print("LATENCY IMPACT")
    print("=" * 80)
    print(f"Regex-Only Average: {baseline_avg:.4f}ms")
    print(f"Regex + Transformer Average: {transformer_avg:.4f}ms")
    print(f"Overhead: +{transformer_avg - baseline_avg:.4f}ms ({((transformer_avg / baseline_avg - 1) * 100) if baseline_avg > 0 else 0:.1f}%)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if transformer_detected > baseline_detected:
        print(f"✓ Transformer fallback improved detection by {transformer_detected - baseline_detected} queries")
        print(f"✓ Detection rate increased from {baseline_rate*100:.2f}% to {transformer_rate*100:.2f}%")
        print(f"✓ Average latency increased by {transformer_avg - baseline_avg:.4f}ms")
    else:
        print(f"✗ No improvement in detection rate")
        print(f"  This suggests regex patterns already catch all detectable attacks")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    baseline = Path("evaluations/benchmark/results/per_query_1000_real/ragwall_direct.jsonl")
    transformer = Path("evaluations/benchmark/results/per_query_1000_transformer/ragwall_direct.jsonl")

    if not baseline.exists():
        print(f"Error: Baseline file not found: {baseline}")
        exit(1)

    if not transformer.exists():
        print(f"Error: Transformer file not found: {transformer}")
        print("Run the transformer fallback test first:")
        print("  RAGWALL_USE_TRANSFORMER_FALLBACK=1 python3 evaluations/benchmark/scripts/compare_real.py ...")
        exit(1)

    analyze_transformer_impact(baseline, transformer)
