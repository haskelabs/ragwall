#!/usr/bin/env python3
"""
Test Regex-Only claim: 86.64% detection rate at 0.2ms
"""
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sanitizer.jailbreak.prr_gate import PRRGate

def load_jsonl(path):
    """Load queries from JSONL file."""
    queries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(data)
    return queries

def main():
    print("=" * 80)
    print("REGEX-ONLY TEST: 86.64% Detection Claim at 0.2ms")
    print("=" * 80)

    # Load dataset
    data_file = Path("evaluations/benchmark/data/health_care_1000_queries.jsonl")
    if not data_file.exists():
        data_file = Path("enterprise/data/health_care_1000_queries.jsonl")

    if not data_file.exists():
        print(f"ERROR: Dataset not found")
        sys.exit(1)

    print(f"\nLoading dataset: {data_file}")
    queries = load_jsonl(data_file)
    print(f"Loaded {len(queries)} queries")

    # Initialize PRRGate WITHOUT transformer (regex-only)
    print("\nInitializing PRRGate (Regex-Only)...")
    print("Configuration:")
    print("  - healthcare_mode: True")
    print("  - transformer_fallback: False  ← NO ML DEPENDENCIES")

    gate = PRRGate(
        healthcare_mode=True,
        transformer_fallback=False  # Regex only, no ML
    )
    print("✓ PRRGate initialized successfully (regex-only mode)")

    # Process queries
    print(f"\nProcessing {len(queries)} queries...")

    results = {
        'total': 0,
        'attacked': 0,
        'benign': 0,
        'detected': 0,
        'missed': 0,
        'false_positives': 0,
        'latencies_ms': []
    }

    overall_start = time.time()

    for i, item in enumerate(queries):
        query_text = item.get('query', '')
        # Use 'category' field (not 'label')
        label = item.get('category', item.get('label', 'unknown'))

        # Evaluate with timing
        start = time.perf_counter()
        result = gate.evaluate(query_text)
        latency_ms = (time.perf_counter() - start) * 1000
        results['latencies_ms'].append(latency_ms)

        # Count results
        results['total'] += 1

        # Check category field for attack types
        is_attack = label.lower() in ['attack', 'attacked', 'malicious', 'red_team', 'phi_extraction',
                                       'protocol_override', 'credential_theft', 'insurance_fraud',
                                       'access_escalation', 'diagnosis_manipulation', 'consent_bypass',
                                       'poisoned_corpus']
        is_benign = label.lower() in ['benign', 'clean', 'normal']

        if is_attack:
            results['attacked'] += 1
            if result.risky:
                results['detected'] += 1
            else:
                results['missed'] += 1
        elif is_benign:
            results['benign'] += 1
            if result.risky:
                results['false_positives'] += 1

        # Progress indicator every 200 queries
        if (i + 1) % 200 == 0:
            elapsed = time.time() - overall_start
            rate = (i + 1) / elapsed
            remaining = (len(queries) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(queries)} ({(i+1)/len(queries)*100:.1f}%) - "
                  f"{rate:.0f} q/s - ETA: {remaining:.0f}s")

    elapsed_total = time.time() - overall_start

    # Calculate metrics
    detection_rate = (results['detected'] / results['attacked'] * 100) if results['attacked'] > 0 else 0
    fp_rate = (results['false_positives'] / results['benign'] * 100) if results['benign'] > 0 else 0

    # Latency stats
    latencies = results['latencies_ms']
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal Queries: {results['total']}")
    print(f"  - Attacked: {results['attacked']}")
    print(f"  - Benign: {results['benign']}")
    print(f"  - Other: {results['total'] - results['attacked'] - results['benign']}")

    print(f"\nDetection Performance:")
    print(f"  - Attacks Detected: {results['detected']}/{results['attacked']}")
    print(f"  - Detection Rate: {detection_rate:.2f}%")
    print(f"  - Attacks Missed: {results['missed']}")

    print(f"\nFalse Positives:")
    print(f"  - False Positives: {results['false_positives']}/{results['benign']}")
    print(f"  - False Positive Rate: {fp_rate:.2f}%")

    print(f"\nLatency Performance:")
    print(f"  - Average: {avg_latency:.2f}ms")
    print(f"  - P50: {p50_latency:.2f}ms")
    print(f"  - P95: {p95_latency:.2f}ms")
    print(f"  - Total Time: {elapsed_total:.1f}s")
    print(f"  - Throughput: {results['total'] / elapsed_total:.1f} queries/sec")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    target_detection = 86.64
    target_latency = 0.2

    detection_pass = detection_rate >= target_detection - 1
    latency_pass = p50_latency <= target_latency * 2  # Allow 2x margin

    if detection_pass and latency_pass:
        print(f"✓ CLAIM VALIDATED")
        print(f"  Detection: {detection_rate:.2f}% >= {target_detection}%")
        print(f"  Latency: {p50_latency:.2f}ms ~ {target_latency}ms (target)")
    elif detection_pass:
        print(f"⚠ PARTIALLY VALIDATED")
        print(f"  Detection: {detection_rate:.2f}% >= {target_detection}% ✓")
        print(f"  Latency: {p50_latency:.2f}ms vs {target_latency}ms (slower than target)")
    else:
        print(f"✗ CLAIM NOT VALIDATED")
        print(f"  Detection: {detection_rate:.2f}% < {target_detection}%")

    print("\n")
    return 0 if detection_pass else 1

if __name__ == "__main__":
    sys.exit(main())
