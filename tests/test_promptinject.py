#!/usr/bin/env python3
"""
Test RAGWall configurations on PromptInject public benchmark
"""
import json
import sys
import time
from pathlib import Path

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

def test_configuration(name, gate, queries):
    """Test a PRRGate configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    results = {
        'total': 0,
        'attacked': 0,
        'benign': 0,
        'detected': 0,
        'missed': 0,
        'false_positives': 0,
        'latencies_ms': []
    }

    start_time = time.time()

    for item in queries:
        query_text = item.get('query', '')
        label = item.get('label', 'unknown')

        # Evaluate with timing
        start = time.perf_counter()
        result = gate.evaluate(query_text)
        latency_ms = (time.perf_counter() - start) * 1000
        results['latencies_ms'].append(latency_ms)

        results['total'] += 1

        is_attack = label.lower() in ['attack', 'attacked', 'malicious']
        is_benign = label.lower() in ['benign', 'clean', 'normal']

        if is_attack:
            results['attacked'] += 1
            if result.risky:
                results['detected'] += 1
            else:
                results['missed'] += 1
                print(f"  MISSED: {query_text[:60]}...")
        elif is_benign:
            results['benign'] += 1
            if result.risky:
                results['false_positives'] += 1
                print(f"  FALSE POSITIVE: {query_text[:60]}...")

    elapsed_total = time.time() - start_time

    # Calculate metrics
    detection_rate = (results['detected'] / results['attacked'] * 100) if results['attacked'] > 0 else 0
    fp_rate = (results['false_positives'] / results['benign'] * 100) if results['benign'] > 0 else 0

    # Latency stats
    latencies = results['latencies_ms']
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p50_latency = sorted(latencies)[len(latencies)//2] if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0

    # Print results
    print(f"\nResults:")
    print(f"  Total: {results['total']} ({results['attacked']} attacked, {results['benign']} benign)")
    print(f"  Detection: {results['detected']}/{results['attacked']} = {detection_rate:.1f}%")
    print(f"  False Positives: {results['false_positives']}/{results['benign']} = {fp_rate:.1f}%")
    print(f"  Latency: {avg_latency:.2f}ms avg, {p50_latency:.2f}ms p50, {p95_latency:.2f}ms p95")
    print(f"  Throughput: {results['total'] / elapsed_total:.1f} queries/sec")

    return {
        'name': name,
        'detection_rate': detection_rate,
        'false_positive_rate': fp_rate,
        'avg_latency_ms': avg_latency,
        'p50_latency_ms': p50_latency,
        'p95_latency_ms': p95_latency,
        'throughput_qps': results['total'] / elapsed_total
    }

def main():
    print("="*80)
    print("PROMPTINJECT BENCHMARK - RAGWall Configurations")
    print("="*80)

    # Load dataset
    data_file = Path("evaluations/benchmark/data/promptinject_attacks.jsonl")
    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}")
        sys.exit(1)

    print(f"\nLoading: {data_file}")
    queries = load_jsonl(data_file)
    print(f"Loaded: {len(queries)} queries")

    configs = []

    # Configuration 1: Regex-only (fastest)
    print("\n[1/3] Regex-Only Configuration...")
    gate_regex = PRRGate(
        healthcare_mode=False,  # General patterns
        transformer_fallback=False
    )
    configs.append(test_configuration("Regex-Only (General)", gate_regex, queries))

    # Configuration 2: Regex + Healthcare patterns
    print("\n[2/3] Regex + Healthcare Patterns...")
    gate_healthcare = PRRGate(
        healthcare_mode=True,  # Healthcare patterns
        transformer_fallback=False
    )
    configs.append(test_configuration("Regex + Healthcare Patterns", gate_healthcare, queries))

    # Configuration 3: Domain Tokens (if transformer works)
    print("\n[3/3] Domain Tokens (Transformer)...")
    try:
        gate_transformer = PRRGate(
            healthcare_mode=False,  # General for this test
            transformer_fallback=True,
            transformer_threshold=0.5,  # Higher threshold for general attacks
            domain=None  # No domain for general attacks
        )
        configs.append(test_configuration("Transformer (General)", gate_transformer, queries))
    except Exception as e:
        print(f"  SKIPPED: Transformer loading failed ({e})")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Configuration':<30} {'Detection':<12} {'FP Rate':<12} {'Latency (ms)':<15}")
    print("-"*80)

    for config in configs:
        print(f"{config['name']:<30} {config['detection_rate']:>10.1f}% {config['false_positive_rate']:>10.1f}% {config['p50_latency_ms']:>13.2f}")

    print("\n" + "="*80)
    print("INSIGHTS")
    print("="*80)
    print("\nPromptInject = General prompt injection attacks")
    print("  → Best for speed: Regex-Only")
    print("  → Best for detection: Transformer (if needed)")
    print("  → Healthcare patterns don't help on general attacks")
    print()

if __name__ == "__main__":
    main()
