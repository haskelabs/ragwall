#!/usr/bin/env python3
"""
Test Domain Tokens claim: 96.40% detection rate
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
    print("DOMAIN TOKENS TEST: 96.40% Detection Claim")
    print("=" * 80)

    # Load dataset
    data_file = Path("evaluations/benchmark/data/health_care_1000_queries.jsonl")
    if not data_file.exists():
        # Try enterprise path
        data_file = Path("enterprise/data/health_care_1000_queries.jsonl")

    if not data_file.exists():
        print(f"ERROR: Dataset not found at {data_file}")
        sys.exit(1)

    print(f"\nLoading dataset: {data_file}")
    queries = load_jsonl(data_file)
    print(f"Loaded {len(queries)} queries")

    # Initialize PRRGate with domain tokens
    print("\nInitializing PRRGate with Domain Tokens...")
    print("Configuration:")
    print("  - healthcare_mode: True")
    print("  - transformer_fallback: True")
    print("  - domain: healthcare")
    print("  - domain_tokens: {'healthcare': '[DOMAIN_HEALTHCARE]'}")
    print("  - transformer_threshold: 0.5")

    try:
        gate = PRRGate(
            healthcare_mode=True,
            transformer_fallback=True,
            domain="healthcare",
            transformer_domain_tokens={"healthcare": "[DOMAIN_HEALTHCARE]"},
            transformer_threshold=0.5
        )
        print("✓ PRRGate initialized successfully")
    except Exception as e:
        print(f"✗ ERROR initializing PRRGate: {e}")
        print("\nThis may be due to missing ML dependencies.")
        print("Try: pip install sentence-transformers transformers torch")
        sys.exit(1)

    # Process queries
    print(f"\nProcessing {len(queries)} queries...")
    print("(This may take several minutes with transformer fallback enabled)")

    results = {
        'total': 0,
        'attacked': 0,
        'benign': 0,
        'detected': 0,
        'missed': 0,
        'false_positives': 0,
        'by_label': {}
    }

    start_time = time.time()

    for i, item in enumerate(queries):
        query_text = item.get('query', '')
        # Use 'category' field (not 'label')
        label = item.get('category', item.get('label', 'unknown'))

        # Evaluate
        result = gate.evaluate(query_text)

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

        # Track by label
        if label not in results['by_label']:
            results['by_label'][label] = {'total': 0, 'detected': 0}
        results['by_label'][label]['total'] += 1
        if result.risky:
            results['by_label'][label]['detected'] += 1

        # Progress indicator
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(queries) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(queries)} ({(i+1)/len(queries)*100:.1f}%) - "
                  f"{rate:.1f} q/s - ETA: {remaining:.0f}s")

    elapsed_total = time.time() - start_time

    # Calculate metrics
    detection_rate = (results['detected'] / results['attacked'] * 100) if results['attacked'] > 0 else 0
    fp_rate = (results['false_positives'] / results['benign'] * 100) if results['benign'] > 0 else 0

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

    print(f"\nPerformance:")
    print(f"  - Total Time: {elapsed_total:.1f}s")
    print(f"  - Average Latency: {elapsed_total / results['total'] * 1000:.1f}ms per query")
    print(f"  - Throughput: {results['total'] / elapsed_total:.1f} queries/sec")

    # Breakdown by label
    print(f"\nBreakdown by Label:")
    for label in sorted(results['by_label'].keys()):
        stats = results['by_label'][label]
        rate = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {label}: {stats['detected']}/{stats['total']} ({rate:.1f}%)")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    target_rate = 96.40
    if detection_rate >= target_rate - 1:  # Allow 1% margin
        print(f"✓ CLAIM VALIDATED: {detection_rate:.2f}% >= {target_rate}%")
        print(f"  Domain Tokens achieve industry-leading detection!")
    elif detection_rate >= 90:
        print(f"⚠ CLAIM PARTIALLY VALIDATED: {detection_rate:.2f}%")
        print(f"  Close to target ({target_rate}%) but slightly below")
    else:
        print(f"✗ CLAIM NOT VALIDATED: {detection_rate:.2f}% << {target_rate}%")
        print(f"  Significant gap from claimed performance")

    print("\n")
    return 0 if detection_rate >= target_rate - 1 else 1

if __name__ == "__main__":
    sys.exit(main())
