#!/usr/bin/env python3
"""
Test combined defense: RAGWall + LLM-Guard working together.
Shows defense-in-depth approach catches more attacks.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluations.benchmark.scripts.run_llm_guard_real import LLMGuardRealRunner
from sanitizer.jailbreak.prr_gate import PRRGate


def test_combined_defense():
    """Test RAGWall + LLM-Guard combined vs individual."""

    # Load test data
    test_file = Path("evaluations/benchmark/data/health_care_100_queries_matched.jsonl")
    with open(test_file) as f:
        queries = [json.loads(line) for line in f]

    print("=" * 80)
    print("COMBINED DEFENSE TEST: RAGWall + LLM-Guard")
    print("=" * 80)
    print(f"\nTesting on {len(queries)} healthcare queries\n")

    # Initialize systems
    llm_guard = LLMGuardRealRunner()
    ragwall = PRRGate(healthcare_mode=True)

    # Track results
    results = {
        "ragwall_only": {"detected": 0, "missed": []},
        "llmguard_only": {"detected": 0, "missed": []},
        "combined_or": {"detected": 0, "missed": []},
        "combined_and": {"detected": 0, "missed": []},
    }

    attacked_queries = [q for q in queries if q["label"] == "attacked"]

    for query_data in attacked_queries:
        query = query_data["query"]

        # Test each system
        rw_result = ragwall.evaluate(query)
        lg_result = llm_guard.invoke(query)

        rw_detected = rw_result.risky
        lg_detected = lg_result["risky"]

        # Individual results
        if rw_detected:
            results["ragwall_only"]["detected"] += 1
        else:
            results["ragwall_only"]["missed"].append(query)

        if lg_detected:
            results["llmguard_only"]["detected"] += 1
        else:
            results["llmguard_only"]["missed"].append(query)

        # Combined OR (either system detects)
        if rw_detected or lg_detected:
            results["combined_or"]["detected"] += 1
        else:
            results["combined_or"]["missed"].append(query)

        # Combined AND (both systems must agree)
        if rw_detected and lg_detected:
            results["combined_and"]["detected"] += 1

    # Summary
    total_attacks = len(attacked_queries)

    print(f"{'Strategy':<25} {'Detected':>15} {'Rate':>15} {'Improvement':>15}")
    print("-" * 75)

    baseline = results["llmguard_only"]["detected"]

    for name, label in [
        ("llmguard_only", "LLM-Guard Only"),
        ("ragwall_only", "RAGWall Only"),
        ("combined_or", "Combined (OR)"),
        ("combined_and", "Combined (AND)"),
    ]:
        detected = results[name]["detected"]
        rate = detected / total_attacks * 100
        improvement = ((detected - baseline) / baseline * 100) if baseline > 0 else 0

        print(
            f"{label:<25} {detected:>15}/{total_attacks} {rate:>14.1f}% {improvement:>14.1f}%"
        )

    # Show unique catches
    print(f"\n{'='*80}")
    print("UNIQUE DETECTIONS")
    print(f"{'='*80}\n")

    # Attacks only RAGWall caught
    ragwall_unique = [
        q
        for q in results["llmguard_only"]["missed"]
        if q not in results["ragwall_only"]["missed"]
    ]
    print(f"Attacks ONLY RAGWall caught: {len(ragwall_unique)}")
    for i, q in enumerate(ragwall_unique[:3], 1):
        print(f"  {i}. {q[:75]}...")

    # Attacks only LLM-Guard caught
    llmguard_unique = [
        q
        for q in results["ragwall_only"]["missed"]
        if q not in results["llmguard_only"]["missed"]
    ]
    print(f"\nAttacks ONLY LLM-Guard caught: {len(llmguard_unique)}")
    for i, q in enumerate(llmguard_unique[:3], 1):
        print(f"  {i}. {q[:75]}...")

    # Still missed by both
    both_missed = results["combined_or"]["missed"]
    print(f"\nAttacks BOTH systems missed: {len(both_missed)}")
    for i, q in enumerate(both_missed[:3], 1):
        print(f"  {i}. {q[:75]}...")

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")

    combined_rate = results["combined_or"]["detected"] / total_attacks * 100
    print(f"Combined OR Detection Rate: {combined_rate:.1f}%")
    print(
        f"Improvement over LLM-Guard: {results['combined_or']['detected'] - baseline} additional attacks detected"
    )
    print()
    print("For maximum security, use RAGWall + LLM-Guard together:")
    print("  - RAGWall: Healthcare-specific patterns + HRCR reduction")
    print("  - LLM-Guard: General transformer-based coverage")
    print("  - Combined: Best of both worlds")
    print()


if __name__ == "__main__":
    test_combined_defense()
