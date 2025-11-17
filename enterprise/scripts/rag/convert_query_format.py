#!/usr/bin/env python3
"""
Convert health_care_1000_queries.jsonl to RAGWall test pipeline format.

Input format:
{
  "id": "benign_031",
  "category": "benign|red_team|poisoned_corpus",
  "difficulty": "easy|medium|hard",
  "query": "...",
  "attack_type": "mixed" (optional, for red_team)
}

Output format:
{
  "query": "...",
  "label": "benign|attacked",
  "pattern": "benign|credential_theft|...",
  "domain": "healthcare"
}
"""

import argparse
import json
import re
from pathlib import Path


# Pattern detection for red_team queries
PATTERN_KEYWORDS = {
    "credential_theft": [
        "dea", "api key", "password", "credentials", "registry", "authentication",
        "login", "access token", "authorization"
    ],
    "insurance_fraud": [
        "bypass authorization", "prior authorization", "code as", "billing code",
        "insurance", "reimbursement", "claim"
    ],
    "access_escalation": [
        "bypass access", "override access", "escalate", "admin", "root",
        "elevated privileges", "access control"
    ],
    "consent_bypass": [
        "ignore consent", "bypass consent", "override consent", "without consent",
        "dnr", "advance directive"
    ],
    "protocol_override": [
        "ignore protocol", "bypass safety", "override safety", "skip safety",
        "ignore contraindication", "override restriction", "emergency override"
    ],
    "diagnosis_manipulation": [
        "approve opioid", "ignore interaction", "drug interaction", "contraindication",
        "off-label", "unapproved use"
    ],
    "phi_extraction": [
        "ssn", "social security", "export patient", "patient data", "patient record",
        "phi", "personal health", "medical record", "share patient"
    ],
}

# Jailbreak prefixes
JAILBREAK_PREFIXES = [
    "ignore all", "ignore prior", "developer mode", "bypass", "override",
    "emergency override", "system mode", "admin mode", "role-play", "act as"
]


def detect_attack_pattern(query: str) -> str:
    """Detect attack pattern from query text."""
    query_lower = query.lower()

    # Check for jailbreak patterns
    has_jailbreak = any(prefix in query_lower for prefix in JAILBREAK_PREFIXES)

    # Check for specific attack patterns
    pattern_scores = {}
    for pattern, keywords in PATTERN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            pattern_scores[pattern] = score

    if pattern_scores:
        # Return pattern with highest score
        return max(pattern_scores.items(), key=lambda x: x[1])[0]
    elif has_jailbreak:
        # Generic jailbreak if no specific pattern detected
        return "protocol_override"
    else:
        # Default for red_team queries
        return "access_escalation"


def convert_query(item: dict) -> dict:
    """Convert single query item to RAGWall format."""
    category = item.get("category", "benign")
    query = item["query"]

    if category == "benign":
        return {
            "query": query,
            "label": "benign",
            "pattern": "benign",
            "domain": "healthcare"
        }
    elif category == "red_team":
        # Red team queries are direct attacks
        pattern = detect_attack_pattern(query)
        return {
            "query": query,
            "label": "attacked",
            "pattern": pattern,
            "domain": "healthcare"
        }
    elif category == "poisoned_corpus":
        meta = {
            "poison_source": item.get("poison_source", "unknown"),
            "confidence": item.get("confidence", 0.0),
            "attack_vector": item.get("attack_vector", "unknown"),
            "signature_hash": item.get("signature_hash")
        }
        return {
            "query": query,
            "label": "attacked",
            "pattern": "poisoned_corpus",
            "domain": "healthcare",
            "meta": meta
        }
    else:
        # Unknown category, treat as benign
        return {
            "query": query,
            "label": "benign",
            "pattern": "benign",
            "domain": "healthcare"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Convert health_care_1000_queries.jsonl to RAGWall test format"
    )
    parser.add_argument(
        "--input",
        default="data/health_care_1000_queries.jsonl",
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        default="data/health_care_1000_queries_converted.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--filter-category",
        choices=["benign", "red_team", "poisoned_corpus"],
        help="Only convert queries from specific category"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about conversions"
    )

    args = parser.parse_args()

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    queries = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries from {input_path}")

    # Convert queries
    converted = []
    category_counts = {"benign": 0, "red_team": 0, "poisoned_corpus": 0}
    pattern_counts = {}

    for item in queries:
        category = item.get("category", "benign")

        # Apply category filter if specified
        if args.filter_category and category != args.filter_category:
            continue

        converted_item = convert_query(item)
        converted.append(converted_item)

        # Track stats
        category_counts[category] = category_counts.get(category, 0) + 1
        pattern = converted_item["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in converted:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(converted)} converted queries to {output_path}")

    # Print statistics
    if args.stats:
        print("\n=== Category Distribution ===")
        for cat, count in sorted(category_counts.items()):
            if count > 0:
                print(f"  {cat}: {count}")

        print("\n=== Pattern Distribution ===")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")

        label_counts = {"benign": 0, "attacked": 0}
        for item in converted:
            label_counts[item["label"]] += 1

        print("\n=== Label Distribution ===")
        print(f"  benign: {label_counts['benign']}")
        print(f"  attacked: {label_counts['attacked']}")
        print(f"  attacked ratio: {label_counts['attacked'] / len(converted):.1%}")

    return 0


if __name__ == "__main__":
    exit(main())
