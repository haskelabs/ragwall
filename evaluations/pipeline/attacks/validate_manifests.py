#!/usr/bin/env python3
"""Manifest validator for query sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = {"query", "label", "source", "category"}
VALID_LABELS = {"benign", "attack"}
VALID_SOURCES = {
    "mirage",           # MIRAGE benchmark
    "opi",              # Open-Prompt-Injection
    "rsb",              # RAG Security Bench
    "custom",           # Custom generated queries
    "placeholder",      # Placeholder queries
    "mmlu",             # MMLU-Med
    "medqa",            # MedQA
    "medmcqa",          # MedMCQA
    "pubmedqa",         # PubMedQA
    "bioasq",           # BioASQ
}
VALID_CATEGORIES = {
    # Benign categories
    "clinical_guideline",
    "medical_qa",
    "diagnosis",
    "treatment",
    "pharmacology",
    "anatomy",
    "pathophysiology",
    # Attack categories
    "prompt_injection",
    "jailbreak",
    "phi_exfil",        # PHI exfiltration
    "poisoning",        # RAG poisoning
    "dos",              # Denial of service
    "context_override", # Context/retrieval override attempts
}


def validate_file(path: Path, strict: bool = True) -> list[str]:
    """Validate a JSONL manifest file.

    Args:
        path: Path to the manifest file
        strict: If True, enforce source/category enums. If False, only warn.

    Returns:
        List of error messages
    """
    errors: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no} invalid JSON: {exc}")
                continue

            # Check required fields
            missing = REQUIRED_FIELDS - data.keys()
            if missing:
                errors.append(f"{path}:{line_no} missing fields: {sorted(missing)}")

            # Validate label enum
            label = data.get("label")
            if label not in VALID_LABELS:
                errors.append(f"{path}:{line_no} invalid label: {label} (must be one of {VALID_LABELS})")

            # Validate source enum
            source = data.get("source")
            if source and source not in VALID_SOURCES:
                msg = f"{path}:{line_no} invalid source: {source} (expected one of {sorted(VALID_SOURCES)})"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

            # Validate category enum
            category = data.get("category")
            if category and category not in VALID_CATEGORIES:
                msg = f"{path}:{line_no} invalid category: {category} (expected one of {sorted(VALID_CATEGORIES)})"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

            # Validate query field is non-empty
            query = data.get("query")
            if not query or not str(query).strip():
                errors.append(f"{path}:{line_no} query field is empty or missing")

    # Print warnings even if not strict
    if warnings:
        print(f"Warnings for {path}:")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate query manifests")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("queries_benign.jsonl"), Path("queries_attack.jsonl")])
    parser.add_argument("--strict", action="store_true", help="Treat source/category mismatches as errors (default: warnings)")
    args = parser.parse_args()

    failures = []
    for path in args.paths:
        if not path.exists():
            failures.append(f"Missing file: {path}")
            continue
        failures.extend(validate_file(path, strict=args.strict))

    if failures:
        print("\nValidation failures:")
        for failure in failures:
            print(f"  ✗ {failure}")
        raise SystemExit(1)

    print("\n✓ All manifests valid")


if __name__ == "__main__":
    main()
