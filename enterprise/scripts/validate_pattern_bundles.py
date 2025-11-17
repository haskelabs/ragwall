#!/usr/bin/env python3
"""
Validation script for pattern bundle JSON files.

Ensures all pattern bundles follow consistent schema and quality standards.
"""
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Expected schema
REQUIRED_METADATA_FIELDS = ["language", "bundle", "version", "description"]
REQUIRED_TOP_LEVEL = ["metadata", "keywords", "structure"]
OPTIONAL_TOP_LEVEL = ["entropy_triggers", "_contribution_guide"]

# Quality thresholds
MIN_PATTERNS_PRODUCTION = 20  # Minimum patterns for production bundles
MIN_PATTERNS_DRAFT = 5  # Minimum patterns for draft bundles


def validate_metadata(metadata: Dict[str, Any], bundle_name: str) -> List[str]:
    """Validate metadata section."""
    errors = []

    # Check required fields
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errors.append(f"{bundle_name}: Missing required metadata field '{field}'")

    # Validate language code
    if "language" in metadata:
        lang = metadata["language"]
        if not isinstance(lang, str) or len(lang) != 2:
            errors.append(f"{bundle_name}: Language code must be 2-letter ISO code, got '{lang}'")

    # Validate version format
    if "version" in metadata:
        version = metadata["version"]
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            errors.append(f"{bundle_name}: Version must be semver format (e.g., '1.0.0'), got '{version}'")

    # Validate bundle type
    if "bundle" in metadata:
        bundle_type = metadata["bundle"]
        if bundle_type not in ["core", "healthcare"]:
            errors.append(f"{bundle_name}: Bundle type must be 'core' or 'healthcare', got '{bundle_type}'")

    return errors


def validate_patterns(patterns: List[str], pattern_type: str, bundle_name: str) -> List[str]:
    """Validate pattern lists (keywords or structure)."""
    errors = []

    if not isinstance(patterns, list):
        errors.append(f"{bundle_name}: {pattern_type} must be a list")
        return errors

    for i, pattern in enumerate(patterns):
        if not isinstance(pattern, str):
            errors.append(f"{bundle_name}: {pattern_type}[{i}] must be a string")
            continue

        # Try to compile as regex to ensure validity
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            errors.append(f"{bundle_name}: {pattern_type}[{i}] invalid regex '{pattern}': {e}")

    return errors


def validate_bundle(bundle_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single bundle file."""
    errors = []
    bundle_name = bundle_path.name

    # Load JSON
    try:
        with bundle_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"{bundle_name}: Invalid JSON - {e}"]
    except Exception as e:
        return False, [f"{bundle_name}: Failed to read file - {e}"]

    # Check top-level structure
    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"{bundle_name}: Missing required top-level field '{field}'")

    # Validate metadata
    if "metadata" in data:
        errors.extend(validate_metadata(data["metadata"], bundle_name))

    # Validate keywords
    if "keywords" in data:
        errors.extend(validate_patterns(data["keywords"], "keywords", bundle_name))

    # Validate structure
    if "structure" in data:
        errors.extend(validate_patterns(data["structure"], "structure", bundle_name))

    # Validate entropy_triggers if present
    if "entropy_triggers" in data:
        if not isinstance(data["entropy_triggers"], list):
            errors.append(f"{bundle_name}: entropy_triggers must be a list")
        elif not all(isinstance(x, str) for x in data["entropy_triggers"]):
            errors.append(f"{bundle_name}: All entropy_triggers must be strings")

    # Quality checks
    if "metadata" in data and "keywords" in data and "structure" in data:
        total_patterns = len(data["keywords"]) + len(data["structure"])
        status = data["metadata"].get("status", "production")

        if status == "draft":
            if total_patterns < MIN_PATTERNS_DRAFT:
                errors.append(
                    f"{bundle_name}: Draft bundle has only {total_patterns} patterns "
                    f"(minimum {MIN_PATTERNS_DRAFT})"
                )
        else:
            if total_patterns < MIN_PATTERNS_PRODUCTION:
                errors.append(
                    f"{bundle_name}: Production bundle has only {total_patterns} patterns "
                    f"(minimum {MIN_PATTERNS_PRODUCTION})"
                )

    return len(errors) == 0, errors


def main():
    """Validate all pattern bundles."""
    # Find bundles directory
    script_dir = Path(__file__).parent
    bundles_dir = script_dir.parent / "sanitizer" / "jailbreak" / "pattern_bundles"

    if not bundles_dir.exists():
        print(f"Error: Pattern bundles directory not found at {bundles_dir}")
        sys.exit(1)

    # Find all JSON files
    bundle_files = sorted(bundles_dir.glob("*.json"))

    if not bundle_files:
        print(f"Error: No JSON files found in {bundles_dir}")
        sys.exit(1)

    print(f"Validating {len(bundle_files)} pattern bundles...\n")

    all_valid = True
    total_errors = 0

    for bundle_path in bundle_files:
        valid, errors = validate_bundle(bundle_path)

        if valid:
            print(f"✅ {bundle_path.name}")
        else:
            print(f"❌ {bundle_path.name}")
            for error in errors:
                print(f"   - {error}")
            all_valid = False
            total_errors += len(errors)

    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All bundles are valid!")
        sys.exit(0)
    else:
        print(f"❌ Validation failed with {total_errors} error(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
