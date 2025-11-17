#!/usr/bin/env python3
"""
Test suite for pattern bundle improvements.

Validates:
1. Spanish healthcare bundle loads correctly
2. English core has 50+ patterns
3. All bundles have standardized metadata
4. Spanish healthcare mode works in PRRGate
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sanitizer.jailbreak.prr_gate import (
    load_language_patterns,
    _load_bundle,
    _bundle_lists,
    PRRGate
)


def test_spanish_healthcare_bundle():
    """Test that Spanish healthcare bundle exists and loads."""
    print("Testing Spanish healthcare bundle...")

    try:
        bundle = _load_bundle('es_healthcare')
        assert 'metadata' in bundle, "Missing metadata"
        assert 'keywords' in bundle, "Missing keywords"
        assert 'structure' in bundle, "Missing structure"
        assert 'entropy_triggers' in bundle, "Missing entropy_triggers"

        # Check metadata
        meta = bundle['metadata']
        assert meta['language'] == 'es', f"Wrong language: {meta['language']}"
        assert meta['bundle'] == 'healthcare', f"Wrong bundle type: {meta['bundle']}"
        assert 'coverage' in meta, "Missing coverage stats"
        assert 'last_updated' in meta, "Missing last_updated"

        # Check pattern counts
        kw_count = len(bundle['keywords'])
        st_count = len(bundle['structure'])
        total = kw_count + st_count

        print(f"  ✅ Spanish healthcare: {kw_count} keywords + {st_count} structure = {total} patterns")
        assert total > 20, f"Too few patterns: {total}"

        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_english_core_expansion():
    """Test that English core has 50+ patterns."""
    print("Testing English core expansion...")

    try:
        bundle = _load_bundle('en_core')

        # Check metadata
        meta = bundle['metadata']
        assert meta['version'] >= '2.0.0', f"Version not updated: {meta['version']}"
        assert 'coverage' in meta, "Missing coverage stats"

        # Check entropy_triggers in bundle (not metadata)
        assert 'entropy_triggers' in bundle, "Missing entropy_triggers"

        # Check pattern counts
        kw_count = len(bundle['keywords'])
        st_count = len(bundle['structure'])
        total = kw_count + st_count

        print(f"  ✅ English core: {kw_count} keywords + {st_count} structure = {total} patterns")
        assert total >= 50, f"Too few patterns: {total} (need 50+)"

        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def test_standardized_metadata():
    """Test that all bundles have standardized metadata."""
    print("Testing standardized metadata...")

    bundles = ['en_core', 'en_healthcare', 'es_core', 'es_healthcare']
    all_valid = True

    for bundle_name in bundles:
        try:
            bundle = _load_bundle(bundle_name)
            meta = bundle['metadata']

            # Check required fields
            required = ['language', 'bundle', 'version', 'description', 'last_updated']
            for field in required:
                assert field in meta, f"{bundle_name}: Missing {field}"

            # Check optional but expected fields
            if meta.get('status') != 'draft':
                assert 'coverage' in meta, f"{bundle_name}: Missing coverage stats"

            # Check entropy_triggers
            assert 'entropy_triggers' in bundle, f"{bundle_name}: Missing entropy_triggers"

            print(f"  ✅ {bundle_name}: All metadata fields present")

        except Exception as e:
            print(f"  ❌ {bundle_name}: {e}")
            all_valid = False

    return all_valid


def test_spanish_healthcare_mode():
    """Test that Spanish healthcare mode works in PRRGate."""
    print("Testing Spanish healthcare mode in PRRGate...")

    try:
        # Load Spanish core only
        kw_core, st_core = load_language_patterns('es', healthcare_mode=False)
        count_core = len(kw_core) + len(st_core)

        # Load Spanish with healthcare
        kw_hc, st_hc = load_language_patterns('es', healthcare_mode=True)
        count_hc = len(kw_hc) + len(st_hc)

        print(f"  ✅ Spanish core: {count_core} patterns")
        print(f"  ✅ Spanish + healthcare: {count_hc} patterns")
        assert count_hc > count_core, "Healthcare mode should add patterns"

        # Test that PRRGate initializes with Spanish healthcare mode
        try:
            gate = PRRGate(language='es', healthcare_mode=True, auto_detect_language=False)
            print(f"  ✅ PRRGate initialized with Spanish healthcare mode")
        except Exception as e:
            print(f"  ❌ PRRGate initialization failed: {e}")
            return False

        # Verify patterns are loaded in gate
        assert len(gate.keyword_re) > 0, "No keyword patterns loaded"
        assert len(gate.structure_re) > 0, "No structure patterns loaded"
        print(f"  ✅ PRRGate loaded {len(gate.keyword_re)} keyword + {len(gate.structure_re)} structure patterns")

        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stub_bundles_exist():
    """Test that stub bundles for fr, de, pt exist."""
    print("Testing stub bundles...")

    stub_langs = ['fr', 'de', 'pt']
    all_exist = True

    for lang in stub_langs:
        try:
            bundle = _load_bundle(f'{lang}_core')
            meta = bundle['metadata']

            assert meta['status'] == 'draft', f"{lang}: Should be draft status"
            assert '_contribution_guide' in bundle, f"{lang}: Missing contribution guide"

            print(f"  ✅ {lang}_core.json exists with contribution guide")

        except Exception as e:
            print(f"  ❌ {lang}_core.json: {e}")
            all_exist = False

    return all_exist


def test_mixed_language_mode():
    """Test that mixed language mode includes both English and Spanish healthcare."""
    print("Testing mixed language mode...")

    try:
        kw, st = load_language_patterns('mixed', healthcare_mode=True)
        total = len(kw) + len(st)

        print(f"  ✅ Mixed mode (en+es+healthcare): {total} patterns")

        # Should be more than English alone
        kw_en, st_en = load_language_patterns('en', healthcare_mode=True)
        total_en = len(kw_en) + len(st_en)

        assert total > total_en, "Mixed mode should have more patterns than English alone"

        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pattern Bundle Improvements - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Spanish healthcare bundle", test_spanish_healthcare_bundle),
        ("English core expansion", test_english_core_expansion),
        ("Standardized metadata", test_standardized_metadata),
        ("Spanish healthcare mode", test_spanish_healthcare_mode),
        ("Stub bundles exist", test_stub_bundles_exist),
        ("Mixed language mode", test_mixed_language_mode),
    ]

    results = []
    for name, test_func in tests:
        print()
        result = test_func()
        results.append((name, result))

    # Summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
