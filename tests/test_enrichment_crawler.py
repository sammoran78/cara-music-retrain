#!/usr/bin/env python3
"""
Test script for Freesound enrichment crawler.

This script tests the key functionality of the enrichment crawler
without making actual API calls.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the module using importlib to handle numeric filename
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enrichment_module", 
    PROJECT_ROOT / "data_pipeline" / "02a_freesound_enrich_attribution.py"
)
enrichment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrichment_module)

# Extract functions
extract_sound_id = enrichment_module.extract_sound_id
normalize_license = enrichment_module.normalize_license
license_family = enrichment_module.license_family
safe_json_dump = enrichment_module.safe_json_dump
OUTPUT_COLUMNS = enrichment_module.OUTPUT_COLUMNS


def test_extract_sound_id():
    """Test sound ID extraction from various row formats."""
    print("Testing sound ID extraction...")
    
    test_cases = [
        # Explicit ID columns
        ({"id": "123456"}, 123456),
        ({"sound_id": "789"}, 789),
        ({"freesound_id": "42"}, 42),
        ({"ID": "999"}, 999),
        
        # URL extraction
        ({"url": "https://freesound.org/sounds/123456/"}, 123456),
        ({"source_url": "http://freesound.org/sounds/789/download/"}, 789),
        ({"link": "freesound.org/sounds/42"}, 42),
        
        # Should NOT extract from arbitrary fields
        ({"name": "Sound 123"}, None),
        ({"description": "Contains 456 samples"}, None),
        ({"tags": "789bpm"}, None),
        
        # Invalid cases
        ({"id": "abc"}, None),
        ({"id": ""}, None),
        ({"url": "https://example.com/123"}, None),
        ({}, None),
    ]
    
    passed = 0
    failed = 0
    
    for row, expected in test_cases:
        result = extract_sound_id(row)
        if result == expected:
            passed += 1
            print(f"  ✓ {row} -> {result}")
        else:
            failed += 1
            print(f"  ✗ {row} -> {result} (expected {expected})")
    
    print(f"\nID extraction: {passed} passed, {failed} failed\n")
    return failed == 0


def test_normalize_license():
    """Test license normalization."""
    print("Testing license normalization...")
    
    test_cases = [
        # CC0 variants
        ("CC0", "cc0"),
        ("cc0", "cc0"),
        ("publicdomain", "cc0"),
        ("Public Domain", "cc0"),
        ("Creative Commons Zero", "cc0"),
        
        # CC-BY variants
        ("CC-BY", "cc-by"),
        ("cc by", "cc-by"),
        ("Creative Commons BY", "cc-by"),
        ("http://creativecommons.org/licenses/by/4.0/", "cc-by"),
        
        # CC-BY-NC variants
        ("CC-BY-NC", "cc-by-nc"),
        ("cc by-nc", "cc-by-nc"),
        ("Creative Commons BY-NC", "cc-by-nc"),
        
        # Sampling+
        ("Sampling+", "sampling+"),
        ("sampling plus", "sampling+"),
        
        # Unknown
        ("Proprietary", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ]
    
    passed = 0
    failed = 0
    
    for input_license, expected in test_cases:
        result = normalize_license(input_license)
        if result == expected:
            passed += 1
            print(f"  ✓ '{input_license}' -> '{result}'")
        else:
            failed += 1
            print(f"  ✗ '{input_license}' -> '{result}' (expected '{expected}')")
    
    print(f"\nLicense normalization: {passed} passed, {failed} failed\n")
    return failed == 0


def test_license_family():
    """Test license family categorization."""
    print("Testing license family categorization...")
    
    test_cases = [
        ("cc0", "cc0"),
        ("cc-by", "cc-by"),
        ("cc-by-nc", "cc-by-nc"),
        ("cc-by-nc-sa", "cc-by-nc"),
        ("cc-by-nc-nd", "cc-by-nc"),
        ("sampling+", "sampling+"),
        ("unknown", "unknown"),
        ("proprietary", "unknown"),
    ]
    
    passed = 0
    failed = 0
    
    for normalized, expected in test_cases:
        result = license_family(normalized)
        if result == expected:
            passed += 1
            print(f"  ✓ '{normalized}' -> '{result}'")
        else:
            failed += 1
            print(f"  ✗ '{normalized}' -> '{result}' (expected '{expected}')")
    
    print(f"\nLicense family: {passed} passed, {failed} failed\n")
    return failed == 0


def test_safe_json_dump():
    """Test safe JSON serialization."""
    print("Testing safe JSON serialization...")
    
    test_cases = [
        ({"key": "value"}, '{"key":"value"}'),
        ([1, 2, 3], '[1,2,3]'),
        (None, '{}'),  # Should handle None gracefully
        ({"unicode": "café"}, '{"unicode":"café"}'),
    ]
    
    passed = 0
    failed = 0
    
    for obj, expected in test_cases:
        try:
            result = safe_json_dump(obj)
            # For None case, we just check it doesn't crash
            if obj is None:
                passed += 1
                print(f"  ✓ {obj} -> handled gracefully")
            elif result == expected:
                passed += 1
                print(f"  ✓ {obj} -> {result}")
            else:
                failed += 1
                print(f"  ✗ {obj} -> {result} (expected {expected})")
        except Exception as e:
            failed += 1
            print(f"  ✗ {obj} -> Exception: {e}")
    
    print(f"\nJSON serialization: {passed} passed, {failed} failed\n")
    return failed == 0


def test_enrichment_output_columns():
    """Verify all expected output columns are defined."""
    print("Testing output column definitions...")
    
    required_columns = [
        # Provenance
        "attribution_row_index",
        "freesound_sound_id",
        "original_attribution_license",
        
        # Current state
        "api_status",
        "current_license",
        "current_duration",
        
        # Analysis
        "analysis_available",
        "genre_inferred",
        
        # Research
        "current_license_normalized",
        "current_license_family",
        "duration_ge_30s",
        "changed_from_original_attribution_license",
    ]
    
    missing = [col for col in required_columns if col not in OUTPUT_COLUMNS]
    
    if not missing:
        print(f"  ✓ All {len(required_columns)} required columns present")
        print(f"  ✓ Total columns defined: {len(OUTPUT_COLUMNS)}")
        return True
    else:
        print(f"  ✗ Missing columns: {missing}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Freesound Enrichment Crawler Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_extract_sound_id,
        test_normalize_license,
        test_license_family,
        test_safe_json_dump,
        test_enrichment_output_columns,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"Summary: {passed}/{total} test suites passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test suites failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
