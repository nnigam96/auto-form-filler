#!/usr/bin/env python3
"""
Run OCR evaluation to find the best configuration.

Usage:
    python -m app.extraction.research.run_eval
"""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.models.schemas import Sex
from app.extraction.research.ocr_eval import find_best_ocr_config, evaluate_all_configs

# Ground truth for evaluation
EXPECTED_PASSPORT = {
    "surname": "Nigam",
    "given_names": "Nikhil Rajesh",
    "passport_number": "N7178292",
    "nationality": "India",
    "date_of_birth": date(1996, 10, 25),
    "sex": Sex.MALE,
    "expiry_date": date(2026, 2, 15),
    "country_of_issue": "India",
}


def main():
    """Run OCR evaluation."""
    # Path to test passport
    passport_path = project_root / "docs" / "local" / "Passport Front.pdf"
    
    if not passport_path.exists():
        print(f"Error: Test passport not found at {passport_path}")
        print("Please ensure the test passport is available.")
        return
    
    print("Running OCR evaluation...")
    print(f"Test passport: {passport_path}")
    print()
    
    # Run full evaluation
    results = evaluate_all_configs(passport_path, EXPECTED_PASSPORT)
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total configs tested: {results['total_configs_tested']}")
    print(f"Successful configs: {results['successful_configs']}")
    print(f"Best method: {results['best_method']}")
    print()
    
    print("Summary by method:")
    for method, stats in results['summary']['by_method'].items():
        print(f"  {method}:")
        print(f"    Total: {stats['total']}")
        print(f"    Successful: {stats['successful']}")
        print(f"    Failed: {stats['failed']}")
        print(f"    Avg time: {stats.get('avg_time_ms', 0):.1f}ms")
    print()
    
    if results['successful_configs'] > 0:
        print("Successful configurations:")
        for config in results['successful_configs_detail']:
            print(f"  - {config['method']}: {config['config']} ({config['time_ms']:.1f}ms)")
        print()
        
        print("Best configuration:")
        best = find_best_ocr_config(passport_path, EXPECTED_PASSPORT)
        if best:
            print(f"  Method: {best['method']}")
            print(f"  Config: {best['config']}")
            print(f"  Full name: {best['full_method_name']}")
            print()
            print("To use this in production, update ocr_production.py:")
            print(f"  BEST_OCR_CONFIG = {best}")
    else:
        print("WARNING: No OCR configurations succeeded with exact match!")
        print("However, some may have extracted data with OCR errors.")
        print("Check the detailed metrics below to see what was extracted.")
        print()
        
        # Find configs that at least found MRZ (checksum passed)
        checksum_passed = [m for m in results['all_metrics'] if m.get('checksum_passed', False)]
        if checksum_passed:
            print(f"Found {len(checksum_passed)} configs that extracted MRZ (but with OCR errors):")
            for m in checksum_passed[:3]:  # Show first 3
                print(f"  - {m['method']}: {m['time_ms']:.1f}ms")
            print()
    
    print("=" * 60)
    print("DETAILED METRICS")
    print("=" * 60)
    for metric in results['all_metrics']:
        status = "✓" if metric['success'] else "✗"
        checksum_status = "✓" if metric.get('checksum_passed', False) else "✗"
        print(f"{status} {metric['method']}: {metric['time_ms']:.1f}ms (checksum: {checksum_status})")
        
        # Show extracted data if available
        if '_extracted' in metric.get('config', {}):
            extracted = metric['config']['_extracted']
            print(f"    Extracted: Passport={extracted.get('passport_number', 'N/A')}, "
                  f"DOB={extracted.get('date_of_birth', 'N/A')}, "
                  f"Expiry={extracted.get('expiry_date', 'N/A')}")
            print(f"    Expected:  Passport={EXPECTED_PASSPORT['passport_number']}, "
                  f"DOB={EXPECTED_PASSPORT['date_of_birth']}, "
                  f"Expiry={EXPECTED_PASSPORT['expiry_date']}")
        
        if metric.get('error'):
            print(f"    Error: {metric['error']}")
        print()


if __name__ == "__main__":
    main()

