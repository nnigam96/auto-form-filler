#!/usr/bin/env python3
"""
CLI for running OCR benchmarks.

Usage:
    python -m app.extraction.research.run_benchmark [OPTIONS]

Examples:
    # Run all methods on all test images
    python -m app.extraction.research.run_benchmark

    # Run specific method
    python -m app.extraction.research.run_benchmark --method tesseract

    # Run on specific image
    python -m app.extraction.research.run_benchmark --image passport_valid.png

    # Enable memory profiling
    python -m app.extraction.research.run_benchmark --memory

    # Output to JSON
    python -m app.extraction.research.run_benchmark --output json

    # All outputs
    python -m app.extraction.research.run_benchmark --output all
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.extraction.research.benchmark import (
    run_benchmark,
    output_console,
    output_json,
    output_csv,
    export_best_config,
    get_test_images,
    TEST_DATA_DIR,
    RESEARCH_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR benchmark on test images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Specific image filename to test (in test_data/)",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["tesseract", "easyocr", "passporteye", "ensemble", "llm_vision"],
        help="Specific method to benchmark",
    )

    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory profiling (adds overhead)",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["console", "json", "csv", "all"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESEARCH_DIR),
        help="Directory for output files",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Don't export best_config.json",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get images
    if args.image:
        image_path = TEST_DATA_DIR / args.image
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            print(f"Available images: {[p.name for p in get_test_images()]}")
            sys.exit(1)
        images = [image_path]
    else:
        images = get_test_images()

    if not images:
        print("No test images found!")
        print(f"Add images to: {TEST_DATA_DIR}")
        print("And update ground_truth.json with expected values.")
        sys.exit(1)

    print(f"Found {len(images)} test image(s)")

    # Get methods
    methods = [args.method] if args.method else None

    # Check LLM Vision availability
    if methods is None or "llm_vision" in methods:
        try:
            from app.extraction.research.ocr_llm_vision import is_available
            if not is_available():
                print("Note: LLM Vision (Ollama) not available - will be skipped")
        except:
            pass

    # Run benchmark
    print("\nRunning benchmark...")
    print("-" * 40)

    results, summary = run_benchmark(
        images=images,
        methods=methods,
        track_memory=args.memory,
    )

    if not results:
        print("No results generated!")
        sys.exit(1)

    # Output results
    output_dir = Path(args.output_dir)

    if args.output in ["console", "all"]:
        output_console(results, summary)

    if args.output in ["json", "all"]:
        json_path = output_dir / "benchmark_results.json"
        output_json(results, summary, json_path)

    if args.output in ["csv", "all"]:
        csv_path = output_dir / "benchmark_results.csv"
        output_csv(results, csv_path)

    # Export best config
    if not args.no_export and summary.best_method:
        export_best_config(summary)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
