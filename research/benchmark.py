"""
OCR Benchmark System.

Runs all OCR methods against test images, compares to ground truth,
and outputs results in multiple formats.

Usage:
    from app.extraction.research.benchmark import run_benchmark
    results = run_benchmark()
"""

import json
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Directory paths
RESEARCH_DIR = Path(__file__).parent
TEST_DATA_DIR = RESEARCH_DIR / "test_data"
GROUND_TRUTH_PATH = TEST_DATA_DIR / "ground_truth.json"


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single method on a single image."""
    image: str
    method: str
    success: bool
    latency_ms: float
    memory_mb: Optional[float]
    checksum_valid: bool
    confidence: float
    fraud_flags: List[str]
    error: Optional[str]

    # Comparison to ground truth
    exact_match_score: float = 0.0
    fuzzy_match_score: float = 0.0
    overall_score: float = 0.0
    field_results: Dict[str, bool] = field(default_factory=dict)

    # Human review detection
    expected_human_review: bool = False
    triggered_human_review: bool = False
    human_review_correct: bool = False


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    timestamp: str
    total_images: int
    total_methods: int
    total_runs: int

    # Per-method stats
    method_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Best config
    best_method: Optional[str] = None
    best_score: float = 0.0
    best_latency_ms: float = 0.0


def load_ground_truth() -> Dict[str, Any]:
    """Load ground truth data from JSON file."""
    if not GROUND_TRUTH_PATH.exists():
        logger.warning(f"Ground truth file not found: {GROUND_TRUTH_PATH}")
        return {"images": {}}

    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def get_test_images() -> List[Path]:
    """Get list of test images from test_data directory."""
    if not TEST_DATA_DIR.exists():
        return []

    image_extensions = {".png", ".jpg", ".jpeg", ".pdf"}
    images = []

    for f in TEST_DATA_DIR.iterdir():
        if f.suffix.lower() in image_extensions:
            images.append(f)

    return sorted(images)


def compare_to_ground_truth(
    parsed: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
) -> tuple:
    """
    Compare parsed result to ground truth.

    Returns:
        (exact_match_score, fuzzy_match_score, field_results)
    """
    if not parsed or not expected:
        return 0.0, 0.0, {}

    # Fields to compare
    exact_fields = ["passport_number", "sex", "nationality"]
    fuzzy_fields = ["surname", "given_names"]
    date_fields = ["date_of_birth", "expiry_date"]

    field_results = {}
    exact_matches = 0
    exact_total = 0
    fuzzy_score = 0.0
    fuzzy_total = 0

    # Exact match fields
    for field in exact_fields:
        expected_val = expected.get(field)
        if expected_val is None:
            continue
        exact_total += 1

        parsed_val = parsed.get(field, "")
        if isinstance(parsed_val, str) and isinstance(expected_val, str):
            match = parsed_val.upper().strip() == expected_val.upper().strip()
        else:
            match = parsed_val == expected_val

        field_results[field] = match
        if match:
            exact_matches += 1

    # Date fields (special handling for MRZ format)
    for field in date_fields:
        expected_val = expected.get(field)
        if expected_val is None:
            continue
        exact_total += 1

        parsed_val = parsed.get(field, "")
        match = _compare_dates(parsed_val, expected_val)
        field_results[field] = match
        if match:
            exact_matches += 1

    # Fuzzy match fields (names)
    for field in fuzzy_fields:
        expected_val = expected.get(field)
        if expected_val is None:
            continue
        fuzzy_total += 1

        parsed_val = parsed.get(field, "")
        similarity = _fuzzy_compare(parsed_val, expected_val)
        field_results[field] = similarity >= 0.8
        fuzzy_score += similarity

    exact_match_score = exact_matches / exact_total if exact_total > 0 else 0.0
    fuzzy_match_score = fuzzy_score / fuzzy_total if fuzzy_total > 0 else 0.0

    return exact_match_score, fuzzy_match_score, field_results


def _compare_dates(parsed: str, expected: str) -> bool:
    """Compare date values, handling MRZ format (YYMMDD) and ISO format."""
    if not parsed or not expected:
        return False

    # If expected is ISO format (YYYY-MM-DD), convert to YYMMDD
    if "-" in expected:
        try:
            parts = expected.split("-")
            expected = f"{parts[0][2:]}{parts[1]}{parts[2]}"
        except:
            pass

    # Clean parsed value
    parsed = parsed.replace("-", "").replace("/", "")[:6]
    expected = expected.replace("-", "").replace("/", "")[:6]

    return parsed == expected


def _fuzzy_compare(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using Levenshtein distance."""
    if not s1 or not s2:
        return 0.0

    s1 = s1.upper().strip()
    s2 = s2.upper().strip()

    if s1 == s2:
        return 1.0

    # Simple Levenshtein ratio
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Create distance matrix
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def run_single_method(
    image_path: Path,
    method_name: str,
    config: Dict[str, Any],
    track_memory: bool = False,
) -> Dict[str, Any]:
    """Run a single OCR method with optional memory tracking."""
    memory_mb = None

    if track_memory:
        import tracemalloc
        tracemalloc.start()

    try:
        if method_name == "tesseract":
            from app.extraction.research.ocr_tesseract import extract_with_tesseract
            result = extract_with_tesseract(
                image_path,
                psm=config.get("psm", 6),
                preprocess=config.get("preprocess", "otsu"),
            )
        elif method_name == "easyocr":
            from app.extraction.research.ocr_easyocr import extract_with_easyocr
            result = extract_with_easyocr(
                image_path,
                preprocess=config.get("preprocess", "none"),
            )
        elif method_name == "passporteye":
            from app.extraction.research.ocr_passporteye import extract_with_passporteye
            result = extract_with_passporteye(image_path)
        elif method_name == "ensemble":
            from app.extraction.research.ocr_ensemble import extract_with_ensemble
            result = extract_with_ensemble(image_path)
        elif method_name == "llm_vision":
            from app.extraction.research.ocr_llm_vision import extract_with_llm_vision
            result = extract_with_llm_vision(
                image_path,
                model=config.get("model", "llama3.2-vision"),
            )
        else:
            result = {"error": f"Unknown method: {method_name}"}
    finally:
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = peak / 1024 / 1024

    if memory_mb is not None:
        result["memory_mb"] = memory_mb

    return result


def run_benchmark(
    images: Optional[List[Path]] = None,
    methods: Optional[List[str]] = None,
    track_memory: bool = False,
) -> tuple:
    """
    Run full benchmark.

    Args:
        images: List of image paths (default: all in test_data)
        methods: List of methods to test (default: all)
        track_memory: Whether to track memory usage

    Returns:
        (results: List[BenchmarkResult], summary: BenchmarkSummary)
    """
    # Load ground truth
    ground_truth = load_ground_truth()

    # Get images
    if images is None:
        images = get_test_images()

    if not images:
        logger.warning("No test images found")
        return [], BenchmarkSummary(
            timestamp=datetime.now().isoformat(),
            total_images=0,
            total_methods=0,
            total_runs=0,
        )

    # Get method configs
    all_configs = _get_all_method_configs()

    if methods:
        all_configs = {k: v for k, v in all_configs.items() if k in methods}

    results: List[BenchmarkResult] = []
    method_stats: Dict[str, Dict[str, Any]] = {}

    total_runs = 0

    for image_path in images:
        image_name = image_path.name
        image_truth = ground_truth.get("images", {}).get(image_name, {})
        expected = image_truth.get("expected", {})
        expects_human_review = image_truth.get("expects_human_review", False)

        logger.info(f"Benchmarking: {image_name}")

        for method_name, configs in all_configs.items():
            for config in configs:
                total_runs += 1
                config_name = config.get("name", str(config))
                full_method = f"{method_name}_{config_name}" if config_name != "default" else method_name

                # Run method
                result = run_single_method(image_path, method_name, config, track_memory)

                # Compare to ground truth
                exact_score, fuzzy_score, field_results = compare_to_ground_truth(
                    result.get("parsed"),
                    expected,
                )
                overall_score = (exact_score * 0.7) + (fuzzy_score * 0.3)

                # Check human review detection
                triggered_review = len(result.get("fraud_flags", [])) > 0
                human_review_correct = triggered_review == expects_human_review

                # Create result object
                benchmark_result = BenchmarkResult(
                    image=image_name,
                    method=result.get("method", full_method),
                    success=result.get("success", False),
                    latency_ms=result.get("latency_ms", 0),
                    memory_mb=result.get("memory_mb"),
                    checksum_valid=result.get("checksum_valid", False),
                    confidence=result.get("confidence", 0),
                    fraud_flags=result.get("fraud_flags", []),
                    error=result.get("error"),
                    exact_match_score=exact_score,
                    fuzzy_match_score=fuzzy_score,
                    overall_score=overall_score,
                    field_results=field_results,
                    expected_human_review=expects_human_review,
                    triggered_human_review=triggered_review,
                    human_review_correct=human_review_correct,
                )
                results.append(benchmark_result)

                # Update method stats
                method_key = result.get("method", full_method)
                if method_key not in method_stats:
                    method_stats[method_key] = {
                        "runs": 0,
                        "successes": 0,
                        "total_score": 0.0,
                        "total_latency": 0.0,
                        "checksum_passes": 0,
                    }

                stats = method_stats[method_key]
                stats["runs"] += 1
                if benchmark_result.success:
                    stats["successes"] += 1
                stats["total_score"] += overall_score
                stats["total_latency"] += benchmark_result.latency_ms
                if benchmark_result.checksum_valid:
                    stats["checksum_passes"] += 1

    # Calculate averages and find best
    best_method = None
    best_score = 0.0
    best_latency = float("inf")

    for method, stats in method_stats.items():
        if stats["runs"] > 0:
            stats["avg_score"] = stats["total_score"] / stats["runs"]
            stats["avg_latency"] = stats["total_latency"] / stats["runs"]
            stats["success_rate"] = stats["successes"] / stats["runs"]

            # Best = highest score, then fastest
            if stats["avg_score"] > best_score or (
                stats["avg_score"] == best_score and stats["avg_latency"] < best_latency
            ):
                best_method = method
                best_score = stats["avg_score"]
                best_latency = stats["avg_latency"]

    summary = BenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        total_images=len(images),
        total_methods=len(all_configs),
        total_runs=total_runs,
        method_stats=method_stats,
        best_method=best_method,
        best_score=best_score,
        best_latency_ms=best_latency,
    )

    return results, summary


def _get_all_method_configs() -> Dict[str, List[Dict[str, Any]]]:
    """Get all method configurations."""
    from app.extraction.research.ocr_tesseract import get_all_configs as get_tesseract_configs
    from app.extraction.research.ocr_easyocr import get_all_configs as get_easyocr_configs
    from app.extraction.research.ocr_passporteye import get_all_configs as get_passporteye_configs
    from app.extraction.research.ocr_ensemble import get_all_configs as get_ensemble_configs
    from app.extraction.research.ocr_llm_vision import get_all_configs as get_llm_configs

    return {
        "tesseract": get_tesseract_configs(),
        "easyocr": get_easyocr_configs(),
        "passporteye": get_passporteye_configs(),
        "ensemble": get_ensemble_configs(),
        "llm_vision": get_llm_configs(),
    }


# =============================================================================
# Output Formatters
# =============================================================================

def output_console(results: List[BenchmarkResult], summary: BenchmarkSummary):
    """Print results to console in table format."""
    print("\n" + "=" * 80)
    print("OCR BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Timestamp: {summary.timestamp}")
    print(f"Images: {summary.total_images} | Methods: {summary.total_methods} | Runs: {summary.total_runs}")
    print()

    # Method summary table
    print("METHOD SUMMARY:")
    print("-" * 80)
    print(f"{'Method':<35} {'Success':<10} {'Avg Score':<12} {'Avg Latency':<12} {'Checksum'}")
    print("-" * 80)

    for method, stats in sorted(summary.method_stats.items()):
        success_rate = f"{stats.get('success_rate', 0)*100:.0f}%"
        avg_score = f"{stats.get('avg_score', 0)*100:.1f}%"
        avg_latency = f"{stats.get('avg_latency', 0):.0f}ms"
        checksum = f"{stats.get('checksum_passes', 0)}/{stats.get('runs', 0)}"
        print(f"{method:<35} {success_rate:<10} {avg_score:<12} {avg_latency:<12} {checksum}")

    print("-" * 80)
    print(f"BEST METHOD: {summary.best_method} (score: {summary.best_score*100:.1f}%, latency: {summary.best_latency_ms:.0f}ms)")
    print("=" * 80)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 80)

    for r in results:
        status = "✓" if r.success else "✗"
        checksum = "✓" if r.checksum_valid else "✗"
        review = "⚠" if r.triggered_human_review else ""

        print(f"{status} {r.image} | {r.method}")
        print(f"   Score: {r.overall_score*100:.1f}% | Latency: {r.latency_ms:.0f}ms | Checksum: {checksum} {review}")
        if r.error:
            print(f"   Error: {r.error}")
        print()


def output_json(results: List[BenchmarkResult], summary: BenchmarkSummary, path: Path):
    """Write results to JSON file."""
    output = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results written to: {path}")


def output_csv(results: List[BenchmarkResult], path: Path):
    """Write results to CSV file."""
    if not results:
        return

    fieldnames = [
        "image", "method", "success", "latency_ms", "memory_mb",
        "checksum_valid", "confidence", "exact_match_score", "fuzzy_match_score",
        "overall_score", "triggered_human_review", "error"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: getattr(r, k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"CSV written to: {path}")


def export_best_config(summary: BenchmarkSummary, path: Optional[Path] = None):
    """Export best configuration to JSON file."""
    if path is None:
        path = RESEARCH_DIR / "best_config.json"

    if not summary.best_method:
        logger.warning("No best method found")
        return

    # Parse method name to get config
    parts = summary.best_method.split("_")
    method = parts[0]

    config = {
        "method": method,
        "full_name": summary.best_method,
        "score": summary.best_score,
        "latency_ms": summary.best_latency_ms,
        "timestamp": summary.timestamp,
    }

    # Add method-specific config
    if method == "tesseract" and len(parts) >= 3:
        config["psm"] = int(parts[1].replace("psm", ""))
        config["preprocess"] = parts[2]
    elif method == "easyocr" and len(parts) >= 2:
        config["preprocess"] = parts[1]

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Best config exported to: {path}")
