"""
Ensemble OCR research module.

Runs multiple OCR methods in parallel and uses voting logic
to select the best result. This wraps the agentic tools.py implementation.

Returns standardized results for benchmarking.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Ensemble has one config - it runs all methods
ENSEMBLE_CONFIGS = [
    {"name": "parallel_voting"},
]


def extract_with_ensemble(
    image_path: Path,
) -> Dict[str, Any]:
    """
    Extract MRZ using ensemble of OCR methods with voting.

    Runs PassportEye, Tesseract, and EasyOCR in parallel,
    then uses voting logic to pick the best result.

    Args:
        image_path: Path to passport image

    Returns:
        Standardized result dict for benchmarking
    """
    start_time = time.time()
    method_name = "ensemble_voting"

    result = {
        "method": method_name,
        "success": False,
        "parsed": None,
        "checksum_valid": False,
        "confidence": 0.0,
        "latency_ms": 0.0,
        "memory_mb": None,
        "fraud_flags": [],
        "error": None,
        "winning_source": None,
        "all_results": [],
    }

    try:
        # Run the async ensemble
        ensemble_result = asyncio.run(_run_ensemble(str(image_path)))

        result["all_results"] = ensemble_result["ocr_results"]
        result["winning_source"] = ensemble_result["source"]
        result["fraud_flags"] = ensemble_result.get("fraud_flags", [])

        if ensemble_result["final_data"]:
            result["success"] = True
            result["parsed"] = ensemble_result["final_data"]
            result["checksum_valid"] = ensemble_result.get("checksum_valid", False)
            result["confidence"] = ensemble_result["confidence"]
        else:
            result["error"] = "No valid result from ensemble"
            if ensemble_result.get("errors"):
                result["error"] = "; ".join(ensemble_result["errors"])

    except Exception as e:
        result["error"] = str(e)
        logger.debug(f"Ensemble extraction failed: {e}")

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


async def _run_ensemble(image_path: str) -> Dict[str, Any]:
    """Run all OCR tools in parallel and vote on results."""
    from app.extraction.ocr_engines import (
        run_passport_eye,
        run_tesseract,
        run_easyocr,
    )
    from app.extraction.voting import vote_on_results, critic_validate

    # Run all tools concurrently
    results = await asyncio.gather(
        run_passport_eye(image_path),
        run_tesseract(image_path),
        run_easyocr(image_path),
        return_exceptions=True,
    )

    # Handle exceptions
    ocr_results = []
    errors = []
    for r in results:
        if isinstance(r, Exception):
            ocr_results.append({
                "source": "unknown",
                "success": False,
                "error": str(r),
            })
            errors.append(str(r))
        else:
            ocr_results.append(r)

    # Vote on results
    best_data, confidence, source = vote_on_results(ocr_results)

    # Run critic validation
    fraud_flags = []
    if best_data:
        is_valid, fraud_flags = critic_validate(best_data)

    # Determine if checksum was valid
    checksum_valid = False
    for r in ocr_results:
        if r.get("source") == source and r.get("checksum_valid"):
            checksum_valid = True
            break

    return {
        "ocr_results": ocr_results,
        "final_data": best_data,
        "confidence": confidence,
        "source": source,
        "checksum_valid": checksum_valid,
        "fraud_flags": fraud_flags,
        "errors": errors,
    }


def get_all_configs() -> List[Dict[str, Any]]:
    """Return all Ensemble configurations to benchmark."""
    return ENSEMBLE_CONFIGS.copy()
