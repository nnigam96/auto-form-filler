"""
LLM Vision research module.

Uses Ollama with llama3.2-vision model for OCR extraction.
Gracefully skips if Ollama is not available.

Returns standardized results for benchmarking.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# LLM Vision configs (could test different models)
LLM_VISION_CONFIGS = [
    {"model": "llama3.2-vision"},
]


def extract_with_llm_vision(
    image_path: Path,
    model: str = "llama3.2-vision",
) -> Dict[str, Any]:
    """
    Extract passport data using LLM Vision (Ollama).

    Args:
        image_path: Path to passport image
        model: Ollama model to use

    Returns:
        Standardized result dict for benchmarking
    """
    start_time = time.time()
    method_name = f"llm_vision_{model.replace(':', '_')}"

    result = {
        "method": method_name,
        "success": False,
        "parsed": None,
        "checksum_valid": False,  # LLM doesn't do checksum validation
        "confidence": 0.0,
        "latency_ms": 0.0,
        "memory_mb": None,
        "fraud_flags": [],
        "error": None,
        "ollama_available": False,
    }

    try:
        from app.extraction.llm_vision import (
            check_ollama_available,
            extract_with_ollama,
        )

        # Check if Ollama is running
        if not check_ollama_available():
            result["error"] = "Ollama not available (run 'ollama serve')"
            result["ollama_available"] = False
            result["latency_ms"] = (time.time() - start_time) * 1000
            logger.warning("Skipping LLM Vision benchmark: Ollama not available")
            return result

        result["ollama_available"] = True

        # Run extraction
        passport_data = extract_with_ollama(Path(image_path))

        if passport_data:
            result["success"] = True
            result["parsed"] = {
                "surname": passport_data.surname,
                "given_names": passport_data.given_names,
                "passport_number": passport_data.passport_number,
                "nationality": passport_data.nationality,
                "date_of_birth": (
                    passport_data.date_of_birth.strftime("%y%m%d")
                    if passport_data.date_of_birth
                    else ""
                ),
                "sex": passport_data.sex.value if hasattr(passport_data.sex, 'value') else str(passport_data.sex),
                "expiry_date": (
                    passport_data.expiry_date.strftime("%y%m%d")
                    if passport_data.expiry_date
                    else ""
                ),
                "country_of_issue": passport_data.country_of_issue or "",
            }
            result["confidence"] = passport_data.confidence_score or 0.9
        else:
            result["error"] = "LLM Vision returned no data"

    except ImportError as e:
        result["error"] = f"Import error: {e}"
    except Exception as e:
        result["error"] = str(e)
        logger.debug(f"LLM Vision extraction failed: {e}")

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


def get_all_configs() -> List[Dict[str, Any]]:
    """Return all LLM Vision configurations to benchmark."""
    return LLM_VISION_CONFIGS.copy()


def is_available() -> bool:
    """Check if LLM Vision (Ollama) is available."""
    try:
        from app.extraction.llm_vision import check_ollama_available
        return check_ollama_available()
    except:
        return False
