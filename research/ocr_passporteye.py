"""
PassportEye research module.

Uses the passporteye library for MRZ detection.
Returns standardized results for benchmarking.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _get_image_paths(file_path: Path) -> List[Path]:
    """Convert PDF to images if needed, return list of image paths."""
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return [file_path]
    
    if file_path.suffix.lower() == '.pdf':
        try:
            from app.utils.pdf_utils import pdf_to_images
            images = pdf_to_images(file_path, dpi=300)
            return images if images else []
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return []
    
    return []

# PassportEye has limited configuration options
PASSPORTEYE_CONFIGS = [
    {"name": "default"},
]


def extract_with_passporteye(
    image_path: Path,
) -> Dict[str, Any]:
    """
    Extract MRZ using PassportEye library.

    Args:
        image_path: Path to passport image

    Returns:
        Standardized result dict for benchmarking
    """
    start_time = time.time()
    method_name = "passporteye"

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
        "valid_score": None,
        "raw_mrz": None,
        "page_used": None,
    }

    try:
        from passporteye import read_mrz

        # Get all pages if PDF, or single image
        image_paths = _get_image_paths(image_path)
        if not image_paths:
            result["error"] = f"Could not convert PDF to image or invalid file type: {image_path}"
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result

        # Try each page until we find MRZ
        for page_idx, page_path in enumerate(image_paths):
            mrz = read_mrz(str(page_path))

            if mrz is None:
                continue  # Try next page

            data = mrz.to_dict()
            result["raw_mrz"] = data
    
            # Extract validity info
            valid_score = data.get("valid_score", 0)
            valid_number = data.get("valid_number", False)
            valid_dob = data.get("valid_date_of_birth", False)
            valid_exp = data.get("valid_expiration_date", False)
    
            result["valid_score"] = valid_score
    
            # Checksum is valid if score >= 60 and at least 2 of 3 checks pass
            checks_passed = sum([valid_number, valid_dob, valid_exp])
            checksum_valid = valid_score >= 60 and checks_passed >= 2
    
            # Parse fields
            parsed = {
                "surname": data.get("surname", "").replace("<", ""),
                "given_names": data.get("names", "").replace("<", " ").strip(),
                "passport_number": data.get("number", "").replace("<", ""),
                "nationality": data.get("nationality", ""),
                "date_of_birth": data.get("date_of_birth", ""),
                "sex": data.get("sex", ""),
                "expiry_date": data.get("expiration_date", ""),
                "country_of_issue": data.get("country", ""),
            }
    
            # Validate parsed data looks reasonable
            if not _validate_parsed_data(parsed):
                result["error"] = "Parsed data failed validation"
                result["fraud_flags"].append("INVALID_PARSED_DATA")
                result["latency_ms"] = (time.time() - start_time) * 1000
                return result

            result["success"] = True
            result["parsed"] = parsed
            result["checksum_valid"] = checksum_valid
            result["confidence"] = (valid_score / 100.0) if valid_score else 0.5
            result["page_used"] = page_idx + 1 if len(image_paths) > 1 else None
            break  # Found MRZ, stop trying pages

        # If no MRZ found on any page
        if not result["success"]:
            result["error"] = "MRZ not detected on any page"

    except ImportError:
        result["error"] = "PassportEye not installed. Run: pip install passporteye"
    except Exception as e:
        result["error"] = str(e)
        logger.debug(f"PassportEye extraction failed: {e}")

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


def get_all_configs() -> List[Dict[str, Any]]:
    """Return all PassportEye configurations to benchmark."""
    return PASSPORTEYE_CONFIGS.copy()


def _validate_parsed_data(parsed: dict) -> bool:
    """Validate parsed data looks reasonable."""
    surname = parsed.get("surname", "")
    passport_num = parsed.get("passport_number", "")

    # Surname should be mostly letters
    if surname:
        alpha_ratio = sum(c.isalpha() for c in surname) / len(surname)
        if alpha_ratio < 0.8:
            return False

    # Passport number should be at least 6 chars
    if len(passport_num) < 6:
        return False

    # Sex should be valid
    sex = parsed.get("sex", "")
    if sex and sex not in ["M", "F", "<", "X"]:
        return False

    return True
