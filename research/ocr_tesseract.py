"""
Tesseract OCR research module.

Experiments with different PSM modes and preprocessing methods.
Returns standardized results for benchmarking.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import pytesseract
from PIL import Image

from app.extraction.research.ocr_preprocessing import PREPROCESSING_METHODS

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

# Configurations to benchmark
TESSERACT_CONFIGS = [
    {"psm": 6, "preprocess": "none"},
    {"psm": 6, "preprocess": "otsu"},
    {"psm": 6, "preprocess": "adaptive"},
    {"psm": 11, "preprocess": "otsu"},
    {"psm": 11, "preprocess": "adaptive"},
    {"psm": 3, "preprocess": "otsu"},
]


def extract_with_tesseract(
    image_path: Path,
    psm: int = 6,
    preprocess: str = "otsu",
    rotations: List[int] = [0, 90, 180, 270],
) -> Dict[str, Any]:
    """
    Extract MRZ using Tesseract with specified configuration.

    Args:
        image_path: Path to passport image
        psm: Tesseract PSM mode (3=auto, 6=block, 11=sparse)
        preprocess: Preprocessing method from ocr_preprocessing
        rotations: List of rotation angles to try

    Returns:
        Standardized result dict for benchmarking
    """
    start_time = time.time()
    method_name = f"tesseract_psm{psm}_{preprocess}"

    result = {
        "method": method_name,
        "success": False,
        "parsed": None,
        "checksum_valid": False,
        "confidence": 0.0,
        "latency_ms": 0.0,
        "memory_mb": None,  # Set by benchmark if --memory flag
        "fraud_flags": [],
        "error": None,
        "rotation_used": None,
        "raw_text": None,
        "page_used": None,
    }

    try:
        # Get all pages if PDF, or single image
        image_paths = _get_image_paths(image_path)
        if not image_paths:
            result["error"] = f"Could not convert PDF to image or invalid file type: {image_path}"
            return result

        # Try each page until we find MRZ
        for page_path in image_paths:
            img = cv2.imread(str(page_path))
            if img is None:
                continue

            for rotation in rotations:
                # Rotate image
                if rotation == 90:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    rotated = img

                # Preprocess
                if preprocess == "none":
                    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                    processed = Image.fromarray(gray)
                else:
                    preprocess_func = PREPROCESSING_METHODS.get(preprocess)
                    if preprocess_func:
                        # Save temp rotated image for preprocessing
                        temp_path = Path(f"/tmp/ocr_temp_{rotation}.png")
                        cv2.imwrite(str(temp_path), rotated)
                        processed = preprocess_func(temp_path)
                    else:
                        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                        processed = Image.fromarray(gray)

                # Run Tesseract
                config = f'--oem 3 --psm {psm}'
                text = pytesseract.image_to_string(processed, config=config)

                # Try to parse MRZ
                parsed, checksum_valid = _parse_mrz_from_text(text)

                if parsed:
                    result["success"] = True
                    result["parsed"] = parsed
                    result["checksum_valid"] = checksum_valid
                    result["confidence"] = 1.0 if checksum_valid else 0.5
                    result["rotation_used"] = rotation
                    result["raw_text"] = text
                    result["page_used"] = image_paths.index(page_path) + 1 if len(image_paths) > 1 else None

                    # If checksum valid, we're done
                    if checksum_valid:
                        break
            
            # If we found a result (even without valid checksum), stop trying pages
            if result["success"]:
                break

    except Exception as e:
        result["error"] = str(e)
        logger.debug(f"Tesseract extraction failed: {e}")

    result["latency_ms"] = (time.time() - start_time) * 1000
    return result


def get_all_configs() -> List[Dict[str, Any]]:
    """Return all Tesseract configurations to benchmark."""
    return TESSERACT_CONFIGS.copy()


def _parse_mrz_from_text(text: str) -> tuple:
    """
    Parse MRZ lines from OCR text.
    Returns (parsed_dict, checksum_valid) or (None, False).
    """
    text = text.upper()
    lines = text.split("\n")

    mrz_candidates = []

    for line in lines:
        line = line.strip().replace(" ", "").replace("«", "<")
        if len(line) >= 40:
            mrz_chars = sum(1 for c in line if c.isalnum() or c == "<")
            if mrz_chars / len(line) > 0.9:
                mrz_candidates.append(line[:44])

    # Find line1 (starts with P) and line2
    for i, line1 in enumerate(mrz_candidates):
        if not line1.startswith("P"):
            continue

        for line2 in mrz_candidates[i + 1 : i + 3]:
            if len(line2) >= 44:
                dob_section = line2[13:19]
                exp_section = line2[21:27]
                digit_count = sum(c.isdigit() for c in dob_section + exp_section)

                if digit_count >= 8:
                    parsed = _extract_fields_from_mrz(line1, line2)
                    checksum_valid = _validate_checksum(line2)
                    return parsed, checksum_valid

    return None, False


def _extract_fields_from_mrz(line1: str, line2: str) -> dict:
    """Extract passport fields from MRZ lines."""
    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]

    names = line1[5:44].split("<<", 1)
    surname = names[0].replace("<", "")
    given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""

    return {
        "surname": surname,
        "given_names": given_names,
        "passport_number": line2[0:9].replace("<", ""),
        "nationality": line2[10:13],
        "date_of_birth": line2[13:19],
        "sex": line2[20],
        "expiry_date": line2[21:27],
        "country_of_issue": line1[2:5],
    }


def _validate_checksum(line2: str) -> bool:
    """Validate MRZ line2 checksums."""
    if len(line2) < 44:
        return False

    def calc_check(data: str) -> int:
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(data):
            if char == "<":
                value = 0
            elif char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord("A") + 10
            else:
                value = 0
            total += value * weights[i % 3]
        return total % 10

    try:
        passport_check = int(line2[9]) if line2[9].isdigit() else -1
        dob_check = int(line2[19]) if line2[19].isdigit() else -1
        expiry_check = int(line2[27]) if line2[27].isdigit() else -1

        checks_passed = 0
        if calc_check(line2[0:9]) == passport_check:
            checks_passed += 1
        if calc_check(line2[13:19]) == dob_check:
            checks_passed += 1
        if calc_check(line2[21:27]) == expiry_check:
            checks_passed += 1

        return checks_passed >= 2
    except:
        return False
