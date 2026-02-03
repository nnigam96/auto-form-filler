"""
Simple OCR Service for Passport MRZ Extraction.

Takes a passport PDF/image and returns the MRZ lines.
Handles:
1. PDF to image conversion
2. Image rotation (0, 90, 180, 270 degrees)
3. Multiple OCR engines (Tesseract, EasyOCR)
4. Multiple preprocessing methods
5. MRZ line detection and validation

Usage:
    from app.extraction.ocr_service import extract_mrz

    mrz = extract_mrz("path/to/passport.pdf")
    if mrz:
        print(mrz.line1, mrz.line2)
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MRZResult:
    """Result of MRZ extraction."""
    line1: str
    line2: str
    method: str
    rotation: int
    confidence: float


def pdf_to_image(pdf_path: Path, dpi: int = 300) -> Optional[Path]:
    """Convert first page of PDF to image."""
    try:
        from pdf2image import convert_from_path
        import tempfile

        images = convert_from_path(str(pdf_path), dpi=dpi)
        if not images:
            return None

        # Save first page
        temp_dir = Path(tempfile.mkdtemp(prefix="ocr_"))
        image_path = temp_dir / "page.png"
        images[0].save(str(image_path), "PNG")
        logger.debug(f"Converted PDF to image: {image_path}")
        return image_path

    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        return None


def rotate_image(image_path: Path, angle: int) -> np.ndarray:
    """Rotate image by given angle (0, 90, 180, 270)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Invalid rotation angle: {angle}")


def preprocess_for_ocr(img: np.ndarray, method: str = "otsu") -> Image.Image:
    """Preprocess image for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "none":
        return Image.fromarray(gray)
    elif method == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(thresh)
    else:
        return Image.fromarray(gray)


def ocr_tesseract(img: Image.Image, psm: int = 6) -> str:
    """Run Tesseract OCR on image."""
    try:
        import pytesseract
        config = f'--oem 3 --psm {psm}'
        return pytesseract.image_to_string(img, config=config)
    except Exception as e:
        logger.debug(f"Tesseract failed: {e}")
        return ""


def ocr_easyocr(img: Image.Image) -> str:
    """Run EasyOCR on image."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        # Convert PIL to numpy
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=0)
        return "\n".join(results)
    except Exception as e:
        logger.debug(f"EasyOCR failed: {e}")
        return ""


def detect_mrz_lines(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect MRZ lines in OCR text.

    MRZ Format (TD3 - Passport):
    - Line 1: P<COUNTRY<<SURNAME<<GIVEN<NAMES... (44 chars)
    - Line 2: PASSPORT#<CHECK<NATIONALITY<DOB<CHECK<SEX<EXPIRY<CHECK... (44 chars)

    Returns (line1, line2) or None if not found.
    """
    # Clean text - normalize whitespace, uppercase
    text = text.upper()
    lines = text.split('\n')

    # Pattern for MRZ lines
    # Line 1 starts with P< (passport)
    # Line 2 is mostly alphanumeric with < as filler

    mrz_candidates = []

    for line in lines:
        # Clean the line
        line = line.strip()
        # Remove spaces (MRZ has no spaces)
        line = line.replace(' ', '')
        # Common OCR errors in MRZ
        line = line.replace('«', '<').replace('≪', '<').replace('K<', '<<')

        # MRZ lines are 44 characters
        if len(line) >= 40:  # Allow some tolerance
            # Check if it looks like MRZ (alphanumeric + <)
            mrz_chars = sum(1 for c in line if c.isalnum() or c == '<')
            if mrz_chars / len(line) > 0.9:  # 90% MRZ chars
                mrz_candidates.append(line[:44])  # Take first 44 chars

    # Find pairs that look like Line1 and Line2
    for i, line1 in enumerate(mrz_candidates):
        # Line 1 should start with P< (passport document)
        if not line1.startswith('P'):
            continue

        # Look for Line 2 (next candidate or within next few)
        for line2 in mrz_candidates[i+1:i+3]:
            # Line 2 should have passport number pattern
            # First 9 chars are passport number + check digit
            if len(line2) >= 44:
                # Validate line2 structure: has digits in expected positions
                dob_section = line2[13:19]
                exp_section = line2[21:27]

                # Should have mostly digits in date sections
                digit_count = sum(c.isdigit() for c in dob_section + exp_section)
                if digit_count >= 8:  # At least 8 of 12 chars are digits
                    logger.info(f"Found MRZ lines")
                    return (line1, line2)

    # Fallback: Look for any two consecutive 44-char lines
    for i in range(len(mrz_candidates) - 1):
        line1 = mrz_candidates[i]
        line2 = mrz_candidates[i + 1]

        # Basic validation
        if line1.startswith('P') and len(line2) >= 44:
            logger.info(f"Found MRZ lines (fallback)")
            return (line1, line2)

    return None


def validate_mrz_checksum(line2: str) -> bool:
    """
    Validate MRZ Line 2 using check digits.

    MRZ uses weights 7, 3, 1 repeating for checksum calculation.

    Check digit positions in Line 2:
    - Position 9: Check for passport number (0-8)
    - Position 19: Check for DOB (13-18)
    - Position 27: Check for expiry (21-26)
    - Position 43: Overall check
    """
    if len(line2) < 44:
        return False

    def calc_check(data: str) -> int:
        """Calculate MRZ check digit."""
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(data):
            if char == '<':
                value = 0
            elif char.isdigit():
                value = int(char)
            elif char.isalpha():
                value = ord(char.upper()) - ord('A') + 10
            else:
                value = 0
            total += value * weights[i % 3]
        return total % 10

    try:
        # Passport number check (positions 0-8, check at 9)
        passport_num = line2[0:9]
        passport_check = int(line2[9]) if line2[9].isdigit() else -1

        # DOB check (positions 13-18, check at 19)
        dob = line2[13:19]
        dob_check = int(line2[19]) if line2[19].isdigit() else -1

        # Expiry check (positions 21-26, check at 27)
        expiry = line2[21:27]
        expiry_check = int(line2[27]) if line2[27].isdigit() else -1

        # Validate each section
        checks_passed = 0

        if calc_check(passport_num) == passport_check:
            checks_passed += 1
        if calc_check(dob) == dob_check:
            checks_passed += 1
        if calc_check(expiry) == expiry_check:
            checks_passed += 1

        # Consider valid if at least 2 of 3 checks pass
        return checks_passed >= 2

    except Exception as e:
        logger.debug(f"Checksum validation error: {e}")
        return False


def extract_mrz(file_path: Path) -> Optional[MRZResult]:
    """
    Main entry point: Extract MRZ from passport file.

    Tries multiple strategies:
    1. Different rotations (0, 90, 180, 270)
    2. Different OCR engines (Tesseract, EasyOCR)
    3. Different preprocessing methods

    Returns MRZResult with the extracted lines, or None if failed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    # Convert PDF to image if needed
    if file_path.suffix.lower() == ".pdf":
        image_path = pdf_to_image(file_path)
        if not image_path:
            return None
    else:
        image_path = file_path

    # Strategies to try
    rotations = [0, 90, 180, 270]
    preprocess_methods = ["none", "otsu", "adaptive"]
    psm_modes = [6, 11, 3]  # 6=block, 11=sparse, 3=auto

    best_result = None
    best_score = 0

    # Try Tesseract first (faster)
    for rotation in rotations:
        try:
            img = rotate_image(image_path, rotation)
        except Exception as e:
            logger.debug(f"Rotation {rotation} failed: {e}")
            continue

        for preprocess in preprocess_methods:
            for psm in psm_modes:
                try:
                    processed = preprocess_for_ocr(img, preprocess)
                    text = ocr_tesseract(processed, psm)

                    if not text:
                        continue

                    mrz = detect_mrz_lines(text)
                    if mrz:
                        line1, line2 = mrz

                        # Score based on checksum validation
                        checksum_valid = validate_mrz_checksum(line2)
                        score = 1.0 if checksum_valid else 0.5

                        if score > best_score:
                            best_score = score
                            best_result = MRZResult(
                                line1=line1,
                                line2=line2,
                                method=f"tesseract_psm{psm}_{preprocess}",
                                rotation=rotation,
                                confidence=score
                            )

                        # If checksum passes, we're done
                        if checksum_valid:
                            logger.info(f"MRZ found with valid checksum: rotation={rotation}, method=tesseract_psm{psm}_{preprocess}")
                            return best_result

                except Exception as e:
                    logger.debug(f"Tesseract attempt failed: {e}")
                    continue

    # Try EasyOCR as fallback (slower but sometimes better)
    if best_result is None or best_score < 1.0:
        for rotation in rotations:
            try:
                img = rotate_image(image_path, rotation)
            except Exception:
                continue

            for preprocess in preprocess_methods:
                try:
                    processed = preprocess_for_ocr(img, preprocess)
                    text = ocr_easyocr(processed)

                    if not text:
                        continue

                    mrz = detect_mrz_lines(text)
                    if mrz:
                        line1, line2 = mrz
                        checksum_valid = validate_mrz_checksum(line2)
                        score = 1.0 if checksum_valid else 0.5

                        if score > best_score:
                            best_score = score
                            best_result = MRZResult(
                                line1=line1,
                                line2=line2,
                                method=f"easyocr_{preprocess}",
                                rotation=rotation,
                                confidence=score
                            )

                        if checksum_valid:
                            logger.info(f"MRZ found with EasyOCR: rotation={rotation}")
                            return best_result

                except Exception as e:
                    logger.debug(f"EasyOCR attempt failed: {e}")
                    continue

    if best_result:
        logger.warning(f"MRZ found but checksum failed: {best_result.method}")
    else:
        logger.error("No MRZ found in image")

    return best_result


# Convenience function for testing
def extract_mrz_from_file(file_path: str) -> Optional[dict]:
    """
    Extract MRZ and return as dictionary.

    Convenience function for testing/debugging.
    """
    result = extract_mrz(Path(file_path))
    if result:
        return {
            "line1": result.line1,
            "line2": result.line2,
            "method": result.method,
            "rotation": result.rotation,
            "confidence": result.confidence
        }
    return None


if __name__ == "__main__":
    """Test the OCR service directly."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.extraction.ocr_service <passport_file>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    file_path = sys.argv[1]
    print(f"Processing: {file_path}")

    result = extract_mrz(Path(file_path))

    if result:
        print("\n=== MRZ EXTRACTED ===")
        print(f"Line 1: {result.line1}")
        print(f"Line 2: {result.line2}")
        print(f"Method: {result.method}")
        print(f"Rotation: {result.rotation} degrees")
        print(f"Confidence: {result.confidence}")
    else:
        print("\nFailed to extract MRZ")
        sys.exit(1)
