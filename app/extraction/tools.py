"""
Async OCR tools for parallel execution.
Each tool returns a standardized dict, not Pydantic models.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running sync OCR in async context
_executor = ThreadPoolExecutor(max_workers=3)


async def run_passport_eye(image_path: str) -> dict:
    """
    Run PassportEye MRZ detection.
    Returns raw dict with extracted fields.
    """
    result = {
        "source": "passport_eye",
        "success": False,
        "raw_text": "",
        "parsed": None,
        "checksum_valid": False,
        "error": None,
    }

    def _extract():
        try:
            from passporteye import read_mrz

            mrz = read_mrz(str(image_path))

            if mrz is None:
                return None, "MRZ not detected"

            data = mrz.to_dict()

            # Check validity
            valid_score = data.get("valid_score", 0)
            valid_number = data.get("valid_number", False)
            valid_dob = data.get("valid_date_of_birth", False)
            valid_exp = data.get("valid_expiration_date", False)

            checksum_valid = valid_score >= 60 and sum([valid_number, valid_dob, valid_exp]) >= 2

            parsed = {
                "surname": data.get("surname", "").replace("<", ""),
                "given_names": data.get("names", "").replace("<", " ").strip(),
                "passport_number": data.get("number", "").replace("<", ""),
                "nationality": data.get("nationality", ""),
                "date_of_birth": data.get("date_of_birth", ""),
                "sex": data.get("sex", ""),
                "expiry_date": data.get("expiration_date", ""),
                "country": data.get("country", ""),
            }

            return {
                "parsed": parsed,
                "checksum_valid": checksum_valid,
                "raw_text": data.get("raw_text", ""),
                "valid_score": valid_score,
            }, None

        except Exception as e:
            return None, str(e)

    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(_executor, _extract)

    if error:
        result["error"] = error
    elif data:
        result["success"] = True
        result["parsed"] = data["parsed"]
        result["checksum_valid"] = data["checksum_valid"]
        result["raw_text"] = data["raw_text"]

    return result


async def run_tesseract(image_path: str) -> dict:
    """
    Run Tesseract OCR with MRZ detection.
    """
    result = {
        "source": "tesseract",
        "success": False,
        "raw_text": "",
        "parsed": None,
        "checksum_valid": False,
        "error": None,
    }

    def _extract():
        try:
            import cv2
            import pytesseract
            from PIL import Image

            # Try multiple rotations
            img = cv2.imread(str(image_path))
            if img is None:
                return None, "Could not read image"

            rotations = [0, 90, 180, 270]

            for rotation in rotations:
                if rotation == 90:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    rotated = img

                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                pil_img = Image.fromarray(gray)

                text = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")

                # Try to find MRZ
                parsed, checksum_valid = _parse_mrz_from_text(text)
                if parsed:
                    return {
                        "parsed": parsed,
                        "checksum_valid": checksum_valid,
                        "raw_text": text,
                        "rotation": rotation,
                    }, None

            return None, "MRZ not found in any rotation"

        except Exception as e:
            return None, str(e)

    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(_executor, _extract)

    if error:
        result["error"] = error
    elif data:
        result["success"] = True
        result["parsed"] = data["parsed"]
        result["checksum_valid"] = data["checksum_valid"]
        result["raw_text"] = data["raw_text"]

    return result


async def run_easyocr(image_path: str) -> dict:
    """
    Run EasyOCR with MRZ detection.
    """
    result = {
        "source": "easyocr",
        "success": False,
        "raw_text": "",
        "parsed": None,
        "checksum_valid": False,
        "error": None,
    }

    def _extract():
        try:
            import easyocr
            import cv2

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)

            img = cv2.imread(str(image_path))
            if img is None:
                return None, "Could not read image"

            # Try rotations
            rotations = [0, 90, 180, 270]

            for rotation in rotations:
                if rotation == 90:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    rotated = img

                gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                results = reader.readtext(gray, detail=0)
                text = "\n".join(results)

                parsed, checksum_valid = _parse_mrz_from_text(text)
                if parsed:
                    return {
                        "parsed": parsed,
                        "checksum_valid": checksum_valid,
                        "raw_text": text,
                        "rotation": rotation,
                    }, None

            return None, "MRZ not found in any rotation"

        except Exception as e:
            return None, str(e)

    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(_executor, _extract)

    if error:
        result["error"] = error
    elif data:
        result["success"] = True
        result["parsed"] = data["parsed"]
        result["checksum_valid"] = data["checksum_valid"]
        result["raw_text"] = data["raw_text"]

    return result


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
                # Validate line2 has digits in date positions
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
    # Pad to 44 chars
    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]

    # Parse line1: P<COUNTRY<SURNAME<<GIVEN<NAMES
    names = line1[5:44].split("<<", 1)
    surname = names[0].replace("<", "")
    given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""

    # Parse line2
    passport_number = line2[0:9].replace("<", "")
    nationality = line2[10:13]
    dob = line2[13:19]
    sex = line2[20]
    expiry = line2[21:27]

    return {
        "surname": surname,
        "given_names": given_names,
        "passport_number": passport_number,
        "nationality": nationality,
        "date_of_birth": dob,
        "sex": sex,
        "expiry_date": expiry,
        "country": line1[2:5],
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
        # Passport number check
        passport_check = int(line2[9]) if line2[9].isdigit() else -1
        # DOB check
        dob_check = int(line2[19]) if line2[19].isdigit() else -1
        # Expiry check
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
