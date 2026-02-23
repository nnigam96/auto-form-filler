"""
Reflection Agent for OCR Error Correction.

Uses a local LLM (Ollama) to clean up OCR errors in extracted passport data.
Implements the Reflection pattern: analyze output, detect errors, fix them.

Improvements over basic OCR:
1. Vision-based correction using LLaVA
2. MRZ cross-validation as ground truth
3. Partial corrections (apply what works, keep original for failures)
4. Multi-pass reflection with error feedback
5. Flexible date parsing
"""

import json
import logging
import os
import re
import requests
import base64
from datetime import date, datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from app.models.schemas import PassportData, Sex

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


# =============================================================================
# Date Parsing Utilities
# =============================================================================

def parse_flexible_date(date_input: Any) -> Optional[date]:
    """
    Parse dates in various formats that LLMs might return.

    Handles:
    - YYYY-MM-DD (ISO)
    - YY-MM-DD (truncated year)
    - YYMMDD (MRZ format)
    - DD/MM/YYYY, MM/DD/YYYY
    - Already a date object
    """
    if date_input is None:
        return None

    if isinstance(date_input, date):
        return date_input

    if isinstance(date_input, datetime):
        return date_input.date()

    date_str = str(date_input).strip()
    if not date_str:
        return None

    # Try ISO format first
    try:
        return datetime.fromisoformat(date_str).date()
    except:
        pass

    # YY-MM-DD format (e.g., "96-10-25" or "19-10-25")
    match = re.match(r'^(\d{2})-(\d{2})-(\d{2})$', date_str)
    if match:
        yy, mm, dd = map(int, match.groups())
        # Determine century: if YY > 50, assume 1900s, else 2000s
        year = 1900 + yy if yy > 50 else 2000 + yy
        try:
            return date(year, mm, dd)
        except:
            pass

    # YYMMDD format (MRZ)
    if len(date_str) == 6 and date_str.isdigit():
        try:
            yy = int(date_str[0:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            year = 1900 + yy if yy > 50 else 2000 + yy
            return date(year, mm, dd)
        except:
            pass

    # DD/MM/YYYY or MM/DD/YYYY
    slash_match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if slash_match:
        p1, p2, year = map(int, slash_match.groups())
        # Heuristic: if first part > 12, it's DD/MM/YYYY
        if p1 > 12:
            try:
                return date(year, p2, p1)
            except:
                pass
        else:
            try:
                return date(year, p1, p2)
            except:
                pass

    # YYYY/MM/DD
    match = re.match(r'^(\d{4})/(\d{2})/(\d{2})$', date_str)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except:
            pass

    logger.debug(f"Could not parse date: {date_str}")
    return None


def parse_sex(sex_input: Any) -> Sex:
    """Parse sex field with flexibility."""
    if isinstance(sex_input, Sex):
        return sex_input

    sex_str = str(sex_input).upper().strip()
    if sex_str in ["M", "MALE"]:
        return Sex.MALE
    elif sex_str in ["F", "FEMALE"]:
        return Sex.FEMALE
    else:
        return Sex.OTHER


# =============================================================================
# MRZ Parsing for Ground Truth
# =============================================================================

def parse_mrz_ground_truth(mrz_lines: List[str]) -> Dict[str, Any]:
    """
    Extract ground truth from MRZ lines.
    MRZ is more reliable than visual OCR for certain fields.
    """
    if not mrz_lines or len(mrz_lines) < 2:
        return {}

    line1 = mrz_lines[0].upper().replace(" ", "")
    line2 = mrz_lines[1].upper().replace(" ", "")

    # Pad to 44 chars
    line1 = line1.ljust(44, "<")[:44]
    line2 = line2.ljust(44, "<")[:44]

    ground_truth = {}

    # Parse line1: P<COUNTRY<SURNAME<<GIVEN<NAMES
    try:
        names = line1[5:44].split("<<", 1)
        ground_truth["surname_mrz"] = names[0].replace("<", "").strip()
        ground_truth["given_names_mrz"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
        ground_truth["country_mrz"] = line1[2:5]
    except:
        pass

    # Parse line2: passport number, nationality, DOB, sex, expiry
    try:
        ground_truth["passport_number_mrz"] = line2[0:9].replace("<", "").strip()
        ground_truth["nationality_mrz"] = line2[10:13]
        ground_truth["dob_mrz"] = line2[13:19]  # YYMMDD
        ground_truth["sex_mrz"] = line2[20]
        ground_truth["expiry_mrz"] = line2[21:27]  # YYMMDD
    except:
        pass

    return ground_truth


# =============================================================================
# Image Handling
# =============================================================================

def _get_image_path_for_reflection(file_path: Path) -> Optional[Path]:
    """Convert PDF to image if needed, or return image path as-is."""
    if file_path.suffix.lower() == ".pdf":
        try:
            from app.utils.pdf_utils import pdf_to_images
            images = pdf_to_images(file_path, dpi=300)
            if images:
                return images[0]
        except Exception as e:
            logger.warning(f"PDF to image conversion failed: {e}")
            return None
    return file_path


# =============================================================================
# Main Entry Point
# =============================================================================

def reflect_and_fix(
    passport_data: PassportData,
    mrz_lines: List[str],
    fraud_flags: List[str],
    image_path: Optional[Path] = None,
    use_vision: bool = True,
    max_attempts: int = 2
) -> Optional[PassportData]:
    """
    Main entry point for the Reflection Agent.

    Strategy:
    1. Parse MRZ for ground truth
    2. Try Vision-based reflection (most accurate)
    3. Fall back to text-only reflection
    4. Apply partial corrections if full parsing fails
    5. Retry with error feedback if needed
    """
    if not check_ollama_available():
        logger.warning("Ollama not available, skipping reflection.")
        return None

    if not fraud_flags:
        return None  # No reflection needed if data is clean

    # Get MRZ ground truth
    mrz_truth = parse_mrz_ground_truth(mrz_lines)

    # Try Vision-Based Reflection first
    if image_path and use_vision and image_path.exists():
        actual_image_path = _get_image_path_for_reflection(image_path)
        if actual_image_path:
            logger.info("Reflection Agent: Engaging Vision Model for pixel-level correction.")

            for attempt in range(max_attempts):
                result = reflect_with_vision(
                    passport_data, mrz_lines, fraud_flags,
                    actual_image_path, mrz_truth, attempt
                )
                if result:
                    return result
                logger.debug(f"Vision reflection attempt {attempt + 1} failed, retrying...")

    # Fallback to Logic-Based Reflection
    logger.info("Reflection Agent: Engaging Logic Model for pattern correction.")
    for attempt in range(max_attempts):
        result = reflect_text_only(passport_data, mrz_lines, fraud_flags, mrz_truth, attempt)
        if result:
            return result

    # Last resort: Apply MRZ ground truth directly for fixable fields
    return apply_mrz_corrections(passport_data, mrz_truth, fraud_flags)


# =============================================================================
# Vision-Based Reflection
# =============================================================================

def reflect_with_vision(
    passport_data: PassportData,
    mrz_lines: List[str],
    fraud_flags: List[str],
    image_path: Path,
    mrz_truth: Dict[str, Any],
    attempt: int = 0
) -> Optional[PassportData]:
    """
    Uses LLaVA (Vision) to resolve ambiguity by looking at the actual document.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Build prompt with MRZ truth for cross-validation
        mrz_info = ""
        if mrz_truth:
            mrz_info = f"""
MRZ GROUND TRUTH (Use this to verify your corrections):
- Passport Number: {mrz_truth.get('passport_number_mrz', 'N/A')}
- Nationality: {mrz_truth.get('nationality_mrz', 'N/A')}
- Date of Birth (YYMMDD): {mrz_truth.get('dob_mrz', 'N/A')}
- Sex: {mrz_truth.get('sex_mrz', 'N/A')}
- Expiry (YYMMDD): {mrz_truth.get('expiry_mrz', 'N/A')}
"""

        retry_hint = ""
        if attempt > 0:
            retry_hint = """
IMPORTANT: Previous attempt failed validation. Please ensure:
- Dates are in YYYY-MM-DD format (e.g., 1996-10-25, NOT 96-10-25)
- For years in 1900s, use full year (1996, not 96)
- Sex is exactly "M", "F", or "X"
"""

        prompt = f"""You are a Document Forensics AI. Fix OCR errors by comparing extracted data to the passport image.

CURRENT EXTRACTED DATA (Contains OCR Errors):
{json.dumps(passport_data.model_dump(mode='json', exclude_none=True), indent=2)}

DETECTED ISSUES:
{json.dumps(fraud_flags, indent=2)}
{mrz_info}
{retry_hint}
INSTRUCTIONS:
1. Look at the passport image carefully.
2. Fix these specific OCR errors:
   - Remove garbage characters (e.g., "NIKHILKKKKK" → "NIKHIL")
   - Fix number/letter confusion: "1ND" → "IND", "0" ↔ "O"
   - Clean watermark artifacts and repeated characters
3. For dates: Convert to YYYY-MM-DD format. If MRZ shows "961025", output "1996-10-25"
4. Keep original if truly unreadable - do NOT hallucinate.

OUTPUT: Return ONLY valid JSON with corrected fields. Example format:
{{
  "surname": "SMITH",
  "given_names": "JOHN WILLIAM",
  "passport_number": "AB1234567",
  "nationality": "USA",
  "date_of_birth": "1985-03-15",
  "sex": "M",
  "expiry_date": "2028-06-20",
  "country_of_issue": "USA"
}}"""

        # Log the prompt
        logger.info("=" * 60)
        logger.info("REFLECTION AGENT - VISION REQUEST")
        logger.info("=" * 60)
        logger.info(f"Model: {OLLAMA_VISION_MODEL}")
        logger.info(f"Prompt:\n{prompt}")
        logger.info("=" * 60)

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_VISION_MODEL,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=90
        )

        # Log raw response
        if response.status_code == 200:
            raw_resp = response.json().get("response", "")
            logger.info("=" * 60)
            logger.info("REFLECTION AGENT - VISION RESPONSE (RAW)")
            logger.info("=" * 60)
            logger.info(f"{raw_resp}")
            logger.info("=" * 60)

        return parse_reflection_response(response, passport_data, mrz_truth)

    except Exception as e:
        logger.error(f"Vision reflection failed: {e}")
        return None


# =============================================================================
# Text-Only Reflection
# =============================================================================

def reflect_text_only(
    passport_data: PassportData,
    mrz_lines: List[str],
    fraud_flags: List[str],
    mrz_truth: Dict[str, Any],
    attempt: int = 0
) -> Optional[PassportData]:
    """
    Uses text LLM to fix logical inconsistencies using MRZ as ground truth.
    """
    try:
        mrz_text = "\n".join(mrz_lines) if mrz_lines else "NOT FOUND"

        retry_hint = ""
        if attempt > 0:
            retry_hint = """
CRITICAL: Previous attempt failed. Ensure:
- date_of_birth and expiry_date are YYYY-MM-DD (e.g., "1996-10-25")
- sex is exactly "M", "F", or "X"
- Remove ALL repeated garbage characters
"""

        prompt = f"""Fix OCR errors in this passport data using the MRZ as ground truth.

MRZ CODE (Source of Truth):
{mrz_text}

NOISY DATA:
{json.dumps(passport_data.model_dump(mode='json', exclude_none=True), indent=2)}

ISSUES TO FIX:
{json.dumps(fraud_flags, indent=2)}
{retry_hint}
RULES:
1. MRZ dates are YYMMDD. Convert to YYYY-MM-DD (e.g., "961025" → "1996-10-25")
2. For YY: if > 50, prefix with 19; else prefix with 20
3. Remove repeated characters: "NIKHILKKKKKK" → "NIKHIL"
4. Fix OCR errors: "1ND" → "IND"
5. Do NOT invent data

Return ONLY the corrected JSON object, nothing else."""

        # Log the prompt
        logger.info("=" * 60)
        logger.info("REFLECTION AGENT - TEXT REQUEST")
        logger.info("=" * 60)
        logger.info(f"Model: {OLLAMA_MODEL}")
        logger.info(f"Prompt:\n{prompt}")
        logger.info("=" * 60)

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=30
        )

        # Log raw response
        if response.status_code == 200:
            raw_resp = response.json().get("response", "")
            logger.info("=" * 60)
            logger.info("REFLECTION AGENT - TEXT RESPONSE (RAW)")
            logger.info("=" * 60)
            logger.info(f"{raw_resp}")
            logger.info("=" * 60)

        return parse_reflection_response(response, passport_data, mrz_truth)

    except Exception as e:
        logger.error(f"Text reflection failed: {e}")
        return None


# =============================================================================
# Response Parsing with Partial Corrections
# =============================================================================

def parse_reflection_response(
    response: requests.Response,
    original_data: PassportData,
    mrz_truth: Dict[str, Any]
) -> Optional[PassportData]:
    """
    Parse LLM output with flexible handling and partial corrections.
    """
    if response.status_code != 200:
        logger.error(f"LLM request failed: {response.status_code}")
        return None

    try:
        result = response.json()
        raw_text = result.get("response", "")

        # Extract JSON from response
        json_str = extract_json_from_text(raw_text)
        if not json_str:
            logger.warning("No JSON found in LLM response")
            return None

        cleaned_dict = json.loads(json_str)
        cleaned_dict.pop("_reasoning", None)

        # Apply corrections field by field with validation
        return apply_corrections(original_data, cleaned_dict, mrz_truth)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to parse reflection response: {e}")
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from LLM response, handling markdown blocks."""
    # Try markdown code blocks first
    if "```json" in text:
        try:
            return text.split("```json")[1].split("```")[0].strip()
        except:
            pass

    if "```" in text:
        try:
            return text.split("```")[1].split("```")[0].strip()
        except:
            pass

    # Try to find JSON object directly
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return match.group()

    # Last resort: the whole text might be JSON
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    return None


def apply_corrections(
    original_data: PassportData,
    corrections: Dict[str, Any],
    mrz_truth: Dict[str, Any]
) -> Optional[PassportData]:
    """
    Apply corrections field by field, keeping original if correction fails.
    This ensures partial corrections work even if some fields are bad.
    """
    original_dict = original_data.model_dump()
    corrected_dict = original_dict.copy()
    corrections_applied = []

    # Text fields - simple replacement with cleanup
    text_fields = ["surname", "given_names", "passport_number", "nationality", "country_of_issue", "place_of_birth"]
    for field in text_fields:
        if field in corrections and corrections[field]:
            new_value = clean_text_field(str(corrections[field]))
            if new_value and new_value != original_dict.get(field):
                corrected_dict[field] = new_value
                corrections_applied.append(field)

    # Date fields - flexible parsing
    date_fields = ["date_of_birth", "expiry_date", "issue_date"]
    for field in date_fields:
        if field in corrections and corrections[field]:
            parsed_date = parse_flexible_date(corrections[field])
            if parsed_date:
                corrected_dict[field] = parsed_date
                corrections_applied.append(field)

    # Sex field
    if "sex" in corrections and corrections["sex"]:
        corrected_dict["sex"] = parse_sex(corrections["sex"])
        corrections_applied.append("sex")

    # Cross-validate with MRZ truth for critical fields
    corrected_dict = cross_validate_with_mrz(corrected_dict, mrz_truth)

    if not corrections_applied:
        logger.debug("No corrections applied from LLM response")
        return None

    # Mark as reflected and boost confidence
    corrected_dict["extraction_method"] = f"{original_dict.get('extraction_method', 'unknown')}_reflected"
    corrected_dict["confidence_score"] = min(0.95, original_dict.get("confidence_score", 0.5) + 0.3)

    logger.info(f"Reflection applied corrections to: {corrections_applied}")

    try:
        return PassportData(**corrected_dict)
    except Exception as e:
        logger.warning(f"Failed to create PassportData with corrections: {e}")
        return None


def clean_text_field(text: str) -> str:
    """Clean garbage characters from text fields."""
    if not text:
        return text

    # Remove repeated characters (3+ in a row)
    cleaned = re.sub(r'(.)\1{2,}', r'\1', text)

    # Remove common garbage patterns
    cleaned = re.sub(r'[kK]{2,}', '', cleaned)
    cleaned = re.sub(r'\d{4,}$', '', cleaned)  # Trailing numbers

    return cleaned.strip()


def cross_validate_with_mrz(data: Dict[str, Any], mrz_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cross-validate corrected data with MRZ ground truth.
    MRZ is authoritative for passport_number, nationality, DOB, sex, expiry.
    """
    if not mrz_truth:
        return data

    # Passport number from MRZ
    if mrz_truth.get("passport_number_mrz"):
        mrz_pn = mrz_truth["passport_number_mrz"]
        if len(mrz_pn) >= 6:  # Valid passport number
            data["passport_number"] = mrz_pn

    # Nationality from MRZ (more reliable than OCR)
    if mrz_truth.get("nationality_mrz"):
        nat = mrz_truth["nationality_mrz"]
        if nat.isalpha() and len(nat) == 3:
            data["nationality"] = nat

    # DOB from MRZ
    if mrz_truth.get("dob_mrz"):
        mrz_dob = parse_flexible_date(mrz_truth["dob_mrz"])
        if mrz_dob:
            data["date_of_birth"] = mrz_dob

    # Expiry from MRZ
    if mrz_truth.get("expiry_mrz"):
        mrz_exp = parse_flexible_date(mrz_truth["expiry_mrz"])
        if mrz_exp:
            data["expiry_date"] = mrz_exp

    # Sex from MRZ
    if mrz_truth.get("sex_mrz") in ["M", "F", "<"]:
        data["sex"] = parse_sex(mrz_truth["sex_mrz"])

    return data


# =============================================================================
# Fallback: Direct MRZ Corrections
# =============================================================================

def apply_mrz_corrections(
    passport_data: PassportData,
    mrz_truth: Dict[str, Any],
    fraud_flags: List[str]
) -> Optional[PassportData]:
    """
    Last resort: Apply MRZ ground truth directly without LLM.
    Useful when LLM fails but MRZ is readable.
    """
    if not mrz_truth:
        return None

    original_dict = passport_data.model_dump()
    corrected = False

    # Apply MRZ truth for fields that commonly have OCR errors
    if "repeated characters" in str(fraud_flags).lower():
        # Clean names using MRZ
        if mrz_truth.get("surname_mrz"):
            original_dict["surname"] = mrz_truth["surname_mrz"].title()
            corrected = True
        if mrz_truth.get("given_names_mrz"):
            original_dict["given_names"] = mrz_truth["given_names_mrz"].title()
            corrected = True

    if "non-alpha" in str(fraud_flags).lower() or "nationality" in str(fraud_flags).lower():
        if mrz_truth.get("nationality_mrz"):
            original_dict["nationality"] = mrz_truth["nationality_mrz"]
            corrected = True

    # Always trust MRZ for passport number
    if mrz_truth.get("passport_number_mrz"):
        original_dict["passport_number"] = mrz_truth["passport_number_mrz"]
        corrected = True

    # Dates from MRZ
    if mrz_truth.get("dob_mrz"):
        dob = parse_flexible_date(mrz_truth["dob_mrz"])
        if dob:
            original_dict["date_of_birth"] = dob
            corrected = True

    if mrz_truth.get("expiry_mrz"):
        exp = parse_flexible_date(mrz_truth["expiry_mrz"])
        if exp:
            original_dict["expiry_date"] = exp
            corrected = True

    if not corrected:
        return None

    original_dict["extraction_method"] = f"{original_dict.get('extraction_method', 'unknown')}_mrz_corrected"
    original_dict["confidence_score"] = 0.85

    logger.info("Applied direct MRZ corrections as fallback")

    try:
        return PassportData(**original_dict)
    except Exception as e:
        logger.warning(f"Failed to apply MRZ corrections: {e}")
        return None
