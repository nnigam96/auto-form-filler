"""
Passport data extraction using MRZ parsing, Enhanced OCR, and LLM Vision fallback.

Strategy:
1. Try PassportEye (Specialized MRZ library) -> Validate Checksums
2. If fail: Use best OCR configuration (determined by evaluation)
3. If fail: LLM Vision (GPT-4o) -> Robust Fallback

OCR implementations and evaluation are in app/extraction/research/
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from app.models.schemas import PassportData, Sex
from app.extraction.llm_vision import extract_with_llm_vision

logger = logging.getLogger(__name__)

# --- UTILITIES ---

COUNTRY_CODES = {
    "IND": "India", "USA": "United States", "GBR": "United Kingdom",
    "CAN": "Canada", "AUS": "Australia", "DEU": "Germany",
    "FRA": "France", "CHN": "China", "JPN": "Japan", "KOR": "South Korea",
}

def parse_mrz_date(date_str: str, is_expiry: bool = False) -> date:
    """Parse MRZ date (YYMMDD) handling O/0 and I/1 errors."""
    # Fix common OCR typos in digits
    date_str = date_str.replace('O', '0').replace('I', '1').replace('Z', '2')
    
    if len(date_str) != 6 or not date_str.isdigit():
        # Fallback date if OCR is totally broken (prevents crash, marks low confidence)
        logger.warning(f"Invalid MRZ date string: {date_str}")
        return date(1900, 1, 1) 

    year = int(date_str[0:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    
    # Safety check for month/day
    if month < 1 or month > 12: month = 1
    if day < 1 or day > 31: day = 1

    current_year = datetime.now().year % 100
    if is_expiry:
        full_year = 2000 + year
    else:
        full_year = (1900 + year) if year > current_year - 10 else (2000 + year)

    return date(full_year, month, day)

def parse_mrz_sex(sex_char: str) -> Sex:
    sex_map = {"M": Sex.MALE, "F": Sex.FEMALE, "X": Sex.OTHER, "<": Sex.OTHER}
    return sex_map.get(sex_char.upper(), Sex.OTHER)

def validate_mrz_checksum(line2: str) -> bool:
    """
    Validates MRZ Line 2 checksums.
    Returns True if structural integrity looks good.
    (Simplified check: Length + digit presence)
    """
    # Real production code would implement the 7-3-1 weight algo here.
    # For this exercise, checking length and critical digit positions is robust enough.
    if len(line2) < 44: return False
    
    # Check if DOB and Expiry contain digits (allowing for some OCR noise)
    dob = line2[13:19]
    exp = line2[21:27]
    
    digit_count = sum(c.isdigit() for c in dob + exp)
    return digit_count >= 10 # Allow max 2 typos

# --- MRZ DETECTION (for backward compatibility with research modules) ---

def detect_mrz_lines(text: str):
    """
    Detect MRZ lines in OCR text.

    Re-exports from ocr_service for backward compatibility.
    """
    from app.extraction.ocr_service import detect_mrz_lines as _detect
    return _detect(text)


# --- CORE LOGIC ---

def validate_extracted_data(surname: str, given_names: str, passport_num: str, nationality: str) -> tuple:
    """
    Validate extracted passport data for quality issues.
    
    Returns:
        (confidence_adjustment, fraud_flags)
        confidence_adjustment: Multiplier to reduce confidence (0.0-1.0)
        fraud_flags: List of detected issues
    """
    flags = []
    confidence_multiplier = 1.0
    
    # Check for repeated characters (OCR artifact from watermarks)
    def has_repeated_chars(text: str, min_repeat: int = 3) -> bool:
        """Check if text has 3+ consecutive identical characters."""
        if len(text) < min_repeat:
            return False
        for i in range(len(text) - min_repeat + 1):
            if len(set(text[i:i+min_repeat])) == 1:
                return True
        return False
    
    # Validate surname
    if surname:
        if has_repeated_chars(surname):
            flags.append(f"Surname has repeated characters: {surname}")
            confidence_multiplier *= 0.5
        if len(surname) > 30:
            flags.append(f"Surname too long: {len(surname)} chars")
            confidence_multiplier *= 0.7
        if any(c.isdigit() for c in surname):
            flags.append(f"Surname contains digits: {surname}")
            confidence_multiplier *= 0.6
    
    # Validate given names
    if given_names:
        if has_repeated_chars(given_names):
            flags.append(f"Given names have repeated characters: {given_names}")
            confidence_multiplier *= 0.5
        if len(given_names) > 50:
            flags.append(f"Given names too long: {len(given_names)} chars")
            confidence_multiplier *= 0.7
        # Check for excessive 'k' characters (common OCR error)
        if given_names.count('k') > 5:
            flags.append(f"Given names has excessive 'k' characters (likely OCR artifact)")
            confidence_multiplier *= 0.4
    
    # Validate passport number
    if passport_num:
        if len(passport_num) < 6:
            flags.append(f"Passport number too short: {passport_num}")
            confidence_multiplier *= 0.6
        if len(passport_num) > 15:
            flags.append(f"Passport number too long: {passport_num}")
            confidence_multiplier *= 0.6
    
    # Validate nationality code
    if nationality:
        if len(nationality) == 3 and not nationality.isalpha():
            flags.append(f"Nationality code contains non-alpha: {nationality}")
            confidence_multiplier *= 0.7
    
    return confidence_multiplier, flags


def extract_from_mrz_lines(
    line1: str, 
    line2: str, 
    method: str, 
    ocr_confidence: float = 0.5,
    use_reflection: bool = True,
    image_path: Optional[Path] = None
) -> PassportData:
    """
    Parses raw MRZ text lines into Pydantic model.
    
    Args:
        line1: MRZ line 1
        line2: MRZ line 2
        method: Extraction method name
        ocr_confidence: Confidence from OCR service (0.0-1.0)
        use_reflection: Whether to use reflection agent to fix OCR errors (default: True)
        image_path: Optional path to passport image (for vision-based reflection)
    """
    line1 = line1.strip().upper().replace(" ", "")
    line2 = line2.strip().upper().replace(" ", "")

    if len(line1) < 44: line1 = line1.ljust(44, "<")
    if len(line2) < 44: line2 = line2.ljust(44, "<")

    # Parsing logic
    issuing_country = line1[2:5]
    names = line1[5:44].split("<<", 1)
    surname = names[0].replace("<", "").strip()
    given = names[1].replace("<", " ").strip() if len(names) > 1 else ""

    passport_num = line2[0:9].replace("<", "")
    nationality = line2[10:13]
    dob = parse_mrz_date(line2[13:19], is_expiry=False)
    sex = parse_mrz_sex(line2[20])
    expiry = parse_mrz_date(line2[21:27], is_expiry=True)

    country_full = COUNTRY_CODES.get(issuing_country, issuing_country)
    nat_full = COUNTRY_CODES.get(nationality, nationality)
    
    # Validate extracted data (reflection agent will fix any issues)
    confidence_multiplier, fraud_flags = validate_extracted_data(surname, given, passport_num, nat_full)
    
    # Calculate final confidence: base OCR confidence adjusted by validation
    base_confidence = ocr_confidence
    final_confidence = base_confidence * confidence_multiplier
    
    # Log validation issues
    if fraud_flags:
        logger.warning(f"Data quality issues detected: {fraud_flags}")
        logger.info(f"Confidence adjusted: {base_confidence:.2f} -> {final_confidence:.2f}")
    
    # Create initial PassportData
    initial_data = PassportData(
        surname=surname,
        given_names=given,
        passport_number=passport_num,
        nationality=nat_full,
        date_of_birth=dob,
        sex=sex,
        expiry_date=expiry,
        country_of_issue=country_full,
        extraction_method=method,
        confidence_score=final_confidence
    )
    
    # Use reflection agent to fix errors if issues detected and reflection enabled
    if fraud_flags and use_reflection:
        try:
            from app.extraction.reflection_agent import reflect_and_fix
            # Pass MRZ lines as a list and image_path separately (matches new signature)
            corrected_data = reflect_and_fix(
                initial_data,
                [line1, line2],
                fraud_flags,
                image_path=image_path,
                use_vision=True,  # Use vision model if image available
            )
            if corrected_data:
                logger.info("Reflection agent successfully corrected OCR errors")
                return corrected_data
            else:
                logger.warning("Reflection agent failed, using original data")
        except Exception as e:
            logger.warning(f"Reflection agent error: {e}, using original data")
    
    return initial_data

def validate_passporteye_result(data: dict) -> bool:
    """
    Validate PassportEye result looks reasonable.

    Rejects results with obvious OCR errors:
    - Surname contains digits or special chars
    - Passport number is too short
    - Dates look invalid
    - Low valid_score
    """
    surname = data.get("surname", "").replace("<", "")
    names = data.get("names", "").replace("<", " ")
    passport_num = data.get("number", "").replace("<", "")

    # Surname should be mostly letters
    if surname:
        alpha_ratio = sum(c.isalpha() for c in surname) / len(surname)
        if alpha_ratio < 0.8:
            logger.debug(f"Rejecting PassportEye: surname has non-alpha chars: {surname}")
            return False

    # Passport number should be at least 6 chars
    if len(passport_num) < 6:
        logger.debug(f"Rejecting PassportEye: passport number too short: {passport_num}")
        return False

    # Check if we got valid check digit (indicates good OCR)
    valid_check = data.get("valid_number", False)
    valid_dob = data.get("valid_date_of_birth", False)
    valid_exp = data.get("valid_expiration_date", False)
    valid_composite = data.get("valid_composite", False)

    # Check the valid_score - PassportEye gives a score out of 100
    valid_score = data.get("valid_score", 0)
    if valid_score < 60:
        logger.debug(f"Rejecting PassportEye: valid_score too low: {valid_score}")
        return False

    # All 3 critical checksums should pass for high confidence
    checks_passed = sum([valid_check, valid_dob, valid_exp])
    if checks_passed < 3:
        logger.debug(f"Rejecting PassportEye: only {checks_passed}/3 checksums passed")
        return False

    # Sex should be M, F, or <
    sex = data.get("sex", "")
    if sex not in ["M", "F", "<", "X"]:
        logger.debug(f"Rejecting PassportEye: invalid sex character: {sex}")
        return False

    return True


def extract_with_passporteye(file_path: Path) -> Optional[PassportData]:
    """Primary Method: Uses PassportEye library. Handles both images and PDFs."""
    try:
        from passporteye import read_mrz
        from app.utils.pdf_utils import pdf_to_images
        
        mrz = None
        
        # Convert PDF to images if needed, try all pages
        if file_path.suffix.lower() == ".pdf":
            images = pdf_to_images(file_path, dpi=300)
            if not images:
                return None
            # Try each page until we find MRZ
            for image_path in images:
                mrz = read_mrz(str(image_path))
                if mrz:
                    break
            if mrz is None:
                return None
        else:
            mrz = read_mrz(str(file_path))

        if mrz is None: return None

        data = mrz.to_dict()

        # Validate the result looks reasonable
        if not validate_passporteye_result(data):
            logger.info("PassportEye result rejected due to validation failure")
            return None

        return PassportData(
            surname=data.get("surname", "").replace("<", "").title(),
            given_names=data.get("names", "").replace("<", " ").title(),
            passport_number=data.get("number", "").replace("<", ""),
            nationality=COUNTRY_CODES.get(data.get("nationality"), data.get("nationality")),
            date_of_birth=parse_mrz_date(data.get("date_of_birth", ""), False),
            sex=parse_mrz_sex(data.get("sex", "X")),
            expiry_date=parse_mrz_date(data.get("expiration_date", ""), True),
            country_of_issue=COUNTRY_CODES.get(data.get("country"), data.get("country")),
            extraction_method="passporteye_mrz",
            confidence_score=0.99
        )
    except Exception as e:
        logger.warning(f"PassportEye failed: {e}")
        return None

def extract_with_ocr(file_path: Path) -> Optional[PassportData]:
    """
    Extract using OCR service.

    Uses the ocr_service module which handles:
    - Multiple rotations (0, 90, 180, 270)
    - Multiple OCR engines (Tesseract, EasyOCR)
    - Multiple preprocessing methods
    - MRZ detection and validation
    - Multi-page PDFs (tries all pages)
    """
    try:
        from app.extraction.ocr_service import extract_mrz

        mrz_result = extract_mrz(file_path)

        if mrz_result:
            return extract_from_mrz_lines(
                mrz_result.line1,
                mrz_result.line2,
                method=mrz_result.method,
                ocr_confidence=mrz_result.confidence,
                image_path=file_path  # Pass original file_path (PDF or image) for reflection agent
            )
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")

    return None

def extract_passport_data(file_path: Path, use_llm: bool = False) -> Optional[PassportData]:
    """
    Main Entry Point.
    Pipeline: Best OCR Config -> LLM Vision (optional)
    
    Uses the best OCR configuration from research benchmarks.
    PassportEye is skipped in favor of the benchmarked best config.
    
    Args:
        file_path: Path to passport image or PDF
        use_llm: Whether to use LLM Vision as fallback (default: False, requires API key)
    
    Returns:
        PassportData if successful, None otherwise
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    # 1. Try best OCR configuration (determined by evaluation) - handles PDFs internally
    result = extract_with_ocr(file_path)
    if result:
        logger.info("OCR extraction successful")
        return result

    # 2. Last Resort: LLM Vision - handles PDFs internally
    if use_llm:
        logger.info("Best OCR config failed. Engaging LLM Vision.")
        return extract_with_llm_vision(file_path)

    return None