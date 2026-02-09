"""
Voting and validation logic for OCR results.
The deterministic "brain" of the agent.
"""

from typing import List, Tuple, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def vote_on_results(results: List[dict]) -> Tuple[Optional[dict], float, str]:
    """
    Vote on OCR results to pick the best one.

    Returns:
        (best_parsed_data, confidence, source)
    """
    successful = [r for r in results if r.get("success") and r.get("parsed")]

    if not successful:
        return None, 0.0, "none"

    # Rule 1: PassportEye with valid checksum wins
    passport_eye = next(
        (r for r in successful if r["source"] == "passport_eye" and r.get("checksum_valid")),
        None,
    )
    if passport_eye:
        logger.info("PassportEye with valid checksum selected")
        return passport_eye["parsed"], 1.0, "passport_eye"

    # Rule 2: Check agreement between Tesseract and EasyOCR
    tesseract = next((r for r in successful if r["source"] == "tesseract"), None)
    easyocr = next((r for r in successful if r["source"] == "easyocr"), None)

    if tesseract and easyocr:
        agreement = calculate_field_agreement(tesseract["parsed"], easyocr["parsed"])
        logger.info(f"Tesseract/EasyOCR agreement: {agreement:.1%}")

        if agreement >= 0.8:
            # High agreement - merge results, prefer one with checksum
            if tesseract.get("checksum_valid"):
                return tesseract["parsed"], 0.85, "tesseract_consensus"
            elif easyocr.get("checksum_valid"):
                return easyocr["parsed"], 0.85, "easyocr_consensus"
            else:
                return tesseract["parsed"], 0.75, "tesseract_consensus"

    # Rule 3: Any result with valid checksum
    checksum_valid = [r for r in successful if r.get("checksum_valid")]
    if checksum_valid:
        best = checksum_valid[0]
        return best["parsed"], 0.7, best["source"]

    # Rule 4: Any result (low confidence)
    if successful:
        best = successful[0]
        return best["parsed"], 0.5, best["source"]

    return None, 0.0, "none"


def calculate_field_agreement(result1: dict, result2: dict) -> float:
    """
    Calculate agreement ratio between two parsed results.
    Returns 0.0 - 1.0.
    """
    if not result1 or not result2:
        return 0.0

    fields = [
        "surname",
        "given_names",
        "passport_number",
        "date_of_birth",
        "expiry_date",
        "sex",
    ]

    matches = 0
    total = 0

    for field in fields:
        val1 = str(result1.get(field, "")).strip().upper()
        val2 = str(result2.get(field, "")).strip().upper()

        if val1 and val2:
            total += 1
            if val1 == val2:
                matches += 1
            elif field in ["surname", "given_names"]:
                # Fuzzy match for names (allow small differences)
                if _similarity(val1, val2) > 0.8:
                    matches += 0.5

    return matches / total if total > 0 else 0.0


def _similarity(s1: str, s2: str) -> float:
    """Simple character-level similarity."""
    if not s1 or not s2:
        return 0.0

    matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
    return matches / max(len(s1), len(s2))


def critic_validate(data: dict) -> Tuple[bool, List[str]]:
    """
    Validate extracted data for consistency and fraud indicators.

    Returns:
        (is_valid, list_of_fraud_flags)
    """
    if not data:
        return False, ["NO_DATA"]

    fraud_flags = []

    # Check 1: Passport number format
    passport_num = data.get("passport_number", "")
    if not passport_num or len(passport_num) < 5:
        fraud_flags.append("INVALID_PASSPORT_NUMBER")

    # Check 2: DOB is reasonable
    dob_str = data.get("date_of_birth", "")
    if dob_str:
        try:
            # MRZ format: YYMMDD
            if len(dob_str) == 6:
                year = int(dob_str[0:2])
                month = int(dob_str[2:4])
                day = int(dob_str[4:6])

                # Determine century
                current_year = datetime.now().year % 100
                if year > current_year:
                    full_year = 1900 + year
                else:
                    full_year = 2000 + year

                dob = date(full_year, month, day)

                # Check age is reasonable (0-120)
                age = (date.today() - dob).days / 365.25
                if age < 0:
                    fraud_flags.append("DOB_IN_FUTURE")
                elif age > 120:
                    fraud_flags.append("DOB_TOO_OLD")
        except:
            fraud_flags.append("DOB_PARSE_ERROR")

    # Check 3: Expiry is after DOB
    exp_str = data.get("expiry_date", "")
    if dob_str and exp_str:
        try:
            if len(exp_str) == 6 and len(dob_str) == 6:
                # Simple comparison: expiry should be > dob
                exp_year = int(exp_str[0:2])
                dob_year = int(dob_str[0:2])

                # Both in 20xx for comparison
                if exp_year < 50:
                    exp_year += 2000
                else:
                    exp_year += 1900

                if dob_year < 50:
                    dob_year += 2000
                else:
                    dob_year += 1900

                if exp_year < dob_year:
                    fraud_flags.append("EXPIRY_BEFORE_BIRTH")
        except:
            pass

    # Check 4: Name contains only valid characters
    surname = data.get("surname", "")
    given_names = data.get("given_names", "")

    for name in [surname, given_names]:
        if name:
            invalid_chars = [
                c
                for c in name
                if not (c.isalpha() or c.isspace() or c == "-" or c == "'")
            ]
            if invalid_chars:
                fraud_flags.append(f"INVALID_NAME_CHARS: {''.join(invalid_chars)}")

    # Check 5: Sex is valid
    sex = data.get("sex", "")
    if sex and sex not in ["M", "F", "<", "X"]:
        fraud_flags.append(f"INVALID_SEX: {sex}")

    is_valid = len(fraud_flags) == 0
    return is_valid, fraud_flags
