"""
Fraud Detection Module for Passport Extraction.

The key insight: In a legitimate passport, Visual Text == MRZ Data.
When they don't match, it's either OCR error OR document tampering.

Strategy:
1. Extract VISUAL text (printed on passport face) using LLM Vision
2. Extract MRZ data using OCR
3. Cross-validate: Visual vs MRZ
4. Flag discrepancies as potential fraud
5. Return VISUAL data (what humans see) + fraud flags

This catches:
- MRZ tampering (changing dates in MRZ zone)
- Photo substitution (if we add face matching later)
- Data field manipulation
"""

import json
import logging
import os
import re
import base64
import requests
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from app.models.schemas import PassportData, Sex

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")


@dataclass
class FraudAnalysis:
    """Result of fraud analysis."""
    visual_data: Dict[str, Any]  # What's printed on passport face
    mrz_data: Dict[str, Any]     # What's encoded in MRZ
    fraud_flags: List[str]
    confidence: float
    is_suspicious: bool
    reasoning: str


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def extract_visual_data(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract data from the VISUAL text on passport face using LLM Vision.

    This reads what's actually printed on the passport, NOT the MRZ.
    """
    if not check_ollama_available():
        logger.warning("Ollama not available for visual extraction")
        return None

    try:
        # Handle PDF
        actual_path = image_path
        if image_path.suffix.lower() == ".pdf":
            from app.utils.pdf_utils import pdf_to_images
            images = pdf_to_images(image_path, dpi=300)
            if images:
                actual_path = images[0]
            else:
                return None

        with open(actual_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """You are an OCR system performing text extraction on a sample identity document image for a document processing application.

Extract the following text fields from this document image and return as JSON:

FIELDS TO EXTRACT:
- surname: The family/last name
- given_names: First and middle names
- nationality: Country name
- date_of_birth: Convert to YYYY-MM-DD format
- sex: M or F
- place_of_birth: Location text
- issue_date: Convert to YYYY-MM-DD format
- expiry_date: Convert to YYYY-MM-DD format
- passport_number: The document number (typically 9 characters)
- country_of_issue: Issuing country

RULES:
1. Read the printed text labels, not the machine-readable zone at bottom
2. Convert all dates to YYYY-MM-DD format
3. Use null for any field you cannot read clearly
4. Return ONLY valid JSON, no explanation

OUTPUT FORMAT:
{"surname": "...", "given_names": "...", "nationality": "...", "date_of_birth": "YYYY-MM-DD", "sex": "M/F", "place_of_birth": "...", "issue_date": "YYYY-MM-DD", "expiry_date": "YYYY-MM-DD", "passport_number": "...", "country_of_issue": "..."}"""

        # Log the prompt being sent
        logger.info("=" * 60)
        logger.info("LLM VISION REQUEST")
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
            timeout=120
        )

        if response.status_code != 200:
            logger.error(f"Vision extraction failed: {response.status_code}")
            return None

        result = response.json()
        raw_text = result.get("response", "")

        # Log the raw response
        logger.info("=" * 60)
        logger.info("LLM VISION RESPONSE (RAW)")
        logger.info("=" * 60)
        logger.info(f"{raw_text}")
        logger.info("=" * 60)

        # Check for model refusal patterns
        refusal_patterns = [
            "unable to assist",
            "cannot assist",
            "can't assist",
            "cannot read",
            "can't read",
            "sensitive document",
            "personal document",
            "i'm a large language model",
            "i am a large language model",
            "don't have the capability",
            "cannot visually examine",
            "cannot process",
        ]
        raw_lower = raw_text.lower()
        if any(pattern in raw_lower for pattern in refusal_patterns):
            logger.warning("LLM refused to process passport image (safety guardrails)")
            logger.warning("Falling back to MRZ-only extraction")
            return None

        # Extract JSON
        json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Try to find JSON in markdown block
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in raw_text:
            json_str = raw_text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)

        logger.warning("No JSON found in LLM response")
        return None

    except Exception as e:
        logger.error(f"Visual extraction failed: {e}")
        return None


def extract_mrz_data(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract data from MRZ using OCR.

    Returns parsed MRZ fields.
    """
    try:
        from app.extraction.ocr_service import extract_mrz

        mrz_result = extract_mrz(image_path)
        if not mrz_result:
            return None

        line1 = mrz_result.line1.upper().replace(" ", "").ljust(44, "<")[:44]
        line2 = mrz_result.line2.upper().replace(" ", "").ljust(44, "<")[:44]

        # Parse line1
        names = line1[5:44].split("<<", 1)
        surname = names[0].replace("<", "").strip()
        given_names = names[1].replace("<", " ").strip() if len(names) > 1 else ""

        # Parse line2
        passport_number = line2[0:9].replace("<", "").strip()
        nationality = line2[10:13]
        dob_raw = line2[13:19]  # YYMMDD
        sex = line2[20]
        expiry_raw = line2[21:27]  # YYMMDD

        # Convert MRZ dates to ISO format
        def mrz_date_to_iso(mrz_date: str, is_expiry: bool = False) -> Optional[str]:
            if len(mrz_date) != 6:
                return None
            try:
                yy = int(mrz_date[0:2])
                mm = int(mrz_date[2:4])
                dd = int(mrz_date[4:6])

                if is_expiry:
                    year = 2000 + yy
                else:
                    year = 1900 + yy if yy > 50 else 2000 + yy

                return f"{year:04d}-{mm:02d}-{dd:02d}"
            except:
                return None

        return {
            "surname": surname,
            "given_names": given_names,
            "passport_number": passport_number,
            "nationality": nationality,
            "date_of_birth": mrz_date_to_iso(dob_raw, False),
            "sex": sex,
            "expiry_date": mrz_date_to_iso(expiry_raw, True),
            "country_of_issue": line1[2:5],
            "_raw_line1": line1,
            "_raw_line2": line2,
            "_dob_raw": dob_raw,
            "_expiry_raw": expiry_raw,
        }

    except Exception as e:
        logger.error(f"MRZ extraction failed: {e}")
        return None


def compare_dates(visual_date: str, mrz_date: str, field_name: str) -> Optional[str]:
    """
    Compare dates from visual and MRZ extraction.

    Returns fraud flag if they don't match, None otherwise.
    """
    if not visual_date or not mrz_date:
        return None

    try:
        # Normalize dates
        v_date = datetime.strptime(visual_date, "%Y-%m-%d").date()
        m_date = datetime.strptime(mrz_date, "%Y-%m-%d").date()

        if v_date != m_date:
            return f"{field_name}_MISMATCH: Visual ({visual_date}) vs MRZ ({mrz_date})"
    except:
        pass

    return None


def compare_fields(visual: Dict[str, Any], mrz: Dict[str, Any]) -> List[str]:
    """
    Compare visual and MRZ data, return list of fraud flags.
    """
    fraud_flags = []

    # Compare dates (most important for fraud detection)
    dob_flag = compare_dates(
        visual.get("date_of_birth"),
        mrz.get("date_of_birth"),
        "DOB"
    )
    if dob_flag:
        fraud_flags.append(dob_flag)

    expiry_flag = compare_dates(
        visual.get("expiry_date"),
        mrz.get("expiry_date"),
        "EXPIRY_DATE"
    )
    if expiry_flag:
        fraud_flags.append(expiry_flag)

    # Compare passport number
    v_passport = (visual.get("passport_number") or "").replace(" ", "").upper()
    m_passport = (mrz.get("passport_number") or "").replace("<", "").upper()
    if v_passport and m_passport and v_passport != m_passport:
        fraud_flags.append(f"PASSPORT_NUMBER_MISMATCH: Visual ({v_passport}) vs MRZ ({m_passport})")

    # Compare sex
    v_sex = (visual.get("sex") or "").upper()[:1]
    m_sex = (mrz.get("sex") or "").upper()[:1]
    if v_sex and m_sex and v_sex != m_sex:
        fraud_flags.append(f"SEX_MISMATCH: Visual ({v_sex}) vs MRZ ({m_sex})")

    # Compare names (fuzzy - allow for OCR variations)
    v_surname = (visual.get("surname") or "").upper().replace(" ", "")
    m_surname = (mrz.get("surname") or "").upper().replace("<", "")

    # Remove common OCR garbage
    v_surname_clean = re.sub(r'[^A-Z]', '', v_surname)
    m_surname_clean = re.sub(r'[^A-Z]', '', m_surname)

    if v_surname_clean and m_surname_clean:
        # Check if one is substring of other (handles "DE LA PAZ" vs "DELAPAZ")
        if v_surname_clean not in m_surname_clean and m_surname_clean not in v_surname_clean:
            # Calculate similarity
            similarity = _string_similarity(v_surname_clean, m_surname_clean)
            if similarity < 0.6:
                fraud_flags.append(f"SURNAME_MISMATCH: Visual ({visual.get('surname')}) vs MRZ ({mrz.get('surname')})")

    # If we have date mismatches, add potential forgery flag
    if dob_flag or expiry_flag:
        fraud_flags.append("POTENTIAL_FORGERY")

    return fraud_flags


def _string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity (0-1)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    # Simple character overlap
    common = sum(1 for c in s1 if c in s2)
    return common / max(len(s1), len(s2))


def analyze_passport(image_path: Path) -> FraudAnalysis:
    """
    Main entry point: Analyze passport for fraud.

    1. Extract visual text (what's printed)
    2. Extract MRZ data
    3. Compare and flag discrepancies
    4. Return visual data (what humans see) with fraud flags
    """
    image_path = Path(image_path)

    # Extract both sources
    visual_data = extract_visual_data(image_path)
    mrz_data = extract_mrz_data(image_path)

    fraud_flags = []
    reasoning = ""

    if not visual_data and not mrz_data:
        return FraudAnalysis(
            visual_data={},
            mrz_data={},
            fraud_flags=["EXTRACTION_FAILED"],
            confidence=0.0,
            is_suspicious=False,
            reasoning="Could not extract data from passport"
        )

    if not visual_data:
        # Fall back to MRZ only (can't do fraud detection)
        return FraudAnalysis(
            visual_data=mrz_data or {},
            mrz_data=mrz_data or {},
            fraud_flags=["VISUAL_EXTRACTION_FAILED"],
            confidence=0.7,
            is_suspicious=False,
            reasoning="Could not extract visual text, using MRZ only"
        )

    if not mrz_data:
        # Visual only (can't validate)
        return FraudAnalysis(
            visual_data=visual_data,
            mrz_data={},
            fraud_flags=["MRZ_EXTRACTION_FAILED"],
            confidence=0.6,
            is_suspicious=False,
            reasoning="Could not extract MRZ, using visual text only"
        )

    # Compare visual and MRZ
    fraud_flags = compare_fields(visual_data, mrz_data)

    is_suspicious = len(fraud_flags) > 0

    if is_suspicious:
        reasoning = f"FRAUD DETECTED: Visual data does not match MRZ. Flags: {', '.join(fraud_flags)}"
        confidence = 0.95  # High confidence in the fraud detection
    else:
        reasoning = "Visual data matches MRZ - document appears authentic"
        confidence = 0.99

    return FraudAnalysis(
        visual_data=visual_data,
        mrz_data=mrz_data,
        fraud_flags=fraud_flags,
        confidence=confidence,
        is_suspicious=is_suspicious,
        reasoning=reasoning
    )


def extract_with_fraud_detection(image_path: Path) -> Optional[PassportData]:
    """
    Extract passport data with fraud detection.

    Returns PassportData with:
    - Visual data (what's printed) as primary
    - Fraud flags if visual != MRZ
    - High confidence extraction
    """
    analysis = analyze_passport(image_path)

    if not analysis.visual_data:
        return None

    visual = analysis.visual_data

    # Parse dates
    def parse_date(date_str: str) -> date:
        if not date_str:
            return date(1900, 1, 1)
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            return date(1900, 1, 1)

    # Parse sex
    def parse_sex(sex_str: str) -> Sex:
        sex_str = (sex_str or "").upper()[:1]
        if sex_str == "M":
            return Sex.MALE
        elif sex_str == "F":
            return Sex.FEMALE
        return Sex.OTHER

    # Build extraction method string
    method = "visual_fraud_detector"
    if analysis.fraud_flags:
        method += f"_flagged"

    try:
        return PassportData(
            surname=visual.get("surname", ""),
            given_names=visual.get("given_names", ""),
            passport_number=visual.get("passport_number", ""),
            nationality=visual.get("nationality", ""),
            date_of_birth=parse_date(visual.get("date_of_birth")),
            sex=parse_sex(visual.get("sex")),
            expiry_date=parse_date(visual.get("expiry_date")),
            country_of_issue=visual.get("country_of_issue", ""),
            place_of_birth=visual.get("place_of_birth"),
            issue_date=parse_date(visual.get("issue_date")) if visual.get("issue_date") else None,
            extraction_method=method,
            confidence_score=analysis.confidence,
        )
    except Exception as e:
        logger.error(f"Failed to create PassportData: {e}")
        return None
