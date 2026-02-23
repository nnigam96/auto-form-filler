"""
LLM Vision extraction using Ollama (local) or OpenAI (cloud).

Recommended models:
- llama3.2-vision (default) - Best quality for document reading
- llava:13b - Alternative, slower
- llava:7b - Faster but less accurate

Usage:
    # Install Ollama: brew install ollama
    # Pull model: ollama pull llama3.2-vision
    # Start server: ollama serve
"""

import base64
import logging
import json
import os
import re
import requests
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from app.models.schemas import PassportData, Sex

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except:
        pass
    return []


def parse_llm_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse date string from LLM response into date object.
    
    Handles multiple formats:
    - DD/MM/YYYY (e.g., '15/02/1996')
    - MM/DD/YYYY (e.g., '02/15/1996')
    - YYYY-MM-DD (e.g., '1996-02-15')
    - YYYY/MM/DD (e.g., '1996/02/15')
    """
    if not date_str:
        return None
    
    if isinstance(date_str, date):
        return date_str
    
    if isinstance(date_str, datetime):
        return date_str.date()
    
    date_str = str(date_str).strip()
    
    # Try ISO format first (YYYY-MM-DD)
    try:
        return datetime.fromisoformat(date_str).date()
    except (ValueError, AttributeError):
        pass
    
    # Try DD/MM/YYYY or MM/DD/YYYY
    # Use heuristics: if first part > 12, assume DD/MM/YYYY
    # Otherwise, try both formats
    slash_match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
    if slash_match:
        part1, part2, year = map(int, slash_match.groups())
        
        # Heuristic: if first part > 12, it's DD/MM/YYYY (day > 12)
        if part1 > 12:
            # DD/MM/YYYY: day=part1, month=part2, year=year
            return date(year, part2, part1)
        # If second part > 12, it's MM/DD/YYYY (day > 12)
        elif part2 > 12:
            # MM/DD/YYYY: month=part1, day=part2, year=year
            return date(year, part1, part2)
        # Ambiguous (both parts <= 12): try DD/MM/YYYY first (more common in international formats)
        else:
            try:
                # Try DD/MM/YYYY: day=part1, month=part2
                return date(year, part2, part1)
            except ValueError:
                # If that fails (invalid date), try MM/DD/YYYY
                try:
                    return date(year, part1, part2)
                except ValueError:
                    # Both failed, return None
                    logger.warning(f"Could not parse ambiguous date: {date_str}")
                    return None
    
    # Try other common formats
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%d.%m.%Y',
        '%m.%d.%Y',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_str}")
    return None

def encode_image(image_path: Path) -> str:
    """Encodes an image to base64 string. Converts PDFs to images first (uses first page)."""
    # Convert PDF to image if needed
    actual_path = image_path
    if image_path.suffix.lower() == '.pdf':
        try:
            from app.utils.pdf_utils import pdf_to_images
            images = pdf_to_images(image_path, dpi=300)
            if images:
                actual_path = images[0]  # LLM Vision uses first page
            else:
                raise ValueError("PDF conversion returned no images")
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            raise
    
    with open(actual_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_with_ollama(image_path: Path) -> Optional[PassportData]:
    """
    Uses Ollama with llama3.2-vision model to extract passport data locally.
    This handles rotated, blurry, or non-standard passports.

    Requires:
        - Ollama installed: brew install ollama
        - Model pulled: ollama pull llama3.2-vision
        - Server running: ollama serve
    """
    if not check_ollama_available():
        logger.error("Ollama server not available. Run 'ollama serve' first.")
        return None

    try:
        base64_image = encode_image(image_path)

        logger.info(f"Sending image to Ollama ({OLLAMA_MODEL}) for extraction: {image_path.name}")

        prompt = """You are an expert OCR system for passports. Analyze this passport image and extract the following fields into JSON format:

- surname: Family name
- given_names: First and middle names
- passport_number: The passport number
- nationality: 3-letter country code (e.g., IND, USA)
- date_of_birth: In YYYY-MM-DD format
- sex: M, F, or X
- expiry_date: In YYYY-MM-DD format
- country_of_issue: 3-letter country code
- place_of_birth: City/state if visible
- issue_date: In YYYY-MM-DD format if visible

Return ONLY valid JSON, no other text. Example:
{"surname": "SMITH", "given_names": "JOHN", "passport_number": "AB123456", "nationality": "USA", "date_of_birth": "1990-05-15", "sex": "M", "expiry_date": "2030-05-14", "country_of_issue": "USA"}"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Deterministic output
                }
            },
            timeout=120  # Vision models can be slow
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None

        result = response.json()
        content = result.get("response", "")

        if not content:
            logger.error("Empty response from Ollama")
            return None

        # Extract JSON from response (LLaVA sometimes adds explanation text)
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if not json_match:
            logger.error(f"Could not find JSON in Ollama response: {content[:200]}")
            return None

        data = json.loads(json_match.group())

        # Parse dates from LLM response (handles various formats)
        date_of_birth = parse_llm_date(data.get("date_of_birth"))
        expiry_date = parse_llm_date(data.get("expiry_date"))
        issue_date = parse_llm_date(data.get("issue_date"))

        # Parse sex - handle string or enum
        sex_value = data.get("sex", "X")
        if isinstance(sex_value, str):
            sex_value = sex_value.upper()
            if sex_value not in ["M", "F", "X"]:
                sex_value = "X"

        # Map JSON to Pydantic Model
        return PassportData(
            surname=data.get("surname", ""),
            given_names=data.get("given_names", ""),
            passport_number=data.get("passport_number", ""),
            nationality=data.get("nationality", ""),
            date_of_birth=date_of_birth,
            sex=sex_value,
            expiry_date=expiry_date,
            country_of_issue=data.get("country_of_issue", ""),
            place_of_birth=data.get("place_of_birth"),
            issue_date=issue_date,
            extraction_method="ollama_vision",
            confidence_score=0.90
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Ollama response: {e}")
        return None
    except Exception as e:
        logger.error(f"Ollama extraction failed: {e}")
        return None


def extract_with_llm_vision(image_path: Path) -> Optional[PassportData]:
    """
    Main entry point for LLM-based vision extraction.
    Uses Ollama (local) by default for privacy and cost savings.
    """
    return extract_with_ollama(image_path)