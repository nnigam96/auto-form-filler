import base64
import logging
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from app.models.schemas import PassportData, Sex

logger = logging.getLogger(__name__)

# Client will be initialized lazily when needed
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Get or initialize OpenAI client. Raises error if API key not configured."""
    global _client
    if _client is None:
        from app.config import settings
        # Try config first, then environment variable
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. LLM extraction requires an API key. "
                "Set OPENAI_API_KEY in .env file or set use_llm_extraction=False in config."
            )
        _client = OpenAI(api_key=api_key)
    return _client


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
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_with_llm_vision(image_path: Path) -> Optional[PassportData]:
    """
    Uses GPT-4o Vision to extract passport data when algorithmic methods fail.
    This handles rotated, blurry, or non-standard passports.
    """
    try:
        base64_image = encode_image(image_path)
        
        logger.info(f"Sending image to LLM for extraction: {image_path.name}")

        client = get_client()
        response = client.chat.completions.create(
            model="gpt-4o",  # Or "gpt-4-turbo"
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert OCR system for passports. Extract the data precisely into JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the following fields from this passport image: surname, given_names, passport_number, nationality, date_of_birth, sex, expiry_date, country_of_issue, place_of_birth, issue_date. Return ONLY valid JSON. For dates, use YYYY-MM-DD format (e.g., '1996-10-25'). For sex, use M, F, or X."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0, # Deterministic output
        )

        content = response.choices[0].message.content
        if not content:
            return None
            
        data = json.loads(content)
        
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
            extraction_method="llm_vision",
            confidence_score=0.95
        )

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return None