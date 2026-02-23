"""
Production OCR implementation using the best evaluated configuration.

This module uses the OCR configuration that performed best during evaluation.
The best config can be updated after running ocr_eval.py.
"""

import logging
from pathlib import Path
from typing import Optional

from app.models.schemas import PassportData
from app.extraction.research.ocr_tesseract import extract_with_tesseract_config
from app.extraction.research.ocr_easyocr import extract_with_easyocr_config

logger = logging.getLogger(__name__)

# Best OCR configuration (determined by evaluation)
# Update this after running ocr_eval.py to find the best config
BEST_OCR_CONFIG = {
    "method": "tesseract",  # or "easyocr"
    "psm_mode": 6,  # Only used for tesseract
    "preprocess": "otsu",  # Preprocessing method
}

# Fallback configs to try if best fails
FALLBACK_CONFIGS = [
    {"method": "tesseract", "psm_mode": 11, "preprocess": "otsu"},
    {"method": "tesseract", "psm_mode": 6, "preprocess": "adaptive"},
    {"method": "easyocr", "preprocess": "none"},
]


def extract_with_best_ocr(image_path: Path) -> Optional[PassportData]:
    """
    Extract passport data using the best OCR configuration.
    
    Tries the best config first, then falls back to alternatives if needed.
    
    Args:
        image_path: Path to passport image
    
    Returns:
        PassportData if successful, None otherwise
    """
    # Try best config first
    configs_to_try = [BEST_OCR_CONFIG] + FALLBACK_CONFIGS
    
    for config in configs_to_try:
        method = config["method"]
        
        try:
            if method == "tesseract":
                result = extract_with_tesseract_config(
                    image_path,
                    psm_mode=config.get("psm_mode", 6),
                    preprocess_method=config.get("preprocess", "otsu"),
                )
            elif method == "easyocr":
                result = extract_with_easyocr_config(
                    image_path,
                    preprocess_method=config.get("preprocess", "none"),
                )
            else:
                continue
            
            if result:
                logger.info(f"OCR extraction successful with {method} ({config.get('preprocess', 'none')})")
                return result
                
        except Exception as e:
            logger.debug(f"OCR config {config} failed: {e}")
            continue
    
    logger.warning("All OCR configurations failed")
    return None


def update_best_config(method: str, config: dict):
    """
    Update the best OCR configuration.
    
    Call this after running evaluation to set the best performing config.
    
    Args:
        method: "tesseract" or "easyocr"
        config: Configuration dict (e.g., {"psm_mode": 6, "preprocess": "otsu"})
    """
    global BEST_OCR_CONFIG
    BEST_OCR_CONFIG = {
        "method": method,
        **config,
    }
    logger.info(f"Updated best OCR config: {BEST_OCR_CONFIG}")

