"""
Tesseract OCR implementations with different configurations.
"""
import cv2
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path: str, method: str = "otsu") -> Image.Image:
    """
    Preprocesses image for OCR using OpenCV.
    """
    # Fix: Ensure path is string for cv2.imread
    img = cv2.imread(str(image_path))
    if img is None:
        # This was the error trigger before - now avoiding it by passing images only
        raise ValueError(f"Could not read image: {image_path}")

    processed = None

    if method == "upscale_otsu":
        # STRATEGY: 2x Upscaling + Otsu
        # Using specific dsize args to avoid ambiguity
        height, width = img.shape[:2]
        img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    else:
        # Default: Standard Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(processed)


def extract_with_tesseract_config(
    image_path: Path, 
    psm_mode: int = 6, 
    preprocess_method: str = "otsu"
) -> Optional[object]:
    try:
        processed_img = preprocess_image(str(image_path), method=preprocess_method)
        custom_config = f'--oem 3 --psm {psm_mode}'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Import internally to avoid circular dependency
        from app.extraction.passport import detect_mrz_lines, extract_from_mrz_lines, validate_mrz_checksum
        
        mrz_lines = detect_mrz_lines(text)
        
        if mrz_lines:
            line1, line2 = mrz_lines
            result = extract_from_mrz_lines(
                line1, 
                line2, 
                method=f"tesseract_psm{psm_mode}_{preprocess_method}"
            )
            
            if validate_mrz_checksum(line2):
                return result
                
        return None

    except Exception as e:
        logger.debug(f"OCR Config Failed ({psm_mode}, {preprocess_method}): {e}")
        return None