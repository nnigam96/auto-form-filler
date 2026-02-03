"""
Robust EasyOCR implementation for MRZ extraction.
"""

import logging
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from app.models.schemas import PassportData
from app.extraction.passport import (
    extract_from_mrz_lines,
    validate_mrz_checksum,
)
# Ensure this import exists or define the dict locally if missing
from app.extraction.research.ocr_preprocessing import PREPROCESSING_METHODS

logger = logging.getLogger(__name__)

# Lazy-loaded EasyOCR reader
_easyocr_reader = None

def get_easyocr_reader():
    """Get or initialize EasyOCR reader (lazy, cached)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            import torch
            # Use MPS (Metal) on Mac if available
            use_gpu = False
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    logger.info("EasyOCR using MPS (Metal GPU)")
                elif torch.cuda.is_available():
                    use_gpu = True
                    logger.info("EasyOCR using CUDA GPU")
                else:
                    logger.info("EasyOCR using CPU")
            except:
                logger.info("EasyOCR using CPU (GPU check failed)")
            
            # Initialize once
            _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            return None
        except Exception as e:
            logger.warning(f"EasyOCR setup failed: {e}")
            return None
    return _easyocr_reader


def extract_mrz_with_easyocr(
    image_path: Path,
    preprocess_method: str = "none",
) -> Optional[Tuple[str, str]]:
    """
    Extract MRZ lines using EasyOCR with specific preprocessing.
    """
    try:
        reader = get_easyocr_reader()
        if reader is None:
            return None
        
        # 1. Image Loading Strategy
        if preprocess_method == "none":
            # Pass path directly (EasyOCR handles loading efficiently)
            image_input = str(image_path)
        else:
            # Apply custom preprocessing
            preprocess_func = PREPROCESSING_METHODS.get(preprocess_method)
            if not preprocess_func:
                logger.warning(f"Unknown preprocessing method: {preprocess_method}")
                image_input = str(image_path)
            else:
                pil_img = preprocess_func(str(image_path))
                # Convert PIL (RGB) to OpenCV (BGR) format for EasyOCR
                image_input = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 2. Run EasyOCR with STRICT Allowlist
        # This is the "magic fix" - forces model to ignore background noise
        results = reader.readtext(
            image_input,
            detail=0,
            paragraph=False, # Line-by-line is better for MRZ
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        )
        
        # 3. Robust Parsing
        candidates = []
        for line in results:
            # Aggressively clean spaces (common OCR error in MRZ)
            clean_line = line.replace(" ", "").upper()
            
            # Filter logic:
            # - Length between 30-55 (TD3 is 44, but we allow noise)
            # - Must contain filler characters '<<' OR start with standard tags
            if 30 < len(clean_line) < 55:
                if "<<" in clean_line or clean_line.startswith(("P<", "I<", "A<")):
                    candidates.append(clean_line)
        
        logger.info(f"EasyOCR ({preprocess_method}) candidates: {candidates}")

        if len(candidates) >= 2:
            # Return the last two lines found (MRZ is at the bottom)
            return candidates[-2], candidates[-1]
        
        return None
        
    except Exception as e:
        logger.debug(f"EasyOCR ({preprocess_method}) failed: {e}")
        return None


def extract_with_easyocr_config(
    image_path: Path,
    preprocess_method: str = "none",
) -> Optional[PassportData]:
    """
    Extract passport data using EasyOCR with specific preprocessing.
    """
    mrz_lines = extract_mrz_with_easyocr(image_path, preprocess_method)
    
    if not mrz_lines:
        return None
    
    line1, line2 = mrz_lines
    
    # Validate checksum
    # Note: We use the fuzzy validator we wrote earlier to be safe
    if not validate_mrz_checksum(line2):
        logger.debug(f"EasyOCR ({preprocess_method}): Checksum failed on {line2}")
        return None
    
    # Extract data
    method_name = f"easyocr_{preprocess_method}"
    return extract_from_mrz_lines(line1, line2, method_name)