"""
Evaluation logic for OCR configurations.
Handles PDF-to-Image conversion automatically before running tests.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.models.schemas import PassportData
from app.extraction.research.ocr_tesseract import extract_with_tesseract_config
# from app.extraction.research.ocr_easyocr import extract_with_easyocr_config 
from app.utils.pdf_utils import pdf_to_images  # CRITICAL IMPORT

logger = logging.getLogger(__name__)

def evaluate_all_configs(passport_path: Path, expected_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a battery of OCR configurations against a single passport.
    """
    results = {
        "total_configs_tested": 0,
        "successful_configs": 0,
        "successful_configs_detail": [],
        "all_metrics": [],
        "best_method": None,
        "summary": {"by_method": {}}
    }

    # --- CRITICAL FIX: CONVERT PDF TO IMAGE ---
    # cv2.imread cannot read PDFs. We must convert it first.
    test_image_path = passport_path
    temp_images = []

    if passport_path.suffix.lower() == ".pdf":
        print(f"Converting PDF to image for evaluation: {passport_path.name}")
        try:
            images = pdf_to_images(passport_path)
            if images:
                test_image_path = images[0]  # Use the first page
                temp_images = images # Keep track to clean up if needed
            else:
                print("Error: PDF conversion returned no images.")
                return results
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return results

    print(f"Running evaluation on image: {test_image_path.name}")

    # --- DEFINE CONFIGS TO TEST ---
    
    tesseract_configs = [
        # The Winner Candidate (Upscale 2x + Otsu)
        {"psm": 6, "preprocess": "upscale_otsu"}, 
        {"psm": 11, "preprocess": "upscale_otsu"},
        
        # Standard Candidates
        {"psm": 6, "preprocess": "otsu"},
        {"psm": 6, "preprocess": "adaptive"},
        {"psm": 11, "preprocess": "otsu"},
    ]

    # --- RUN EVALUATION ---

    results["summary"]["by_method"]["tesseract"] = {"total": 0, "successful": 0, "failed": 0, "avg_time_ms": 0}
    
    for config in tesseract_configs:
        results["total_configs_tested"] += 1
        results["summary"]["by_method"]["tesseract"]["total"] += 1
        
        # Pass the IMAGE path, not the PDF path
        metric = _test_single_config(
            "tesseract", 
            extract_with_tesseract_config, 
            test_image_path, 
            expected_data,
            psm_mode=config["psm"],
            preprocess_method=config["preprocess"]
        )
        
        results["all_metrics"].append(metric)
        if metric["success"]:
            results["successful_configs"] += 1
            results["summary"]["by_method"]["tesseract"]["successful"] += 1
            results["successful_configs_detail"].append(metric)
        else:
            results["summary"]["by_method"]["tesseract"]["failed"] += 1

    # Find best config
    if results["successful_configs_detail"]:
        best = sorted(results["successful_configs_detail"], key=lambda x: x["time_ms"])[0]
        results["best_method"] = best

    return results

def _test_single_config(method_name: str, extract_func, image_path: Path, expected: Dict, **kwargs) -> Dict:
    """Helper to run one specific configuration and measure performance."""
    
    config_str = "_".join([f"{k}{v}" for k, v in kwargs.items()])
    full_name = f"{method_name}_{config_str}"
    
    start_time = time.time()
    try:
        # Run Extraction
        result: Optional[PassportData] = extract_func(image_path, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        if result:
            is_match = _compare_result(result, expected)
            return {
                "method": method_name,
                "full_method_name": full_name,
                "config": kwargs,
                "time_ms": duration,
                "success": is_match,
                "checksum_passed": True,
                "extracted_data": result.dict()
            }
        else:
            return {
                "method": method_name,
                "full_method_name": full_name,
                "config": kwargs,
                "time_ms": duration,
                "success": False,
                "checksum_passed": False,
                "error": "No MRZ found or Checksum failed"
            }
            
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        return {
            "method": method_name,
            "full_method_name": full_name,
            "config": kwargs,
            "time_ms": duration,
            "success": False,
            "checksum_passed": False,
            "error": str(e)
        }

def find_best_ocr_config(passport_path: Path, expected: Dict) -> Optional[Dict]:
    results = evaluate_all_configs(passport_path, expected)
    return results["best_method"]

def _compare_result(result: PassportData, expected: Dict) -> bool:
    try:
        if result.passport_number.replace(" ", "") != expected["passport_number"]: return False
        # Add tolerance for dates if needed, or check strict equality
        if result.date_of_birth != expected["date_of_birth"]: return False
        return True
    except Exception:
        return False