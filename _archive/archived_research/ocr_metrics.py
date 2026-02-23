"""
OCR metrics tracking for evaluation.

Tracks performance of different OCR configurations to identify
the best approach for production use.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OCRMetrics:
    """Track OCR method performance."""
    method: str
    config: Dict[str, any]  # e.g., {"psm_mode": 6, "preprocess": "otsu"}
    success: bool
    checksum_passed: bool
    mrz_lines_found: int
    confidence_score: float = 0.0
    extraction_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# Global metrics storage
_ocr_metrics: List[OCRMetrics] = []


def record_ocr_attempt(metrics: OCRMetrics):
    """Record an OCR attempt for metrics tracking."""
    _ocr_metrics.append(metrics)
    logger.debug(f"OCR Metrics: {metrics.method} - Success: {metrics.success}, Checksum: {metrics.checksum_passed}")


def get_ocr_metrics() -> List[OCRMetrics]:
    """Get accumulated OCR metrics."""
    return _ocr_metrics.copy()


def clear_ocr_metrics():
    """Clear accumulated metrics (useful for testing)."""
    global _ocr_metrics
    _ocr_metrics = []


def get_metrics_summary() -> Dict:
    """Get summary statistics of OCR metrics."""
    if not _ocr_metrics:
        return {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0.0,
            "by_method": {},
        }
    
    successful = [m for m in _ocr_metrics if m.success]
    failed = [m for m in _ocr_metrics if not m.success]
    
    # Group by method
    by_method = {}
    for m in _ocr_metrics:
        method_name = m.method.split('_')[0]  # e.g., "tesseract" or "easyocr"
        if method_name not in by_method:
            by_method[method_name] = {"total": 0, "successful": 0, "failed": 0, "avg_time_ms": 0.0}
        by_method[method_name]["total"] += 1
        if m.success:
            by_method[method_name]["successful"] += 1
            by_method[method_name]["avg_time_ms"] += m.extraction_time_ms
        else:
            by_method[method_name]["failed"] += 1
    
    # Calculate averages
    for method_name in by_method:
        successful_count = by_method[method_name]["successful"]
        if successful_count > 0:
            by_method[method_name]["avg_time_ms"] /= successful_count
    
    return {
        "total_attempts": len(_ocr_metrics),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(_ocr_metrics) if _ocr_metrics else 0.0,
        "by_method": by_method,
    }


def get_best_method() -> Optional[str]:
    """
    Determine the best OCR method based on metrics.
    
    Returns the method name with highest success rate and fastest time.
    """
    if not _ocr_metrics:
        return None
    
    # Group by full method name (including config)
    method_stats = {}
    for m in _ocr_metrics:
        if m.method not in method_stats:
            method_stats[m.method] = {
                "successful": 0,
                "total": 0,
                "total_time": 0.0,
            }
        method_stats[m.method]["total"] += 1
        if m.success:
            method_stats[m.method]["successful"] += 1
            method_stats[m.method]["total_time"] += m.extraction_time_ms
    
    # Find best: highest success rate, then fastest
    best_method = None
    best_score = -1
    
    for method, stats in method_stats.items():
        success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        avg_time = stats["total_time"] / stats["successful"] if stats["successful"] > 0 else float('inf')
        
        # Score: success rate weighted more, but prefer faster methods
        score = success_rate * 100 - (avg_time / 10)  # Time penalty
        
        if score > best_score:
            best_score = score
            best_method = method
    
    return best_method

