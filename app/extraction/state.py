"""
State definition for the passport extraction agent graph.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from app.extraction.pipeline import ExtractionResult


class PassportState(TypedDict, total=False):
    """State passed between nodes in the extraction graph."""

    # Input
    image_path: str

    # OCR outputs (raw dicts from each tool)
    ocr_results: List[dict]

    # Final extracted data (dict, converted to Pydantic at the end)
    final_data: Optional[dict]

    # Confidence score (0.0 - 1.0)
    confidence: float

    # Errors encountered
    errors: List[str]

    # Which method produced the final result
    source: str

    # Flags for routing
    needs_human_review: bool
    fraud_flags: List[str]

    # LLM settings
    use_llm: bool

    # V3: Field aggregation
    low_confidence_fields: List[str]

    # V4: Fraud detection
    visual_data: Optional[Dict[str, Any]]  # Data from LLM Vision (what's printed)
    mrz_data: Optional[Dict[str, Any]]     # Data from MRZ OCR

    # V5: HITL result
    extraction_result: Optional[Any]  # ExtractionResult from graph_v5
    mrz_lines: Optional[List[str]]
    has_valid_checksum: bool
    mrz_confidence: float
