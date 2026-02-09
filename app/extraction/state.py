"""
State definition for the passport extraction agent graph.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class PassportState(TypedDict):
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
