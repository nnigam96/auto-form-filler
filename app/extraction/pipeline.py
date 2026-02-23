"""
V5: Production Pipeline with Human-In-The-Loop (HITL)

Flow:
1. Parallel OCR (PassportEye, Tesseract, EasyOCR) → MRZ data
2. Field-level aggregation
3. LLM Vision → Visual text (if enabled)
4. Cross-validate MRZ vs Visual
5. If aligned → High confidence output
6. If misaligned → HITL: Return both values for user correction

Key insight:
- MRZ with valid checksum = cryptographically verified
- Visual text = what humans see
- Mismatch = either OCR error OR document tampering
- Let the human decide when uncertain
"""

import asyncio
import logging
import warnings
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END

from app.extraction.state import PassportState
from app.extraction.ocr_engines import run_passport_eye, run_tesseract, run_easyocr
from app.extraction.aggregator import aggregate_ocr_results
from app.extraction.voting import critic_validate

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class FieldResult:
    """Result for a single field, showing MRZ vs Visual values."""
    field_name: str
    mrz_value: Any
    visual_value: Any
    final_value: Any
    confidence: float
    needs_review: bool
    source: str  # "mrz", "visual", "aligned", "user"


@dataclass
class ExtractionResult:
    """Complete extraction result with HITL support."""
    success: bool
    fields: Dict[str, FieldResult]
    overall_confidence: float
    needs_human_review: bool
    fraud_flags: List[str]
    review_reason: Optional[str]
    mrz_checksum_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "fields": {k: asdict(v) for k, v in self.fields.items()},
            "overall_confidence": self.overall_confidence,
            "needs_human_review": self.needs_human_review,
            "fraud_flags": self.fraud_flags,
            "review_reason": self.review_reason,
            "mrz_checksum_valid": self.mrz_checksum_valid,
        }

    def get_final_data(self) -> Dict[str, Any]:
        """Get final values for all fields."""
        return {k: v.final_value for k, v in self.fields.items()}


# =============================================================================
# Pipeline Nodes
# =============================================================================

async def node_parallel_ocr(state: PassportState) -> PassportState:
    """Step 1: Run 3 OCR engines in parallel."""
    file_path = state.get("image_path")
    logger.info(f"[V5] Parallel OCR: {file_path}")

    results = await asyncio.gather(
        run_passport_eye(file_path),
        run_tesseract(file_path),
        run_easyocr(file_path),
        return_exceptions=True,
    )

    ocr_results = []
    for i, r in enumerate(results):
        source = ["passport_eye", "tesseract", "easyocr"][i]
        if isinstance(r, Exception):
            ocr_results.append({"source": source, "success": False, "error": str(r)})
        else:
            ocr_results.append(r)

    # Log results
    for r in ocr_results:
        status = "✓" if r.get("success") else "✗"
        checksum = "✓" if r.get("checksum_valid") else "✗"
        logger.info(f"  {r.get('source')}: {status} checksum={checksum}")

    return {
        **state,
        "ocr_results": ocr_results,
        "errors": [r.get("error") for r in ocr_results if r.get("error")],
    }


async def node_aggregate_mrz(state: PassportState) -> PassportState:
    """Step 2: Aggregate MRZ results."""
    ocr_results = state.get("ocr_results", [])

    if not ocr_results:
        return {**state, "mrz_data": None, "has_valid_checksum": False}

    has_valid_checksum = any(
        r.get("checksum_valid") for r in ocr_results if r.get("success")
    )

    # Extract MRZ lines
    mrz_lines = None
    for r in ocr_results:
        if r.get("success") and r.get("raw_text"):
            mrz_lines = _extract_mrz_lines(r.get("raw_text", ""))
            if mrz_lines:
                break

    # Aggregate
    aggregated = aggregate_ocr_results(ocr_results, mrz_lines)
    mrz_data = {name: af.value for name, af in aggregated.fields.items()}

    logger.info(f"[V5] MRZ aggregated: checksum_valid={has_valid_checksum}")

    return {
        **state,
        "mrz_data": mrz_data,
        "mrz_lines": mrz_lines,
        "has_valid_checksum": has_valid_checksum,
        "mrz_confidence": aggregated.overall_confidence,
    }


async def node_visual_extraction(state: PassportState) -> PassportState:
    """Step 3: Extract visual text using LLM Vision."""
    if not state.get("use_llm", True):
        logger.info("[V5] Visual extraction disabled")
        return {**state, "visual_data": None}

    file_path = state.get("image_path")

    from pathlib import Path
    from app.extraction.fraud_detector import extract_visual_data

    loop = asyncio.get_event_loop()
    try:
        visual_data = await loop.run_in_executor(
            _executor, extract_visual_data, Path(file_path)
        )
        logger.info(f"[V5] Visual extraction: {'success' if visual_data else 'failed'}")
    except Exception as e:
        logger.warning(f"[V5] Visual extraction error: {e}")
        visual_data = None

    return {**state, "visual_data": visual_data}


async def node_compare_and_decide(state: PassportState) -> PassportState:
    """Step 4: Compare MRZ vs Visual, decide if HITL needed."""
    mrz_data = state.get("mrz_data") or {}
    visual_data = state.get("visual_data") or {}
    has_valid_checksum = state.get("has_valid_checksum", False)

    fields = {}
    fraud_flags = []
    needs_review = False
    review_reasons = []

    # Define all fields to check
    field_names = [
        "surname", "given_names", "passport_number", "nationality",
        "date_of_birth", "sex", "expiry_date", "country",
        "place_of_birth", "issue_date"
    ]

    for field_name in field_names:
        mrz_val = mrz_data.get(field_name)
        visual_val = visual_data.get(field_name) if visual_data else None

        # Normalize for comparison
        mrz_norm = _normalize(mrz_val)
        visual_norm = _normalize(visual_val)

        # Determine alignment and final value
        if not mrz_val and not visual_val:
            # Neither has value
            fields[field_name] = FieldResult(
                field_name=field_name,
                mrz_value=None,
                visual_value=None,
                final_value=None,
                confidence=0.0,
                needs_review=False,
                source="none"
            )
        elif not visual_val:
            # Only MRZ has value - use it
            fields[field_name] = FieldResult(
                field_name=field_name,
                mrz_value=mrz_val,
                visual_value=None,
                final_value=mrz_val,
                confidence=0.95 if has_valid_checksum else 0.7,
                needs_review=False,
                source="mrz"
            )
        elif not mrz_val:
            # Only visual has value (e.g., place_of_birth, issue_date)
            fields[field_name] = FieldResult(
                field_name=field_name,
                mrz_value=None,
                visual_value=visual_val,
                final_value=visual_val,
                confidence=0.8,
                needs_review=False,
                source="visual"
            )
        elif mrz_norm == visual_norm:
            # Aligned - high confidence
            fields[field_name] = FieldResult(
                field_name=field_name,
                mrz_value=mrz_val,
                visual_value=visual_val,
                final_value=mrz_val,  # Use MRZ as canonical
                confidence=0.99,
                needs_review=False,
                source="aligned"
            )
        else:
            # MISMATCH - needs human review
            needs_review = True
            review_reasons.append(f"{field_name}: MRZ≠Visual")

            # Check if it's likely LLM hallucination
            is_hallucination = _is_likely_hallucination(field_name, mrz_val, visual_val)

            if is_hallucination:
                # Trust MRZ, but still flag for awareness
                fields[field_name] = FieldResult(
                    field_name=field_name,
                    mrz_value=mrz_val,
                    visual_value=visual_val,
                    final_value=mrz_val,
                    confidence=0.85,
                    needs_review=True,
                    source="mrz_preferred"
                )
            else:
                # Real mismatch - could be fraud, needs human decision
                fraud_flags.append(
                    f"{field_name.upper()}_MISMATCH: MRZ({mrz_val}) vs Visual({visual_val})"
                )
                fields[field_name] = FieldResult(
                    field_name=field_name,
                    mrz_value=mrz_val,
                    visual_value=visual_val,
                    final_value=None,  # Human must decide
                    confidence=0.0,
                    needs_review=True,
                    source="conflict"
                )

    # Add fraud flag if significant mismatches
    if len([f for f in fields.values() if f.source == "conflict"]) > 0:
        fraud_flags.append("POTENTIAL_DOCUMENT_TAMPERING")

    # Calculate overall confidence
    confidences = [f.confidence for f in fields.values() if f.final_value is not None]
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Build review reason
    review_reason = "; ".join(review_reasons) if review_reasons else None

    logger.info(f"[V5] Comparison: needs_review={needs_review}, confidence={overall_confidence:.2f}")

    return {
        **state,
        "extraction_result": ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=overall_confidence,
            needs_human_review=needs_review,
            fraud_flags=fraud_flags,
            review_reason=review_reason,
            mrz_checksum_valid=has_valid_checksum,
        ),
        "needs_human_review": needs_review,
        "fraud_flags": fraud_flags,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_mrz_lines(text: str) -> Optional[List[str]]:
    """Extract MRZ lines from OCR text."""
    text = text.upper()
    lines = text.split('\n')

    candidates = []
    for line in lines:
        line = line.strip().replace(' ', '').replace('«', '<')
        if len(line) >= 40:
            mrz_chars = sum(1 for c in line if c.isalnum() or c == '<')
            if mrz_chars / len(line) > 0.9:
                candidates.append(line[:44])

    for i, line in enumerate(candidates):
        if line.startswith('P') and i + 1 < len(candidates):
            return [line, candidates[i + 1]]
    return None


def _normalize(value: Any) -> str:
    """Normalize value for comparison."""
    if value is None:
        return ""
    return str(value).upper().replace(" ", "").replace("-", "").replace("<", "")


def _is_likely_hallucination(field_name: str, mrz_val: Any, visual_val: Any) -> bool:
    """Detect if visual value is likely an LLM hallucination."""
    if not mrz_val or not visual_val:
        return False

    mrz_str = str(mrz_val)
    visual_str = str(visual_val)

    # Passport number: if lengths differ significantly, LLM hallucinated
    if field_name == "passport_number":
        if abs(len(mrz_str) - len(visual_str)) > 2:
            return True
        # If visual has X's or clearly fake pattern
        if "XXX" in visual_str or "000" in visual_str:
            return True

    # Dates: if formats are wildly different
    if field_name in ["date_of_birth", "expiry_date", "issue_date"]:
        # If one is clearly not a date format
        if len(visual_str) < 6:
            return True

    return False


# =============================================================================
# Build Graph
# =============================================================================

def build_v5_graph() -> StateGraph:
    """
    V5 Pipeline:

    parallel_ocr → aggregate_mrz → visual_extraction → compare_and_decide → END

    No complex routing - always runs full pipeline, lets compare_and_decide
    determine if HITL is needed.
    """
    workflow = StateGraph(PassportState)

    workflow.add_node("parallel_ocr", node_parallel_ocr)
    workflow.add_node("aggregate_mrz", node_aggregate_mrz)
    workflow.add_node("visual_extraction", node_visual_extraction)
    workflow.add_node("compare_and_decide", node_compare_and_decide)

    workflow.set_entry_point("parallel_ocr")

    # Simple linear flow
    workflow.add_edge("parallel_ocr", "aggregate_mrz")
    workflow.add_edge("aggregate_mrz", "visual_extraction")
    workflow.add_edge("visual_extraction", "compare_and_decide")
    workflow.add_edge("compare_and_decide", END)

    return workflow.compile()


# Compiled graph
graph_v5 = build_v5_graph()


# =============================================================================
# Convenience Functions
# =============================================================================

async def extract_passport_v5(
    file_path: str,
    use_llm: bool = True
) -> ExtractionResult:
    """
    Main entry point for V5 extraction.

    Returns ExtractionResult with:
    - fields: Dict of FieldResult with MRZ/Visual/Final values
    - needs_human_review: True if user should verify
    - fraud_flags: List of detected issues
    """
    initial_state: PassportState = {
        "image_path": file_path,
        "ocr_results": [],
        "final_data": None,
        "confidence": 0.0,
        "errors": [],
        "source": "v5",
        "needs_human_review": False,
        "fraud_flags": [],
        "use_llm": use_llm,
    }

    result = await graph_v5.ainvoke(initial_state)
    return result.get("extraction_result")
