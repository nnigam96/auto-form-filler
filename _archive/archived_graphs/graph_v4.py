"""
LangGraph workflow V4: Fraud-Aware Extraction.

Flow:
1. Run 3 OCR engines in parallel (PassportEye, Tesseract, EasyOCR)
2. Aggregate MRZ results (field-level, best per field)
3. Use LLM Vision to READ visual text for comparison (not as primary source)
4. Cross-validate: If Visual != MRZ AND MRZ checksum valid → Visual is likely wrong OR document is forged
5. If issues detected → Reflection agent to fix
6. Return MRZ-validated data + fraud flags if visual doesn't match

Key insight: MRZ with valid checksum is CRYPTOGRAPHICALLY VERIFIED.
LLM Vision can hallucinate. Never trust LLM over checksum-validated MRZ.
"""

import asyncio
import logging
import warnings
from typing import Literal, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END

from app.extraction.state import PassportState
from app.extraction.tools import run_passport_eye, run_tesseract, run_easyocr
from app.extraction.aggregator import aggregate_ocr_results
from app.extraction.logic import critic_validate

logger = logging.getLogger(__name__)

# Suppress sklearn/skimage deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="passporteye")
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Thread pool for sync operations
_executor = ThreadPoolExecutor(max_workers=2)


async def node_parallel_ocr(state: PassportState) -> PassportState:
    """
    Step 1: Run all 3 OCR engines in parallel for MRZ extraction.
    """
    file_path = state.get("image_path") or state.get("file_path")
    logger.info(f"Running parallel OCR on: {file_path}")

    # Run all OCR engines in parallel
    results = await asyncio.gather(
        run_passport_eye(file_path),
        run_tesseract(file_path),
        run_easyocr(file_path),
        return_exceptions=True,
    )

    # Handle exceptions
    ocr_results = []
    for i, r in enumerate(results):
        source = ["passport_eye", "tesseract", "easyocr"][i]
        if isinstance(r, Exception):
            ocr_results.append({
                "source": source,
                "success": False,
                "error": str(r),
            })
        else:
            ocr_results.append(r)

    # Log results
    for r in ocr_results:
        status = "✓" if r.get("success") else "✗"
        checksum = "✓" if r.get("checksum_valid") else "✗"
        logger.info(f"  {status} {r.get('source')}: checksum={checksum}")

    errors = [r.get("error") for r in ocr_results if r.get("error")]

    return {
        **state,
        "ocr_results": ocr_results,
        "errors": errors,
    }


async def node_aggregate(state: PassportState) -> PassportState:
    """
    Step 2: Aggregate OCR results using field-level strategy.
    MRZ with valid checksum is the primary source of truth.
    """
    ocr_results = state.get("ocr_results", [])

    if not ocr_results:
        return {
            **state,
            "final_data": None,
            "confidence": 0.0,
            "source": "none",
            "fraud_flags": ["NO_OCR_RESULTS"],
        }

    # Check if any result has valid checksum
    has_valid_checksum = any(r.get("checksum_valid") for r in ocr_results if r.get("success"))

    # Extract MRZ lines from raw text
    mrz_lines = None
    for r in ocr_results:
        if r.get("success") and r.get("raw_text"):
            mrz_lines = _extract_mrz_lines(r.get("raw_text", ""))
            if mrz_lines:
                break

    # Aggregate results
    aggregated = aggregate_ocr_results(ocr_results, mrz_lines)

    # Convert to flat dict
    final_data = {}
    for field_name, agg_field in aggregated.fields.items():
        final_data[field_name] = agg_field.value

    # Identify low-confidence fields for potential reflection
    low_confidence_fields = [
        f for f, af in aggregated.fields.items()
        if af.confidence < 0.7 and final_data.get(f)
    ]

    # Run critic validation
    is_valid, fraud_flags = critic_validate(final_data) if final_data else (False, ["NO_DATA"])

    # If we have valid checksum, confidence is high
    confidence = aggregated.overall_confidence
    if has_valid_checksum:
        confidence = max(confidence, 0.95)

    logger.info(
        f"Aggregation: confidence={confidence:.2f}, checksum_valid={has_valid_checksum}, "
        f"sources={aggregated.sources_used}"
    )

    return {
        **state,
        "final_data": final_data,
        "confidence": confidence,
        "source": "aggregated_" + "+".join(aggregated.sources_used),
        "fraud_flags": fraud_flags,
        "low_confidence_fields": low_confidence_fields,
        "has_valid_checksum": has_valid_checksum,
        "mrz_lines": mrz_lines,
        "needs_human_review": False,
    }


async def node_vision_check(state: PassportState) -> PassportState:
    """
    Step 3: Use LLM Vision to read visual text for COMPARISON only.

    This is NOT the primary source - it's a check to detect:
    - Document tampering (visual != MRZ)
    - OCR errors that need reflection

    If LLM Vision disagrees with checksum-validated MRZ, LLM is likely wrong.
    """
    if not state.get("use_llm", True):
        logger.info("LLM Vision check disabled")
        return state

    file_path = state.get("image_path")
    has_valid_checksum = state.get("has_valid_checksum", False)
    final_data = state.get("final_data", {})

    # Run LLM Vision in executor
    from pathlib import Path
    from app.extraction.fraud_detector import extract_visual_data

    loop = asyncio.get_event_loop()
    try:
        visual_data = await loop.run_in_executor(
            _executor,
            extract_visual_data,
            Path(file_path)
        )
    except Exception as e:
        logger.warning(f"Vision check failed: {e}")
        visual_data = None

    if not visual_data:
        logger.info("Vision check: No visual data extracted")
        return {
            **state,
            "visual_data": None,
        }

    logger.info(f"Vision check: Got visual data")

    # Compare visual to MRZ-aggregated data
    fraud_flags = list(state.get("fraud_flags", []))
    discrepancies = []

    # Only flag as fraud if MRZ is checksum-validated AND visual clearly disagrees
    if has_valid_checksum:
        # Compare key fields
        comparisons = [
            ("passport_number", final_data.get("passport_number"), visual_data.get("passport_number")),
            ("date_of_birth", str(final_data.get("date_of_birth", "")), visual_data.get("date_of_birth")),
            ("expiry_date", str(final_data.get("expiry_date", "")), visual_data.get("expiry_date")),
        ]

        for field, mrz_val, visual_val in comparisons:
            if mrz_val and visual_val:
                mrz_clean = _normalize_for_compare(mrz_val)
                visual_clean = _normalize_for_compare(visual_val)

                if mrz_clean != visual_clean:
                    # MRZ is checksum-validated, so if visual disagrees:
                    # Either LLM hallucinated OR document is forged
                    discrepancies.append({
                        "field": field,
                        "mrz_value": mrz_val,
                        "visual_value": visual_val,
                    })
                    logger.warning(f"Discrepancy in {field}: MRZ={mrz_val}, Visual={visual_val}")

    # Determine if this is LLM hallucination or real fraud
    if discrepancies:
        # If passport number differs significantly, LLM probably hallucinated
        # Real passport numbers are 8-9 chars with specific format
        passport_disc = next((d for d in discrepancies if d["field"] == "passport_number"), None)
        if passport_disc:
            mrz_pn = passport_disc["mrz_value"]
            vis_pn = passport_disc["visual_value"]

            # If visual passport# is very different length or format, it's hallucination
            if abs(len(mrz_pn) - len(vis_pn)) > 2:
                logger.warning(f"LLM likely hallucinated passport number: {vis_pn} (MRZ: {mrz_pn})")
                # Don't flag as fraud - LLM is wrong, not the document
                discrepancies = [d for d in discrepancies if d["field"] != "passport_number"]

        # Remaining discrepancies might be real fraud (date tampering)
        for d in discrepancies:
            fraud_flags.append(f"{d['field'].upper()}_MISMATCH: Visual ({d['visual_value']}) vs MRZ ({d['mrz_value']})")

        if discrepancies:
            fraud_flags.append("POTENTIAL_DOCUMENT_TAMPERING")

    return {
        **state,
        "visual_data": visual_data,
        "fraud_flags": fraud_flags,
        "discrepancies": discrepancies,
        "needs_human_review": len(discrepancies) > 0,
    }


async def node_reflection(state: PassportState) -> PassportState:
    """
    Step 4: Run reflection agent for low-confidence fields or discrepancies.

    Only runs if:
    - There are low-confidence fields from OCR
    - There are discrepancies between visual and MRZ (potential fraud)
    """
    low_conf_fields = state.get("low_confidence_fields", [])
    discrepancies = state.get("discrepancies", [])
    fraud_flags = state.get("fraud_flags", [])

    # Check if reflection is needed
    needs_reflection = (
        len(low_conf_fields) > 0 or
        "POTENTIAL_DOCUMENT_TAMPERING" in fraud_flags
    )

    if not needs_reflection:
        logger.info("No reflection needed - high confidence results")
        return state

    if not state.get("use_llm", True):
        logger.info("Reflection disabled (use_llm=False)")
        return state

    logger.info(f"Running reflection for: low_conf={low_conf_fields}, fraud_flags={fraud_flags}")

    try:
        from app.extraction.reflection_agent import reflect_and_fix
        from app.models.schemas import PassportData
        from pathlib import Path
        from datetime import date

        final_data = state.get("final_data", {})
        mrz_lines = state.get("mrz_lines", [])

        # Build PassportData for reflection
        passport_data = _dict_to_passport_data(final_data)
        if not passport_data:
            logger.warning("Could not create PassportData for reflection")
            return state

        # Create fraud flags for reflection
        reflection_flags = list(fraud_flags) + [f"LOW_CONFIDENCE_{f.upper()}" for f in low_conf_fields]

        # Run reflection
        file_path = state.get("image_path")
        corrected = reflect_and_fix(
            passport_data,
            mrz_lines or [],
            reflection_flags,
            image_path=Path(file_path) if file_path else None,
            use_vision=True,
            max_attempts=2
        )

        if corrected:
            # Update fields that were corrected
            corrected_dict = corrected.model_dump()
            updated_data = final_data.copy()

            fields_updated = []
            for field in low_conf_fields:
                if field in corrected_dict and corrected_dict[field]:
                    old_val = updated_data.get(field)
                    new_val = corrected_dict[field]
                    if str(old_val) != str(new_val):
                        updated_data[field] = new_val
                        fields_updated.append(field)
                        logger.info(f"Reflection updated {field}: {old_val} → {new_val}")

            if fields_updated:
                return {
                    **state,
                    "final_data": updated_data,
                    "confidence": min(0.95, state.get("confidence", 0.5) + 0.1),
                    "source": state.get("source", "") + "_reflected",
                }

    except Exception as e:
        logger.warning(f"Reflection failed: {e}")
        errors = list(state.get("errors", []))
        errors.append(f"Reflection: {str(e)}")
        return {**state, "errors": errors}

    return state


async def node_human_review(state: PassportState) -> PassportState:
    """Mark for human review when potential fraud detected."""
    logger.warning(f"Document flagged for human review: {state.get('fraud_flags')}")
    return {**state, "needs_human_review": True}


def route_after_vision(
    state: PassportState,
) -> Literal["reflection", "human_review", "__end__"]:
    """Route based on vision check results."""
    low_conf_fields = state.get("low_confidence_fields", [])
    fraud_flags = state.get("fraud_flags", [])
    use_llm = state.get("use_llm", True)

    # If potential tampering, go to human review after reflection
    if "POTENTIAL_DOCUMENT_TAMPERING" in fraud_flags:
        if use_llm:
            return "reflection"
        return "human_review"

    # If low confidence fields and LLM enabled, try reflection
    if low_conf_fields and use_llm:
        return "reflection"

    return END


def route_after_reflection(state: PassportState) -> Literal["human_review", "__end__"]:
    """Route after reflection."""
    fraud_flags = state.get("fraud_flags", [])

    if "POTENTIAL_DOCUMENT_TAMPERING" in fraud_flags:
        return "human_review"

    return END


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_mrz_lines(text: str) -> Optional[List[str]]:
    """Extract MRZ lines from OCR text."""
    text = text.upper()
    lines = text.split('\n')

    mrz_candidates = []
    for line in lines:
        line = line.strip().replace(' ', '').replace('«', '<')
        if len(line) >= 40:
            mrz_chars = sum(1 for c in line if c.isalnum() or c == '<')
            if mrz_chars / len(line) > 0.9:
                mrz_candidates.append(line[:44])

    for i, line in enumerate(mrz_candidates):
        if line.startswith('P') and i + 1 < len(mrz_candidates):
            return [line, mrz_candidates[i + 1]]

    return None


def _normalize_for_compare(value: str) -> str:
    """Normalize value for comparison."""
    if not value:
        return ""
    # Remove spaces, dashes, convert to uppercase
    return str(value).upper().replace(" ", "").replace("-", "").replace("<", "")


def _dict_to_passport_data(data: dict):
    """Convert dict to PassportData."""
    try:
        from app.models.schemas import PassportData, Sex
        from datetime import date, datetime

        def parse_date(val):
            if isinstance(val, date):
                return val
            if isinstance(val, str):
                if len(val) == 6 and val.isdigit():
                    yy = int(val[0:2])
                    mm = int(val[2:4])
                    dd = int(val[4:6])
                    year = 1900 + yy if yy > 50 else 2000 + yy
                    return date(year, mm, dd)
                try:
                    return datetime.fromisoformat(val).date()
                except:
                    pass
            return date(1900, 1, 1)

        def parse_sex(val):
            if isinstance(val, Sex):
                return val
            val = str(val).upper()[:1]
            if val == "M":
                return Sex.MALE
            elif val == "F":
                return Sex.FEMALE
            return Sex.OTHER

        return PassportData(
            surname=data.get("surname", ""),
            given_names=data.get("given_names", ""),
            passport_number=data.get("passport_number", ""),
            nationality=data.get("nationality", ""),
            date_of_birth=parse_date(data.get("date_of_birth")),
            sex=parse_sex(data.get("sex", "X")),
            expiry_date=parse_date(data.get("expiry_date")),
            country_of_issue=data.get("country", ""),
            extraction_method="v4",
            confidence_score=0.8
        )
    except Exception as e:
        logger.warning(f"Could not convert to PassportData: {e}")
        return None


# =============================================================================
# Build Graph
# =============================================================================

def build_extraction_graph_v4() -> StateGraph:
    """
    Build V4 extraction workflow:

    parallel_ocr → aggregate → vision_check → [reflection] → [human_review] → END
    """
    workflow = StateGraph(PassportState)

    # Add nodes
    workflow.add_node("parallel_ocr", node_parallel_ocr)
    workflow.add_node("aggregate", node_aggregate)
    workflow.add_node("vision_check", node_vision_check)
    workflow.add_node("reflection", node_reflection)
    workflow.add_node("human_review", node_human_review)

    # Entry point
    workflow.set_entry_point("parallel_ocr")

    # Linear flow: OCR → Aggregate → Vision Check
    workflow.add_edge("parallel_ocr", "aggregate")
    workflow.add_edge("aggregate", "vision_check")

    # Conditional after vision check
    workflow.add_conditional_edges(
        "vision_check",
        route_after_vision,
        {
            "reflection": "reflection",
            "human_review": "human_review",
            END: END,
        },
    )

    # Conditional after reflection
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "human_review": "human_review",
            END: END,
        },
    )

    workflow.add_edge("human_review", END)

    return workflow.compile()


# Compiled graph
graph_v4 = build_extraction_graph_v4()
