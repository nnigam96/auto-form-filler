"""
LangGraph workflow V3: Field-Level Aggregation.

Replaces result-level voting with field-level aggregation:
1. Run all OCR engines in parallel
2. Aggregate best value for each field
3. Run reflection only for low-confidence fields
4. Final validation
"""

import asyncio
import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.extraction.state import PassportState
from app.extraction.tools import run_passport_eye, run_tesseract, run_easyocr
from app.extraction.aggregator import aggregate_ocr_results
from app.extraction.logic import critic_validate

logger = logging.getLogger(__name__)


async def node_parallel_ocr(state: PassportState) -> PassportState:
    """
    Run all OCR tools in parallel.
    Does NOT aggregate yet - just collects raw results.
    """
    file_path = state.get("image_path") or state.get("file_path")
    logger.info(f"Running parallel OCR on: {file_path}")

    # Run all tools concurrently
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
        logger.info(f"  {status} {r.get('source')}: checksum={r.get('checksum_valid')}")

    errors = [r.get("error") for r in ocr_results if r.get("error")]

    return {
        **state,
        "ocr_results": ocr_results,
        "errors": errors,
    }


async def node_aggregate(state: PassportState) -> PassportState:
    """
    Aggregate OCR results using field-level strategy.
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

    # Extract MRZ lines from raw text if available
    mrz_lines = None
    for r in ocr_results:
        if r.get("success") and r.get("raw_text"):
            mrz_lines = _extract_mrz_lines(r.get("raw_text", ""))
            if mrz_lines:
                break

    # Aggregate
    aggregated = aggregate_ocr_results(ocr_results, mrz_lines)

    # Convert to flat dict for PassportData
    final_data = {}
    field_confidences = {}

    for field_name, agg_field in aggregated.fields.items():
        final_data[field_name] = agg_field.value
        field_confidences[field_name] = agg_field.confidence

    # Identify low-confidence fields
    low_confidence_fields = [
        f for f, c in field_confidences.items()
        if c < 0.7 and final_data.get(f)
    ]

    # Run critic validation
    is_valid, fraud_flags = critic_validate(final_data) if final_data else (False, ["NO_DATA"])

    logger.info(
        f"Aggregation result: confidence={aggregated.overall_confidence:.2f}, "
        f"sources={aggregated.sources_used}, low_conf_fields={low_confidence_fields}"
    )

    return {
        **state,
        "final_data": final_data,
        "confidence": aggregated.overall_confidence,
        "source": "aggregated_" + "+".join(aggregated.sources_used),
        "fraud_flags": fraud_flags,
        "low_confidence_fields": low_confidence_fields,
        "needs_human_review": False,
    }


async def node_reflection(state: PassportState) -> PassportState:
    """
    Run reflection agent ONLY for low-confidence fields.
    More targeted than reflecting on everything.
    """
    low_conf_fields = state.get("low_confidence_fields", [])

    if not low_conf_fields:
        logger.info("No low-confidence fields, skipping reflection")
        return state

    if not state.get("use_llm", False):
        logger.info("LLM disabled, skipping reflection")
        return state

    logger.info(f"Running reflection for low-confidence fields: {low_conf_fields}")

    try:
        from app.extraction.reflection_agent import reflect_and_fix
        from app.models.schemas import PassportData
        from pathlib import Path
        from datetime import date

        # Convert current data to PassportData for reflection
        final_data = state.get("final_data", {})

        # Build PassportData (need to handle date conversion)
        passport_data = _dict_to_passport_data(final_data)
        if not passport_data:
            return state

        # Get MRZ lines
        mrz_lines = []
        for r in state.get("ocr_results", []):
            if r.get("success") and r.get("raw_text"):
                mrz_lines = _extract_mrz_lines(r.get("raw_text", ""))
                if mrz_lines:
                    break

        # Create fraud flags focused on low-confidence fields
        fraud_flags = [f"LOW_CONFIDENCE_{f.upper()}" for f in low_conf_fields]

        # Run reflection
        file_path = state.get("image_path") or state.get("file_path")
        corrected = reflect_and_fix(
            passport_data,
            mrz_lines,
            fraud_flags,
            image_path=Path(file_path) if file_path else None,
            use_vision=True,
            max_attempts=1  # Single pass for efficiency
        )

        if corrected:
            # Update only the reflected fields
            updated_data = final_data.copy()
            corrected_dict = corrected.model_dump()

            for field in low_conf_fields:
                if field in corrected_dict and corrected_dict[field]:
                    updated_data[field] = corrected_dict[field]
                    logger.info(f"Reflection updated {field}: {final_data.get(field)} → {corrected_dict[field]}")

            return {
                **state,
                "final_data": updated_data,
                "confidence": min(0.95, state.get("confidence", 0.5) + 0.15),
                "source": state.get("source", "") + "_reflected",
            }

    except Exception as e:
        logger.warning(f"Reflection failed: {e}")
        errors = list(state.get("errors", []))
        errors.append(f"Reflection: {str(e)}")
        return {**state, "errors": errors}

    return state


async def node_human_review(state: PassportState) -> PassportState:
    """Mark state for human review."""
    logger.warning(f"Flagged for human review: {state.get('fraud_flags')}")
    return {**state, "needs_human_review": True}


def route_after_aggregate(
    state: PassportState,
) -> Literal["reflection", "human_review", "__end__"]:
    """Decide next step after aggregation."""
    confidence = state.get("confidence", 0.0)
    fraud_flags = state.get("fraud_flags", [])
    low_conf_fields = state.get("low_confidence_fields", [])
    use_llm = state.get("use_llm", False)

    # If serious fraud detected, go to human review
    if fraud_flags and any(
        f.startswith("FRAUD") or f == "DOB_IN_FUTURE" or f == "EXPIRY_BEFORE_BIRTH"
        for f in fraud_flags
    ):
        logger.info(f"Routing to human_review due to: {fraud_flags}")
        return "human_review"

    # If high confidence and no issues, we're done
    if confidence >= 0.9 and not low_conf_fields:
        logger.info(f"Routing to END (confidence={confidence:.2f})")
        return END

    # If low confidence fields and LLM enabled, try reflection
    if low_conf_fields and use_llm:
        logger.info(f"Routing to reflection for: {low_conf_fields}")
        return "reflection"

    # Otherwise end (even with some low confidence)
    logger.info(f"Routing to END (confidence={confidence:.2f}, no LLM)")
    return END


def route_after_reflection(state: PassportState) -> Literal["human_review", "__end__"]:
    """Decide after reflection."""
    fraud_flags = state.get("fraud_flags", [])
    confidence = state.get("confidence", 0.0)

    if fraud_flags and confidence < 0.7:
        return "human_review"

    return END


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_mrz_lines(text: str) -> list:
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

    return []


def _dict_to_passport_data(data: dict):
    """Convert dict to PassportData, handling date parsing."""
    try:
        from app.models.schemas import PassportData, Sex
        from datetime import date

        # Handle dates
        def parse_date(val):
            if isinstance(val, date):
                return val
            if isinstance(val, str):
                # Try YYMMDD
                if len(val) == 6 and val.isdigit():
                    yy = int(val[0:2])
                    mm = int(val[2:4])
                    dd = int(val[4:6])
                    year = 1900 + yy if yy > 50 else 2000 + yy
                    return date(year, mm, dd)
                # Try ISO
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(val).date()
                except:
                    pass
            return date(1900, 1, 1)  # Fallback

        # Handle sex
        def parse_sex(val):
            if isinstance(val, Sex):
                return val
            val = str(val).upper()
            if val in ["M", "MALE"]:
                return Sex.MALE
            elif val in ["F", "FEMALE"]:
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
            extraction_method="aggregated",
            confidence_score=0.8
        )
    except Exception as e:
        logger.warning(f"Could not convert to PassportData: {e}")
        return None


# =============================================================================
# Build Graph
# =============================================================================

def build_extraction_graph_v3() -> StateGraph:
    """Build V3 extraction workflow with field-level aggregation."""

    workflow = StateGraph(PassportState)

    # Add nodes
    workflow.add_node("parallel_ocr", node_parallel_ocr)
    workflow.add_node("aggregate", node_aggregate)
    workflow.add_node("reflection", node_reflection)
    workflow.add_node("human_review", node_human_review)

    # Set entry point
    workflow.set_entry_point("parallel_ocr")

    # OCR → Aggregate
    workflow.add_edge("parallel_ocr", "aggregate")

    # Aggregate → conditional routing
    workflow.add_conditional_edges(
        "aggregate",
        route_after_aggregate,
        {
            "reflection": "reflection",
            "human_review": "human_review",
            END: END,
        },
    )

    # Reflection → conditional routing
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


# Compiled graph instance
graph_v3 = build_extraction_graph_v3()
