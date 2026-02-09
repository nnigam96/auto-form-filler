"""
LangGraph workflow for passport extraction.
Orchestrates OCR tools, voting, and fallback logic.
"""

import asyncio
import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.extraction.state import PassportState
from app.extraction.tools import run_passport_eye, run_tesseract, run_easyocr
from app.extraction.logic import vote_on_results, critic_validate

logger = logging.getLogger(__name__)


async def node_ensemble(state: PassportState) -> PassportState:
    """
    Run all OCR tools in parallel and vote on results.
    """
    logger.info(f"Running ensemble OCR on: {state['image_path']}")

    # Run all tools concurrently
    results = await asyncio.gather(
        run_passport_eye(state["image_path"]),
        run_tesseract(state["image_path"]),
        run_easyocr(state["image_path"]),
        return_exceptions=True,
    )

    # Handle exceptions
    ocr_results = []
    for r in results:
        if isinstance(r, Exception):
            ocr_results.append(
                {
                    "source": "unknown",
                    "success": False,
                    "error": str(r),
                }
            )
        else:
            ocr_results.append(r)

    # Log results
    for r in ocr_results:
        status = "✓" if r.get("success") else "✗"
        logger.info(f"  {status} {r.get('source')}: checksum={r.get('checksum_valid')}")

    # Vote on results
    best_data, confidence, source = vote_on_results(ocr_results)

    # Run critic validation
    is_valid, fraud_flags = (
        critic_validate(best_data) if best_data else (False, ["NO_DATA"])
    )

    # Collect errors
    errors = [r.get("error") for r in ocr_results if r.get("error")]

    logger.info(
        f"Ensemble result: confidence={confidence:.2f}, source={source}, valid={is_valid}"
    )

    return {
        **state,
        "ocr_results": ocr_results,
        "final_data": best_data,
        "confidence": confidence,
        "source": source,
        "errors": errors,
        "fraud_flags": fraud_flags,
        "needs_human_review": False,
    }


async def node_vision_fallback(state: PassportState) -> PassportState:
    """
    Call LLM Vision API as fallback.
    Only runs if use_llm is enabled.
    """
    if not state.get("use_llm", False):
        logger.info("LLM fallback disabled, skipping")
        return state

    logger.info("Running LLM Vision fallback...")

    try:
        from app.extraction.llm_vision import extract_with_llm_vision
        from pathlib import Path

        result = extract_with_llm_vision(Path(state["image_path"]))

        if result:
            # Convert Pydantic to dict
            data = {
                "surname": result.surname,
                "given_names": result.given_names,
                "passport_number": result.passport_number,
                "nationality": result.nationality,
                "date_of_birth": (
                    result.date_of_birth.strftime("%y%m%d") if result.date_of_birth else ""
                ),
                "sex": result.sex.value if result.sex else "",
                "expiry_date": (
                    result.expiry_date.strftime("%y%m%d") if result.expiry_date else ""
                ),
                "country": result.country_of_issue,
            }

            # Re-validate with critic
            is_valid, fraud_flags = critic_validate(data)

            logger.info("LLM Vision extraction successful")

            return {
                **state,
                "final_data": data,
                "confidence": 0.9,
                "source": "llm_vision",
                "fraud_flags": fraud_flags,
                "needs_human_review": not is_valid,
            }

    except Exception as e:
        logger.error(f"LLM Vision failed: {e}")
        errors = list(state.get("errors", []))
        errors.append(f"LLM Vision: {str(e)}")
        return {**state, "errors": errors}

    return state


async def node_human_review(state: PassportState) -> PassportState:
    """
    Mark state for human review.
    In production, this would queue for manual verification.
    """
    logger.warning(f"Flagged for human review: {state.get('fraud_flags')}")

    return {
        **state,
        "needs_human_review": True,
    }


def route_after_ensemble(
    state: PassportState,
) -> Literal["vision_fallback", "human_review", "__end__"]:
    """
    Decide next step after ensemble OCR.
    """
    confidence = state.get("confidence", 0.0)
    fraud_flags = state.get("fraud_flags", [])
    use_llm = state.get("use_llm", False)

    # If fraud detected, go to human review
    if fraud_flags and any(
        f.startswith("FRAUD") or f == "DOB_IN_FUTURE" for f in fraud_flags
    ):
        logger.info(f"Routing to human_review due to: {fraud_flags}")
        return "human_review"

    # If high confidence and valid, we're done
    if confidence >= 0.9 and not fraud_flags:
        logger.info(f"Routing to END (confidence={confidence:.2f})")
        return END

    # If low confidence and LLM enabled, try vision fallback
    if confidence < 0.9 and use_llm:
        logger.info(f"Routing to vision_fallback (confidence={confidence:.2f})")
        return "vision_fallback"

    # Otherwise end (even with low confidence)
    logger.info(f"Routing to END (confidence={confidence:.2f}, no LLM)")
    return END


def route_after_vision(state: PassportState) -> Literal["human_review", "__end__"]:
    """
    Decide after vision fallback.
    """
    fraud_flags = state.get("fraud_flags", [])

    if fraud_flags:
        return "human_review"

    return END


# Build the graph
def build_extraction_graph() -> StateGraph:
    """Build and compile the extraction workflow."""

    workflow = StateGraph(PassportState)

    # Add nodes
    workflow.add_node("ensemble", node_ensemble)
    workflow.add_node("vision_fallback", node_vision_fallback)
    workflow.add_node("human_review", node_human_review)

    # Set entry point
    workflow.set_entry_point("ensemble")

    # Add edges
    workflow.add_conditional_edges(
        "ensemble",
        route_after_ensemble,
        {
            "vision_fallback": "vision_fallback",
            "human_review": "human_review",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "vision_fallback",
        route_after_vision,
        {
            "human_review": "human_review",
            END: END,
        },
    )

    workflow.add_edge("human_review", END)

    return workflow.compile()


# Compiled graph instance
graph = build_extraction_graph()
