"""
Deprecated API Endpoints (V1-V4)

These endpoints are kept for backward compatibility and comparison.
Use V5 (/extract/passport) for production.

Version History:
- V1: Sequential OCR + Reflection Agent
- V2: Parallel OCR + Result-level Voting
- V3: Parallel OCR + Field-level Aggregation
- V4: Parallel OCR + LLM Vision + Fraud Detection
- V5: Parallel OCR + LLM Vision + HITL (Current)
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
import aiofiles

from app.config import settings
from app.models.schemas import PassportData
from app.extraction.passport import extract_passport_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract/passport", tags=["deprecated"])


@router.post("/v1")
async def extract_passport_v1(
    file: UploadFile = File(...),
    use_llm: bool = False,
):
    """
    [DEPRECATED] V1: Sequential OCR + Reflection Agent

    Uses the original extraction pipeline:
    - Sequential OCR (best config from research, then exhaustive search)
    - Reflection agent for error correction
    - Optional LLM Vision fallback
    """
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")

    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"passport_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    logger.info(f"[V1] Processing: {file_path}")

    try:
        passport_data = extract_passport_data(file_path, use_llm=use_llm)

        if passport_data is None:
            raise HTTPException(status_code=422, detail="Could not extract data from passport.")

        return {
            "success": True,
            "file_id": file_id,
            "data": passport_data.model_dump(),
            "version": "v1",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V1] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2")
async def extract_passport_v2(
    file: UploadFile = File(...),
    use_llm: bool = False,
):
    """
    [DEPRECATED] V2: Parallel OCR + Result-level Voting

    Uses LangGraph orchestration with:
    - Parallel OCR execution (PassportEye, Tesseract, EasyOCR)
    - Result-level voting to pick best result
    - Critic validation for fraud detection
    """
    from app.extraction.graph import graph
    from app.extraction.state import PassportState
    from app.extraction.passport import parse_mrz_date, parse_mrz_sex

    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")

    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"passport_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    logger.info(f"[V2] Processing: {file_path}")

    initial_state: PassportState = {
        "image_path": str(file_path),
        "ocr_results": [],
        "final_data": None,
        "confidence": 0.0,
        "errors": [],
        "source": "",
        "needs_human_review": False,
        "fraud_flags": [],
        "use_llm": use_llm,
    }

    try:
        result = await graph.ainvoke(initial_state)

        response = {
            "success": result["final_data"] is not None,
            "file_id": file_id,
            "data": result["final_data"],
            "confidence": result["confidence"],
            "source": result["source"],
            "version": "v2",
        }

        if result["final_data"]:
            try:
                data = result["final_data"]
                passport_data = PassportData(
                    surname=data.get("surname", ""),
                    given_names=data.get("given_names", ""),
                    passport_number=data.get("passport_number", ""),
                    nationality=data.get("nationality", ""),
                    date_of_birth=parse_mrz_date(data.get("date_of_birth", ""), False),
                    sex=parse_mrz_sex(data.get("sex", "X")),
                    expiry_date=parse_mrz_date(data.get("expiry_date", ""), True),
                    country_of_issue=data.get("country", ""),
                    extraction_method=result["source"],
                    confidence_score=result["confidence"],
                )
                response["data"] = passport_data.model_dump()
            except Exception as e:
                logger.warning(f"[V2] Pydantic conversion failed: {e}")

        return response
    except Exception as e:
        logger.error(f"[V2] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v3")
async def extract_passport_v3(
    file: UploadFile = File(...),
    use_llm: bool = False,
):
    """
    [DEPRECATED] V3: Parallel OCR + Field-level Aggregation

    - Parallel OCR execution (PassportEye, Tesseract, EasyOCR)
    - Field-level aggregation (best value per field)
    - MRZ as ground truth for critical fields
    """
    from app.extraction.graph_v3 import graph_v3
    from app.extraction.state import PassportState
    from app.extraction.passport import parse_mrz_date, parse_mrz_sex

    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")

    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"passport_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    logger.info(f"[V3] Processing: {file_path}")

    initial_state: PassportState = {
        "image_path": str(file_path),
        "ocr_results": [],
        "final_data": None,
        "confidence": 0.0,
        "errors": [],
        "source": "",
        "needs_human_review": False,
        "fraud_flags": [],
        "use_llm": use_llm,
    }

    try:
        result = await graph_v3.ainvoke(initial_state)

        response = {
            "success": result["final_data"] is not None,
            "file_id": file_id,
            "data": result["final_data"],
            "confidence": result["confidence"],
            "source": result["source"],
            "low_confidence_fields": result.get("low_confidence_fields", []),
            "version": "v3",
        }

        if result["final_data"]:
            try:
                data = result["final_data"]
                passport_data = PassportData(
                    surname=data.get("surname", ""),
                    given_names=data.get("given_names", ""),
                    passport_number=data.get("passport_number", ""),
                    nationality=data.get("nationality", ""),
                    date_of_birth=parse_mrz_date(str(data.get("date_of_birth", "")), False),
                    sex=parse_mrz_sex(data.get("sex", "X")),
                    expiry_date=parse_mrz_date(str(data.get("expiry_date", "")), True),
                    country_of_issue=data.get("country", ""),
                    extraction_method=result["source"],
                    confidence_score=result["confidence"],
                )
                response["data"] = passport_data.model_dump()
            except Exception as e:
                logger.warning(f"[V3] Pydantic conversion failed: {e}")

        return response
    except Exception as e:
        logger.error(f"[V3] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v4")
async def extract_passport_v4(
    file: UploadFile = File(...),
):
    """
    [DEPRECATED] V4: Parallel OCR + LLM Vision + Fraud Detection

    - Parallel MRZ OCR (PassportEye, Tesseract, EasyOCR)
    - LLM Vision for visual text
    - Cross-validation: Visual vs MRZ
    - Fraud detection for mismatches
    """
    from app.extraction.graph_v4 import graph_v4
    from app.extraction.state import PassportState
    from datetime import datetime

    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")

    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"passport_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    logger.info(f"[V4] Processing: {file_path}")

    initial_state: PassportState = {
        "image_path": str(file_path),
        "ocr_results": [],
        "final_data": None,
        "confidence": 0.0,
        "errors": [],
        "source": "",
        "needs_human_review": False,
        "fraud_flags": [],
        "use_llm": True,
    }

    try:
        result = await graph_v4.ainvoke(initial_state)

        if result["final_data"] is None:
            raise HTTPException(status_code=422, detail="Could not extract data from passport.")

        data = result["final_data"]
        return {
            "success": True,
            "file_id": file_id,
            "data": data,
            "confidence": result["confidence"],
            "source": result["source"],
            "fraud_flags": result.get("fraud_flags", []),
            "needs_human_review": result.get("needs_human_review", False),
            "version": "v4",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V4] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
