"""
Auto Form Filler - FastAPI Application

Main entry point for document extraction and form automation.

Endpoints:
- /extract/passport (V5) - Production endpoint with HITL support
- /upload/g28 - G-28 form extraction
- /extract - Combined passport + G-28 extraction
- /fill-form - Full automation pipeline

Deprecated endpoints available at /extract/passport/v1-v4
"""

import logging
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import aiofiles

from app.config import settings
from app.models.schemas import PassportData, FormData, ExtractionResult
from app.extraction.g28 import extract_g28_data
from app.automation.form_filler import fill_form_async

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Auto Form Filler",
    description="Document automation system for immigration forms with fraud detection",
    version="5.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Deprecated V1-V4 endpoints moved to scratch/archived_graphs/
# To restore: from app.api.deprecated_endpoints import router as deprecated_router
# app.include_router(deprecated_router)


# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the upload interface."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "5.0.0",
        "llm_enabled": settings.use_llm_extraction,
    }


@app.post("/extract/passport")
async def extract_passport(
    file: UploadFile = File(...),
    use_llm: bool = True,
):
    """
    Extract passport data with Human-In-The-Loop (HITL) support.

    V5 Pipeline:
    1. Parallel OCR (PassportEye, Tesseract, EasyOCR) → MRZ data
    2. Field-level aggregation
    3. LLM Vision → Visual text (what's printed)
    4. Cross-validate MRZ vs Visual
    5. If aligned → High confidence output
    6. If misaligned → Return both values for user correction

    Returns:
    - fields: Dict with MRZ value, Visual value, final value, needs_review flag
    - needs_human_review: True if user should verify conflicting fields
    - fraud_flags: List of detected issues
    """
    from app.extraction.pipeline import extract_passport_v5

    # Validate file type
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"passport_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"[V5] Processing: {file_path}")

    try:
        result = await extract_passport_v5(str(file_path), use_llm=use_llm)

        if result is None:
            raise HTTPException(
                status_code=422,
                detail="Could not extract data from passport."
            )

        return {
            "success": result.success,
            "file_id": file_id,
            "fields": {k: {
                "mrz_value": v.mrz_value,
                "visual_value": v.visual_value,
                "final_value": v.final_value,
                "confidence": v.confidence,
                "needs_review": v.needs_review,
                "source": v.source,
            } for k, v in result.fields.items()},
            "final_data": result.get_final_data(),
            "overall_confidence": result.overall_confidence,
            "needs_human_review": result.needs_human_review,
            "review_reason": result.review_reason,
            "fraud_flags": result.fraud_flags,
            "mrz_checksum_valid": result.mrz_checksum_valid,
            "version": "v5",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[V5] Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/passport/confirm")
async def confirm_passport_extraction(
    file_id: str = Form(...),
    corrections: str = Form(...),  # JSON string of field corrections
):
    """
    Confirm/correct passport extraction after HITL review.

    User provides corrections for conflicting fields.
    Returns finalized PassportData.
    """
    import json

    try:
        corrections_dict = json.loads(corrections)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid corrections JSON")

    # TODO: Retrieve original extraction, apply corrections, save final result
    # For now, just return the corrections as confirmation

    logger.info(f"[V5] Confirmed extraction for {file_id}: {corrections_dict}")

    return {
        "success": True,
        "file_id": file_id,
        "data": corrections_dict,
        "status": "confirmed",
    }


@app.post("/upload/g28")
async def upload_g28(file: UploadFile = File(...)):
    """Extract data from G-28 form."""
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    file_id = str(uuid.uuid4())[:8]
    file_path = settings.upload_dir / f"g28_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    logger.info(f"Processing G-28: {file_path}")

    try:
        g28_data = extract_g28_data(file_path)

        if g28_data is None:
            raise HTTPException(
                status_code=422,
                detail="Could not extract data from G-28 form."
            )

        return {
            "success": True,
            "file_id": file_id,
            "data": g28_data.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"G-28 extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_documents(
    passport: UploadFile = File(...),
    g28: UploadFile = File(...),
):
    """
    Extract data from both passport and G-28 form.

    Uses V5 pipeline for passport extraction.
    Returns combined FormData ready for form filling.
    """
    from app.extraction.pipeline import extract_passport_v5

    result = ExtractionResult(success=False)

    # Validate file types
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png"]
    passport_ext = Path(passport.filename).suffix.lower()
    g28_ext = Path(g28.filename).suffix.lower()

    if passport_ext not in allowed_types:
        result.errors.append(f"Invalid passport file type. Allowed: {allowed_types}")
        return result.model_dump()

    if g28_ext not in allowed_types:
        result.errors.append(f"Invalid G-28 file type. Allowed: {allowed_types}")
        return result.model_dump()

    # Process passport
    passport_id = str(uuid.uuid4())[:8]
    passport_path = settings.upload_dir / f"passport_{passport_id}{passport_ext}"

    try:
        async with aiofiles.open(passport_path, "wb") as f:
            await f.write(await passport.read())

        v5_result = await extract_passport_v5(str(passport_path), use_llm=True)

        if v5_result is None or not v5_result.success:
            result.errors.append("Failed to extract passport data.")
        else:
            # Convert to PassportData
            final_data = v5_result.get_final_data()

            def parse_date(val):
                if not val:
                    return datetime(1900, 1, 1).date()
                if isinstance(val, str):
                    try:
                        return datetime.strptime(val, "%Y-%m-%d").date()
                    except:
                        pass
                    if len(val) == 6 and val.isdigit():
                        yy, mm, dd = int(val[0:2]), int(val[2:4]), int(val[4:6])
                        year = 1900 + yy if yy > 50 else 2000 + yy
                        return datetime(year, mm, dd).date()
                return datetime(1900, 1, 1).date()

            def parse_sex(val):
                from app.models.schemas import Sex
                val = (val or "").upper()[:1]
                return Sex.MALE if val == "M" else Sex.FEMALE if val == "F" else Sex.OTHER

            passport_data = PassportData(
                surname=final_data.get("surname", ""),
                given_names=final_data.get("given_names", ""),
                passport_number=final_data.get("passport_number", ""),
                nationality=final_data.get("nationality", ""),
                date_of_birth=parse_date(final_data.get("date_of_birth")),
                sex=parse_sex(final_data.get("sex")),
                expiry_date=parse_date(final_data.get("expiry_date")),
                country_of_issue=final_data.get("country", ""),
                extraction_method="v5",
                confidence_score=v5_result.overall_confidence,
            )
            result.passport_data = passport_data

            # Add warnings for fields needing review
            if v5_result.needs_human_review:
                result.errors.append(f"REVIEW NEEDED: {v5_result.review_reason}")

    except Exception as e:
        logger.error(f"Passport processing failed: {e}")
        result.errors.append(f"Passport processing error: {str(e)}")

    # Process G-28
    g28_id = str(uuid.uuid4())[:8]
    g28_path = settings.upload_dir / f"g28_{g28_id}{g28_ext}"

    try:
        async with aiofiles.open(g28_path, "wb") as f:
            await f.write(await g28.read())

        g28_data = extract_g28_data(g28_path)
        if g28_data is None:
            result.errors.append("Failed to extract G-28 data.")
        else:
            result.g28_data = g28_data
    except Exception as e:
        logger.error(f"G-28 processing failed: {e}")
        result.errors.append(f"G-28 processing error: {str(e)}")

    # Combine if both successful
    if result.passport_data and result.g28_data:
        try:
            result.form_data = FormData.from_extracted_data(
                result.passport_data,
                result.g28_data
            )
            result.success = True
        except Exception as e:
            logger.error(f"Data combination failed: {e}")
            result.errors.append(f"Failed to combine data: {str(e)}")

    return result.model_dump()


@app.post("/fill-form")
async def fill_form_endpoint(
    passport: UploadFile = File(...),
    g28: UploadFile = File(...),
    headless: str = Form("true"),
):
    """
    Full automation: Extract documents and fill target form.

    Returns screenshot of filled form.
    """
    extract_result = await extract_documents(passport, g28)

    if not extract_result["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"Extraction failed: {extract_result['errors']}"
        )

    form_data = FormData(**extract_result["form_data"])

    screenshot_id = str(uuid.uuid4())[:8]
    screenshot_path = settings.screenshot_dir / f"filled_form_{screenshot_id}.png"

    try:
        headless_bool = headless.lower() in ("true", "1", "yes")
        fill_result = await fill_form_async(
            form_data,
            screenshot_path=screenshot_path,
            headless=headless_bool,
        )

        if not fill_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Form filling failed: {fill_result['errors']}"
            )

        return {
            "success": True,
            "extraction": extract_result,
            "fill_result": {
                "filled_fields": fill_result["filled_fields"],
                "failed_fields": fill_result["failed_fields"],
            },
            "screenshot_url": f"/screenshots/{screenshot_path.name}" if screenshot_path.exists() else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form filling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screenshots/{filename}")
async def get_screenshot(filename: str):
    """Serve screenshot images."""
    file_path = settings.screenshot_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Screenshot not found")

    return FileResponse(file_path, media_type="image/png")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
