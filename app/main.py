"""
FastAPI application entry point.

Provides API endpoints for:
- Document upload and extraction
- Form filling automation
- Health checks
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import aiofiles

from app.config import settings
from app.models.schemas import (
    PassportData,
    G28Data,
    FormData,
    ExtractionResult,
)
from app.extraction.passport import extract_passport_data
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
    description="Document automation system for immigration forms",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the upload interface."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_enabled": settings.use_llm_extraction,
    }


@app.get("/metrics/ocr")
async def get_ocr_metrics():
    """Get OCR performance metrics."""
    from app.extraction.passport import get_ocr_metrics
    
    metrics = get_ocr_metrics()
    
    # Aggregate statistics
    total_attempts = len(metrics)
    successful = [m for m in metrics if m.success]
    failed = [m for m in metrics if not m.success]
    
    # Group by method
    by_method = {}
    for m in metrics:
        method_name = m.method.split('_')[0]  # e.g., "tesseract" or "easyocr"
        if method_name not in by_method:
            by_method[method_name] = {"total": 0, "successful": 0, "failed": 0}
        by_method[method_name]["total"] += 1
        if m.success:
            by_method[method_name]["successful"] += 1
        else:
            by_method[method_name]["failed"] += 1
    
    return {
        "total_attempts": total_attempts,
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / total_attempts if total_attempts > 0 else 0,
        "by_method": by_method,
        "recent_attempts": [
            {
                "method": m.method,
                "success": m.success,
                "checksum_passed": m.checksum_passed,
                "mrz_lines_found": m.mrz_lines_found,
                "confidence": m.confidence_score,
                "error": m.error
            }
            for m in metrics[-10:]  # Last 10 attempts
        ]
    }


@app.post("/upload/passport")
async def upload_passport(file: UploadFile = File(...)):
    """
    Upload and process a passport document.

    Returns extracted passport data.
    """
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

    logger.info(f"Saved passport to {file_path}")

    # Extract data
    try:
        passport_data = extract_passport_data(file_path, use_llm=settings.use_llm_extraction)

        if passport_data is None:
            raise HTTPException(
                status_code=422,
                detail="Could not extract data from passport. Ensure MRZ is visible."
            )

        return {
            "success": True,
            "file_id": file_id,
            "data": passport_data.model_dump(),
        }

    except Exception as e:
        logger.error(f"Passport extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/g28")
async def upload_g28(file: UploadFile = File(...)):
    """
    Upload and process a G-28 form.

    Returns extracted attorney and client data.
    """
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
    file_path = settings.upload_dir / f"g28_{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Saved G-28 to {file_path}")

    # Extract data
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

    except Exception as e:
        logger.error(f"G-28 extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_documents(
    passport: UploadFile = File(...),
    g28: UploadFile = File(...),
):
    """
    Upload and extract data from both passport and G-28 form.

    Returns combined FormData ready for form filling.
    """
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

        passport_data = extract_passport_data(passport_path, use_llm=settings.use_llm_extraction)
        if passport_data is None:
            result.errors.append("Failed to extract passport data. Ensure MRZ is visible and readable.")
        else:
            result.passport_data = passport_data
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
            result.errors.append("Failed to extract G-28 data. Ensure form is complete and readable.")
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
    Upload documents, extract data, and fill the target form.

    Returns screenshot of filled form.
    """
    # First extract the data
    extract_result = await extract_documents(passport, g28)

    if not extract_result["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"Extraction failed: {extract_result['errors']}"
        )

    # Reconstruct FormData from dict
    form_data = FormData(**extract_result["form_data"])

    # Generate screenshot path
    screenshot_id = str(uuid.uuid4())[:8]
    screenshot_path = settings.screenshot_dir / f"filled_form_{screenshot_id}.png"

    # Fill the form
    try:
        # Convert headless string to boolean
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
# V2 Agentic Endpoints (LangGraph)
# =============================================================================

@app.post("/extract/passport/v2")
async def extract_passport_v2(
    file: UploadFile = File(...),
    use_llm: bool = False,
):
    """
    Extract passport data using the agentic graph.

    This is the v2 endpoint using LangGraph orchestration with:
    - Parallel OCR execution (PassportEye, Tesseract, EasyOCR)
    - Voting logic to pick best result
    - Critic validation for fraud detection
    - Optional LLM Vision fallback
    """
    from app.extraction.graph import graph
    from app.extraction.state import PassportState
    from app.extraction.passport import parse_mrz_date, parse_mrz_sex

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

    logger.info(f"Saved passport to {file_path}")

    # Convert PDF to image if needed
    if file_ext == ".pdf":
        from app.utils.pdf_utils import pdf_to_images

        images = pdf_to_images(file_path)
        if not images:
            raise HTTPException(status_code=422, detail="Could not convert PDF to image")
        image_path = str(images[0])
    else:
        image_path = str(file_path)

    # Initialize state
    initial_state: PassportState = {
        "image_path": image_path,
        "ocr_results": [],
        "final_data": None,
        "confidence": 0.0,
        "errors": [],
        "source": "",
        "needs_human_review": False,
        "fraud_flags": [],
        "use_llm": use_llm,
    }

    # Run the graph
    try:
        result = await graph.ainvoke(initial_state)

        # Build response
        response = {
            "success": result["final_data"] is not None,
            "file_id": file_id,
            "data": result["final_data"],
            "confidence": result["confidence"],
            "source": result["source"],
            "needs_human_review": result["needs_human_review"],
            "fraud_flags": result["fraud_flags"],
            "errors": result["errors"],
            "ocr_results_summary": [
                {
                    "source": r.get("source"),
                    "success": r.get("success"),
                    "checksum_valid": r.get("checksum_valid"),
                }
                for r in result.get("ocr_results", [])
            ],
        }

        # Convert to Pydantic if successful
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
                logger.warning(f"Could not convert to Pydantic: {e}")

        return response

    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running directly with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
