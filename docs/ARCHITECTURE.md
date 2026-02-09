# Architecture

## Overview

This document describes the system architecture for the document automation pipeline, focusing on the agentic extraction system built with LangGraph.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Web Interface                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │   Upload    │  │   Results   │  │      HITL Modal         │   │
│  │   Forms     │  │   Display   │  │  (Conflict Resolution)  │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ /extract/      │  │ /upload/g28    │  │ /fill-form       │   │
│  │   passport     │  │                │  │                  │   │
│  └───────┬────────┘  └───────┬────────┘  └────────┬─────────┘   │
└──────────┼───────────────────┼────────────────────┼──────────────┘
           │                   │                    │
           ▼                   ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ LangGraph        │  │ G-28 Extractor   │  │ Playwright       │
│ Pipeline         │  │ (PDF Fields)     │  │ Form Filler      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

## LangGraph Pipeline

The passport extraction uses a **StateGraph** workflow with the following nodes:

```
┌─────────────────┐
│  parallel_ocr   │  ← Run 3 OCR engines concurrently
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  aggregate_mrz  │  ← Field-level aggregation from OCR results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ visual_extract  │  ← LLM Vision reads printed text (optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│compare_decide   │  ← Cross-validate MRZ vs Visual, set HITL flag
└────────┬────────┘
         │
         ▼
       [END]
```

### State Definition

```python
class PassportState(TypedDict):
    image_path: str
    ocr_results: List[dict]       # Results from each OCR engine
    mrz_data: Dict[str, Any]      # Aggregated MRZ fields
    visual_data: Dict[str, Any]   # LLM-extracted visual fields
    extraction_result: ExtractionResult  # Final output with HITL info
    needs_human_review: bool
    fraud_flags: List[str]
    has_valid_checksum: bool
```

### Output Structure

```python
@dataclass
class FieldResult:
    field_name: str
    mrz_value: Any        # Value from MRZ OCR
    visual_value: Any     # Value from LLM Vision
    final_value: Any      # Resolved value (or None if conflict)
    confidence: float
    needs_review: bool
    source: str           # "aligned", "mrz", "visual", "conflict"

@dataclass
class ExtractionResult:
    success: bool
    fields: Dict[str, FieldResult]
    overall_confidence: float
    needs_human_review: bool
    fraud_flags: List[str]
    review_reason: Optional[str]
    mrz_checksum_valid: bool
```

## Agentic Patterns

### Pattern 1: Parallel Tool Execution

**Problem**: Single OCR engine may fail or produce errors on specific images.

**Solution**: Run multiple OCR engines in parallel, aggregate results.

```python
results = await asyncio.gather(
    run_passport_eye(image_path),
    run_tesseract(image_path),
    run_easyocr(image_path),
    return_exceptions=True,
)
```

**Trade-offs**:
- (+) Redundancy: if one fails, others may succeed
- (+) Speed: parallel execution vs sequential
- (-) Resource usage: 3x memory/CPU during extraction
- (-) Complexity: need aggregation logic

### Pattern 2: Field-Level Aggregation

**Problem**: Result-level voting picks one "winner" but each engine has different strengths.

**Solution**: Aggregate best value per field based on source reliability.

```
Field Priority:
1. Checksum-validated MRZ value (highest trust)
2. Majority vote across engines
3. Single source with highest confidence
```

**Example**:
```
surname:         PassportEye="SMITH", Tesseract="SMTH", EasyOCR="SMITH"
                 → Final: "SMITH" (majority)

passport_number: PassportEye="AB123456" (checksum valid)
                 → Final: "AB123456" (checksum wins)
```

### Pattern 3: Cross-Validation with LLM Vision

**Problem**: MRZ data is machine-encoded; visual text is what humans see. Tampering may alter one but not both.

**Solution**: Use LLM to read visual text, compare with MRZ.

```
MRZ (checksummed):  passport_number = "910239248"
Visual (LLM):       passport_number = "A456789"

→ MISMATCH DETECTED
→ Fraud flag: "PASSPORT_NUMBER_MISMATCH"
→ needs_human_review = True
```

**Key Insight**: MRZ with valid checksum is cryptographically verified. If visual differs, either:
1. LLM hallucinated (common)
2. Document was tampered (rare but serious)

### Pattern 4: Human-In-The-Loop (HITL)

**Problem**: Automated systems can't resolve all conflicts with certainty.

**Solution**: Surface conflicts to user with both values, let human decide.

```
┌─────────────────────────────────────────────┐
│  passport_number: MRZ ≠ Visual              │
│                                             │
│  ┌──────────────┐    ┌──────────────┐      │
│  │ MRZ          │    │ Visual       │      │
│  │ 910239248    │    │ A456789      │      │
│  │ [Selected]   │    │              │      │
│  └──────────────┘    └──────────────┘      │
│                                             │
│              [Confirm Selections]           │
└─────────────────────────────────────────────┘
```

### Pattern 5: Graceful Degradation

**Problem**: External dependencies (LLM, OCR) can fail.

**Solution**: Fallback chain with confidence adjustment.

```python
# LLM refusal detection
if "unable to assist" in response.lower():
    logger.warning("LLM refused, falling back to MRZ-only")
    return None  # Pipeline continues with MRZ data only

# OCR failure handling
for result in ocr_results:
    if isinstance(result, Exception):
        ocr_results.append({"source": source, "success": False})
        # Other engines may still succeed
```

## System Design Decisions

### Why MRZ is Ground Truth

| Aspect | MRZ | Visual Text |
|--------|-----|-------------|
| Validation | Checksum-verified | None |
| Tampering | Breaks checksum | Undetectable |
| OCR Accuracy | High (fixed format) | Variable |
| Fields | Subset (no place of birth) | Complete |

**Decision**: Trust MRZ when checksum valid. Use visual for fields not in MRZ (place_of_birth, issue_date).

### Why Local LLM (Ollama)

| Option | Pros | Cons |
|--------|------|------|
| OpenAI GPT-4V | Best accuracy | Privacy, cost, latency |
| Claude Vision | Good accuracy | Same issues |
| Ollama (local) | Privacy, free, fast | Lower accuracy, GPU needed |

**Decision**: Default to Ollama for privacy. User can enable cloud LLM if needed.

### Why HITL vs Auto-Resolution

| Approach | When to Use |
|----------|-------------|
| Auto (trust MRZ) | Checksum valid, high confidence |
| Auto (trust Visual) | Field not in MRZ |
| HITL | Conflict between sources |

**Decision**: Never auto-resolve conflicts. The cost of error (wrong passport data in legal filing) exceeds cost of human review.

## Data Flow

### Happy Path (No Conflicts)

```
1. Upload passport image
2. parallel_ocr: 3 engines extract MRZ
3. aggregate_mrz: combine into single result
4. visual_extract: LLM reads printed text
5. compare_decide: MRZ == Visual for all fields
6. Return: needs_human_review=False, confidence=0.99
```

### Conflict Path (HITL Required)

```
1. Upload passport image
2. parallel_ocr: 3 engines extract MRZ
3. aggregate_mrz: passport_number="910239248"
4. visual_extract: LLM reads passport_number="A456789"
5. compare_decide: MISMATCH detected
6. Return: needs_human_review=True, fraud_flags=["MISMATCH"]
7. UI shows HITL modal
8. User selects MRZ value
9. POST /confirm with corrections
10. Return: final verified data
```

### Degraded Path (LLM Unavailable)

```
1. Upload passport image
2. parallel_ocr: 3 engines extract MRZ
3. aggregate_mrz: combine into single result
4. visual_extract: LLM refuses or unavailable
5. compare_decide: MRZ only, no cross-validation
6. Return: needs_human_review=False, confidence=0.95
```

## Performance Considerations

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| PDF to Image | 200-500ms | Poppler conversion |
| PassportEye | 1-2s | MRZ detection |
| Tesseract | 2-3s | Full page OCR |
| EasyOCR | 3-5s | Deep learning inference |
| LLM Vision | 5-30s | Depends on model/hardware |
| **Total (parallel)** | **5-30s** | Bottleneck is LLM |

**Optimization**: OCR runs in parallel. LLM is optional and can be disabled for speed.

## Security Considerations

1. **File Validation**: Only PDF/JPEG/PNG accepted
2. **Temp File Cleanup**: Uploaded files deleted after processing
3. **No Data Persistence**: Extraction results not stored
4. **Local LLM**: Sensitive data never leaves server (when using Ollama)
5. **HITL Logging**: User corrections logged for audit trail

## Future Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Planned Enhancements                        │
├─────────────────────────────────────────────────────────────────┤
│  • Batch Processing Queue (Redis/Celery)                        │
│  • HITL Correction Database (for fine-tuning)                   │
│  • Face Matching Service (photo verification)                   │
│  • Document Type Classifier (passport/visa/I-94)                │
│  • Confidence Calibration (historical accuracy tracking)        │
└─────────────────────────────────────────────────────────────────┘
```
