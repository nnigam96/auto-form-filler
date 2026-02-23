# OCR Architecture Comparison

## The Problem

Given a passport image, we run multiple OCR engines. Each produces slightly different results:

```
PassportEye: { surname: "NIGAM",    passport: "N7178292", dob: "961025" } ✓ checksum
Tesseract:   { surname: "N1GAM",    passport: "N7178292", dob: "961025" } ✓ checksum
EasyOCR:     { surname: "NIGAMKKK", passport: "N7I78292", dob: "96I025" } ✗ checksum
```

How do we get the best possible final result?

---

## Approach 1: Result-Level Voting (Current V2)

**Strategy**: Pick the single best result, use all its fields.

```python
# Current logic.py
def vote_on_results(results):
    # Rule 1: PassportEye with checksum wins
    if passporteye.checksum_valid:
        return passporteye  # Use ALL fields from PassportEye

    # Rule 2: Tesseract/EasyOCR agreement
    if agreement >= 80%:
        return tesseract  # Use ALL fields from Tesseract

    # Rule 3: Any checksum-valid result
    return first_valid_result
```

**Output**:
```
{ surname: "NIGAM", passport: "N7178292", dob: "961025" }
```

**Problem**: What if PassportEye got the surname wrong but Tesseract got it right?

```
PassportEye: { surname: "NIGAM",  passport: "N7178292" } ✓ checksum  ← WINNER
Tesseract:   { surname: "NIKHIL", passport: "N7178292" } ✓ checksum
```

You get "NIGAM" when "NIKHIL" was correct because you picked the whole result.

---

## Approach 2: Reflection Agent (Current V1)

**Strategy**: Pick one result, then use LLM to fix errors.

```python
# Current flow
result = ocr_service.extract()  # Single best result
if has_errors(result):
    result = reflection_agent.fix(result, image)  # LLM tries to fix
```

**Problems**:

1. **Garbage In, Garbage Out**: If OCR produces "NIKHILKKKKKKKKK", reflection agent is polishing garbage
2. **LLM Dependency**: Requires Ollama running, adds latency
3. **Inconsistent**: LLM might hallucinate or make things worse
4. **Single Source**: Still using one OCR result as base

---

## Approach 3: Field-Level Aggregation (Proposed)

**Strategy**: For EACH field, pick the best value across ALL sources.

```python
# New aggregator.py
def aggregate_ocr_results(results):
    for field in ["surname", "given_names", "passport_number", ...]:
        # Collect all values for this field
        values = [r.parsed[field] for r in successful_results]

        # Pick best using field-specific strategy
        if field in MRZ_FIELDS:
            best = use_mrz_ground_truth(values)
        elif field in NAME_FIELDS:
            best = majority_vote_with_cleanup(values)
        else:
            best = majority_vote(values)
```

**Example**:

| Field | PassportEye | Tesseract | EasyOCR | Strategy | Result |
|-------|-------------|-----------|---------|----------|--------|
| surname | NIGAM | NIKHIL | NIGAMKKK | Clean + majority → "NIGAM" vs "NIKHIL" | NIGAM (2-1) |
| given_names | RAJESH | RAJESH | RAJESHKK | Clean + majority | RAJESH (3-0) |
| passport_number | N7178292 | N7178292 | N7I78292 | MRZ ground truth | N7178292 |
| dob | 961025 | 961025 | 96I025 | MRZ ground truth | 961025 |

**Benefits**:
- Best of all engines, not just one winner
- MRZ provides ground truth for critical fields
- Name cleaning catches garbage characters
- Per-field confidence scores identify weak spots
- Reflection agent only needed for low-confidence fields

---

## Reliability Comparison

| Scenario | Result Voting | Reflection | Field Aggregation |
|----------|---------------|------------|-------------------|
| All engines agree | ✅ Works | ✅ Works | ✅ Works |
| One engine wrong on one field | ❌ Might pick wrong | ⚠️ LLM might fix | ✅ Majority wins |
| All engines have same error | ❌ Error propagates | ⚠️ LLM might fix | ❌ Error propagates |
| Garbage characters in names | ❌ Picks garbage | ⚠️ LLM might clean | ✅ Cleaned automatically |
| MRZ readable but OCR fails | ❌ No MRZ usage | ✅ Uses MRZ | ✅ Uses MRZ |
| Ollama unavailable | ✅ Still works | ❌ No correction | ✅ Still works |

---

## Recommended Architecture

```
                    ┌─────────────────┐
                    │  Input (PDF/IMG)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │PassportEye│   │Tesseract │   │ EasyOCR  │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  Field-Level Aggregator │
                │  • MRZ ground truth     │
                │  • Majority voting      │
                │  • Name cleanup         │
                │  • Per-field confidence │
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  Low Confidence Check   │
                │  (any field < 0.7?)     │
                └────────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES                         │ NO
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Reflection Agent │           │   Final Result   │
    │ (only for weak   │           │   (high conf)    │
    │  fields + image) │           └─────────────────┘
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Final Result    │
    │  (reflected)     │
    └─────────────────┘
```

---

## Summary

| Approach | Reliability | Speed | Complexity | LLM Required |
|----------|-------------|-------|------------|--------------|
| Result Voting (V2) | ⭐⭐ | Fast | Low | No |
| Reflection (V1) | ⭐⭐⭐ | Slow | Medium | Yes |
| Field Aggregation | ⭐⭐⭐⭐ | Fast | Medium | No (optional) |
| Aggregation + Reflection | ⭐⭐⭐⭐⭐ | Medium | High | Optional |

**Recommendation**: Use Field-Level Aggregation as the primary strategy, with Reflection Agent as an optional enhancement for low-confidence fields only.
