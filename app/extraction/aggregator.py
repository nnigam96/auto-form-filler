"""
Field-Level Aggregator for OCR Results.

Instead of picking one "winner" result, this aggregates the BEST VALUE
for each field across all OCR sources.

Strategy:
1. Run all OCR engines in parallel
2. For each field, collect all values from successful engines
3. Use field-specific strategies to pick the best value:
   - MRZ fields (passport_number, nationality, DOB, sex, expiry): MRZ is ground truth
   - Name fields: Majority vote + cleanup heuristics
   - Checksum-validated values get priority
4. Build composite result with per-field confidence
5. Optionally run reflection agent on low-confidence fields only

This is more reliable than result-level voting because:
- One engine might be good at names but bad at dates
- Another might read passport numbers perfectly but mangle names
- Aggregating per-field gets the best of each engine
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FieldValue:
    """A single field value from one OCR source."""
    value: Any
    source: str
    confidence: float
    checksum_validated: bool = False


@dataclass
class AggregatedField:
    """Aggregated result for a single field."""
    value: Any
    confidence: float
    sources: List[str]
    method: str  # "mrz_ground_truth", "majority", "highest_confidence", "single_source"


@dataclass
class AggregatedResult:
    """Complete aggregated passport data."""
    fields: Dict[str, AggregatedField]
    overall_confidence: float
    sources_used: List[str]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for PassportData construction."""
        return {
            name: af.value
            for name, af in self.fields.items()
        }

    def get_low_confidence_fields(self, threshold: float = 0.7) -> List[str]:
        """Get field names with confidence below threshold."""
        return [
            name for name, af in self.fields.items()
            if af.confidence < threshold
        ]


# =============================================================================
# Field-Specific Aggregation Strategies
# =============================================================================

def aggregate_mrz_field(
    field_name: str,
    values: List[FieldValue],
    mrz_value: Optional[str] = None
) -> AggregatedField:
    """
    Aggregate MRZ-sourced fields. MRZ is ground truth when available.

    Fields: passport_number, nationality, date_of_birth, sex, expiry_date
    """
    # If we have MRZ ground truth, use it
    if mrz_value:
        # Find sources that agree with MRZ
        agreeing_sources = [v.source for v in values if _normalize(v.value) == _normalize(mrz_value)]

        return AggregatedField(
            value=mrz_value,
            confidence=1.0 if agreeing_sources else 0.95,
            sources=agreeing_sources or ["mrz"],
            method="mrz_ground_truth"
        )

    # No MRZ - use checksum-validated values first
    checksum_valid = [v for v in values if v.checksum_validated]
    if checksum_valid:
        # Majority among checksum-validated
        value_counts = Counter(_normalize(v.value) for v in checksum_valid)
        best_value, count = value_counts.most_common(1)[0]

        # Find original (non-normalized) value
        original = next(v.value for v in checksum_valid if _normalize(v.value) == best_value)
        sources = [v.source for v in checksum_valid if _normalize(v.value) == best_value]

        return AggregatedField(
            value=original,
            confidence=0.9 if count > 1 else 0.8,
            sources=sources,
            method="checksum_majority" if count > 1 else "checksum_single"
        )

    # Fallback: majority vote
    return aggregate_by_majority(field_name, values)


def aggregate_name_field(
    field_name: str,
    values: List[FieldValue],
    mrz_value: Optional[str] = None
) -> AggregatedField:
    """
    Aggregate name fields (surname, given_names).

    Strategy:
    1. Clean all values (remove garbage chars)
    2. If MRZ available, use as reference
    3. Majority vote with fuzzy matching
    4. Prefer longer, cleaner values
    """
    if not values:
        return AggregatedField(
            value="",
            confidence=0.0,
            sources=[],
            method="no_data"
        )

    # Clean all values
    cleaned = [(v, _clean_name(v.value)) for v in values]
    cleaned = [(v, c) for v, c in cleaned if c]  # Remove empty

    if not cleaned:
        return AggregatedField(
            value=values[0].value if values else "",
            confidence=0.3,
            sources=[values[0].source] if values else [],
            method="uncleaned_fallback"
        )

    # If MRZ available, use it but prefer visual text for casing
    if mrz_value:
        mrz_clean = _clean_name(mrz_value)

        # Find values that match MRZ (case-insensitive)
        matches = [(v, c) for v, c in cleaned if c.upper() == mrz_clean.upper()]

        if matches:
            # Prefer properly cased version
            best = max(matches, key=lambda x: _case_quality(x[1]))
            return AggregatedField(
                value=best[1],
                confidence=0.95,
                sources=[v.source for v, _ in matches],
                method="mrz_validated"
            )
        else:
            # MRZ doesn't match any - trust MRZ but flag it
            return AggregatedField(
                value=mrz_clean.title(),
                confidence=0.85,
                sources=["mrz"],
                method="mrz_only"
            )

    # No MRZ - majority vote with fuzzy matching
    # Group by similarity
    groups = _group_by_similarity([c for _, c in cleaned])

    if not groups:
        best = max(cleaned, key=lambda x: (len(x[1]), x[0].confidence))
        return AggregatedField(
            value=best[1],
            confidence=best[0].confidence * 0.7,
            sources=[best[0].source],
            method="best_single"
        )

    # Find largest group
    largest_group = max(groups, key=len)

    # Get sources for this group
    group_values = set(largest_group)
    matching = [(v, c) for v, c in cleaned if c in group_values]

    # Pick best representation (proper casing, longest)
    best = max(matching, key=lambda x: (_case_quality(x[1]), len(x[1])))

    return AggregatedField(
        value=best[1],
        confidence=0.8 if len(matching) > 1 else 0.6,
        sources=[v.source for v, _ in matching],
        method="majority" if len(matching) > 1 else "single"
    )


def aggregate_by_majority(
    field_name: str,
    values: List[FieldValue]
) -> AggregatedField:
    """Generic majority vote aggregation."""
    if not values:
        return AggregatedField(
            value=None,
            confidence=0.0,
            sources=[],
            method="no_data"
        )

    # Normalize and count
    normalized = [(_normalize(v.value), v) for v in values]
    value_counts = Counter(n for n, _ in normalized)

    if not value_counts:
        return AggregatedField(
            value=values[0].value,
            confidence=0.4,
            sources=[values[0].source],
            method="fallback"
        )

    best_normalized, count = value_counts.most_common(1)[0]

    # Find original value and sources
    matching = [(n, v) for n, v in normalized if n == best_normalized]
    original_value = matching[0][1].value
    sources = [v.source for _, v in matching]

    # Confidence based on agreement
    total = len(values)
    confidence = (count / total) * 0.9 if count > 1 else 0.5

    return AggregatedField(
        value=original_value,
        confidence=confidence,
        sources=sources,
        method="majority" if count > 1 else "single"
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize(value: Any) -> str:
    """Normalize value for comparison."""
    if value is None:
        return ""
    return str(value).strip().upper().replace(" ", "").replace("<", "")


def _clean_name(name: str) -> str:
    """Clean garbage from name fields."""
    if not name:
        return ""

    import re

    # Remove repeated characters (3+ in a row)
    cleaned = re.sub(r'(.)\1{2,}', r'\1', str(name))

    # Remove trailing/leading garbage
    cleaned = re.sub(r'^[^a-zA-Z]+', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z]+$', '', cleaned)

    # Remove digit sequences
    cleaned = re.sub(r'\d+', '', cleaned)

    # Remove common OCR garbage patterns
    cleaned = re.sub(r'[kK]{2,}', '', cleaned)

    return cleaned.strip()


def _case_quality(text: str) -> int:
    """Score how well a string is cased (prefer Title Case)."""
    if not text:
        return 0

    # Title case is best
    if text == text.title():
        return 3
    # All caps is okay (common in passports)
    if text == text.upper():
        return 2
    # Mixed is worst
    return 1


def _group_by_similarity(values: List[str], threshold: float = 0.8) -> List[List[str]]:
    """Group similar strings together."""
    if not values:
        return []

    groups = []
    used = set()

    for v in values:
        if v in used:
            continue

        # Find all similar values
        group = [v]
        used.add(v)

        for other in values:
            if other in used:
                continue
            if _similarity(v, other) >= threshold:
                group.append(other)
                used.add(other)

        groups.append(group)

    return groups


def _similarity(s1: str, s2: str) -> float:
    """Calculate string similarity (0-1)."""
    if not s1 or not s2:
        return 0.0

    s1, s2 = s1.upper(), s2.upper()

    if s1 == s2:
        return 1.0

    # Levenshtein-like similarity
    matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
    return matches / max(len(s1), len(s2))


# =============================================================================
# Main Aggregation Function
# =============================================================================

def aggregate_ocr_results(
    ocr_results: List[Dict[str, Any]],
    mrz_lines: Optional[List[str]] = None
) -> AggregatedResult:
    """
    Aggregate multiple OCR results into a single best result.

    Args:
        ocr_results: List of OCR result dicts, each with:
            - source: str (e.g., "tesseract", "easyocr", "passporteye")
            - success: bool
            - parsed: dict with extracted fields
            - checksum_valid: bool
        mrz_lines: Optional MRZ lines for ground truth

    Returns:
        AggregatedResult with best value for each field
    """
    # Filter successful results
    successful = [r for r in ocr_results if r.get("success") and r.get("parsed")]

    if not successful:
        return AggregatedResult(
            fields={},
            overall_confidence=0.0,
            sources_used=[],
            errors=["No successful OCR results"]
        )

    # Parse MRZ for ground truth
    mrz_truth = {}
    if mrz_lines and len(mrz_lines) >= 2:
        mrz_truth = _parse_mrz_ground_truth(mrz_lines)

    # Collect field values from all sources
    field_values: Dict[str, List[FieldValue]] = {}

    # Define expected fields
    all_fields = [
        "surname", "given_names", "passport_number", "nationality",
        "date_of_birth", "sex", "expiry_date", "country"
    ]

    for field_name in all_fields:
        field_values[field_name] = []

    for result in successful:
        parsed = result["parsed"]
        source = result.get("source", "unknown")
        checksum_valid = result.get("checksum_valid", False)
        base_confidence = result.get("confidence", 0.5)

        for field_name in all_fields:
            if field_name in parsed and parsed[field_name]:
                field_values[field_name].append(FieldValue(
                    value=parsed[field_name],
                    source=source,
                    confidence=base_confidence,
                    checksum_validated=checksum_valid
                ))

    # Aggregate each field using appropriate strategy
    aggregated_fields = {}

    # MRZ-sourced fields (highest reliability from MRZ)
    mrz_fields = ["passport_number", "nationality", "date_of_birth", "sex", "expiry_date"]
    for field_name in mrz_fields:
        mrz_val = mrz_truth.get(f"{field_name}_mrz") or mrz_truth.get(field_name)
        aggregated_fields[field_name] = aggregate_mrz_field(
            field_name,
            field_values.get(field_name, []),
            mrz_val
        )

    # Name fields (need cleaning, use visual + MRZ)
    for field_name in ["surname", "given_names"]:
        mrz_val = mrz_truth.get(f"{field_name}_mrz") or mrz_truth.get(field_name)
        aggregated_fields[field_name] = aggregate_name_field(
            field_name,
            field_values.get(field_name, []),
            mrz_val
        )

    # Country (simple majority)
    aggregated_fields["country"] = aggregate_by_majority(
        "country",
        field_values.get("country", [])
    )

    # Calculate overall confidence
    confidences = [af.confidence for af in aggregated_fields.values() if af.value]
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Collect all sources used
    all_sources = set()
    for af in aggregated_fields.values():
        all_sources.update(af.sources)

    return AggregatedResult(
        fields=aggregated_fields,
        overall_confidence=overall_confidence,
        sources_used=list(all_sources)
    )


def _parse_mrz_ground_truth(mrz_lines: List[str]) -> Dict[str, str]:
    """Extract ground truth values from MRZ lines."""
    if len(mrz_lines) < 2:
        return {}

    line1 = mrz_lines[0].upper().replace(" ", "").ljust(44, "<")[:44]
    line2 = mrz_lines[1].upper().replace(" ", "").ljust(44, "<")[:44]

    truth = {}

    # Line 1: P<COUNTRY<SURNAME<<GIVEN<NAMES
    try:
        names = line1[5:44].split("<<", 1)
        truth["surname_mrz"] = names[0].replace("<", "").strip()
        truth["given_names_mrz"] = names[1].replace("<", " ").strip() if len(names) > 1 else ""
        truth["country_mrz"] = line1[2:5]
    except:
        pass

    # Line 2
    try:
        truth["passport_number_mrz"] = line2[0:9].replace("<", "").strip()
        truth["nationality_mrz"] = line2[10:13]
        truth["date_of_birth_mrz"] = line2[13:19]
        truth["sex_mrz"] = line2[20]
        truth["expiry_date_mrz"] = line2[21:27]
    except:
        pass

    return truth


# =============================================================================
# Async Wrapper for Full Pipeline
# =============================================================================

async def aggregate_passport_extraction(file_path: Path) -> AggregatedResult:
    """
    Run all OCR engines in parallel and aggregate results.

    This is the recommended entry point for passport extraction.
    """
    from app.extraction.ocr_engines import run_passport_eye, run_tesseract, run_easyocr

    # Run all OCR engines in parallel
    results = await asyncio.gather(
        run_passport_eye(str(file_path)),
        run_tesseract(str(file_path)),
        run_easyocr(str(file_path)),
        return_exceptions=True
    )

    # Convert exceptions to error dicts
    ocr_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            ocr_results.append({
                "source": ["passport_eye", "tesseract", "easyocr"][i],
                "success": False,
                "error": str(r)
            })
        else:
            ocr_results.append(r)

    # Log individual results
    for r in ocr_results:
        status = "✓" if r.get("success") else "✗"
        logger.info(f"  {status} {r.get('source')}: checksum={r.get('checksum_valid')}")

    # Extract MRZ lines from first successful result
    mrz_lines = None
    for r in ocr_results:
        if r.get("success") and r.get("raw_text"):
            # Try to extract MRZ lines from raw text
            mrz_lines = _extract_mrz_lines_from_text(r.get("raw_text", ""))
            if mrz_lines:
                break

    # Aggregate
    result = aggregate_ocr_results(ocr_results, mrz_lines)

    logger.info(f"Aggregation complete: confidence={result.overall_confidence:.2f}, sources={result.sources_used}")

    return result


def _extract_mrz_lines_from_text(text: str) -> Optional[List[str]]:
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

    # Find P< line and following line
    for i, line in enumerate(mrz_candidates):
        if line.startswith('P') and i + 1 < len(mrz_candidates):
            return [line, mrz_candidates[i + 1]]

    return None
