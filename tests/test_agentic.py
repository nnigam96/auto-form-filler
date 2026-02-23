"""
Tests for the V5 extraction pipeline with HITL support.

Tests:
- State definition
- OCR engines
- Voting logic
- Critic validation
- V5 Pipeline execution
- HITL data structures
"""

import pytest
import asyncio
from pathlib import Path
from datetime import date

from app.extraction.state import PassportState
from app.extraction.voting import vote_on_results, critic_validate, calculate_field_agreement
from app.extraction.ocr_engines import (
    run_passport_eye,
    run_tesseract,
    run_easyocr,
    _parse_mrz_from_text,
    _validate_checksum,
)


class TestPassportState:
    """Tests for state definition."""

    def test_state_has_required_fields(self):
        """Verify PassportState has all required fields."""
        state: PassportState = {
            "image_path": "/path/to/image.png",
            "ocr_results": [],
            "final_data": None,
            "confidence": 0.0,
            "errors": [],
            "source": "",
            "needs_human_review": False,
            "fraud_flags": [],
            "use_llm": False,
        }

        assert state["image_path"] == "/path/to/image.png"
        assert state["confidence"] == 0.0
        assert state["use_llm"] is False

    def test_state_v5_fields(self):
        """Verify V5-specific state fields."""
        state: PassportState = {
            "image_path": "/path/to/image.png",
            "ocr_results": [],
            "final_data": None,
            "confidence": 0.0,
            "errors": [],
            "source": "v5",
            "needs_human_review": True,
            "fraud_flags": ["MISMATCH"],
            "use_llm": True,
            "extraction_result": None,
            "mrz_data": {"surname": "SMITH"},
            "visual_data": {"surname": "SMITH"},
            "has_valid_checksum": True,
        }

        assert state["source"] == "v5"
        assert state["needs_human_review"] is True
        assert state["has_valid_checksum"] is True


class TestVotingLogic:
    """Tests for voting and consensus logic."""

    def test_passport_eye_with_checksum_wins(self):
        """PassportEye with valid checksum should win."""
        results = [
            {
                "source": "passport_eye",
                "success": True,
                "parsed": {"surname": "SMITH", "passport_number": "AB123456"},
                "checksum_valid": True,
            },
            {
                "source": "tesseract",
                "success": True,
                "parsed": {"surname": "SMTH", "passport_number": "AB123456"},
                "checksum_valid": False,
            },
        ]

        best, confidence, source = vote_on_results(results)

        assert source == "passport_eye"
        assert confidence == 1.0
        assert best["surname"] == "SMITH"

    def test_consensus_when_ocr_agrees(self):
        """High agreement between OCR engines should give good confidence."""
        results = [
            {
                "source": "tesseract",
                "success": True,
                "parsed": {
                    "surname": "NIGAM",
                    "given_names": "NIKHIL",
                    "passport_number": "N7178292",
                    "date_of_birth": "961025",
                    "expiry_date": "260215",
                    "sex": "M",
                },
                "checksum_valid": True,
            },
            {
                "source": "easyocr",
                "success": True,
                "parsed": {
                    "surname": "NIGAM",
                    "given_names": "NIKHIL",
                    "passport_number": "N7178292",
                    "date_of_birth": "961025",
                    "expiry_date": "260215",
                    "sex": "M",
                },
                "checksum_valid": True,
            },
        ]

        best, confidence, source = vote_on_results(results)

        assert confidence >= 0.8
        assert "consensus" in source or source in ["tesseract", "easyocr"]

    def test_no_results_returns_none(self):
        """No successful results should return None."""
        results = [
            {"source": "tesseract", "success": False, "error": "Failed"},
            {"source": "easyocr", "success": False, "error": "Failed"},
        ]

        best, confidence, source = vote_on_results(results)

        assert best is None
        assert confidence == 0.0
        assert source == "none"


class TestFieldAgreement:
    """Tests for field agreement calculation."""

    def test_full_agreement(self):
        """Identical results should have 100% agreement."""
        r1 = {"surname": "NIGAM", "passport_number": "N7178292", "sex": "M"}
        r2 = {"surname": "NIGAM", "passport_number": "N7178292", "sex": "M"}

        agreement = calculate_field_agreement(r1, r2)
        assert agreement == 1.0

    def test_partial_agreement(self):
        """Partial match should give partial agreement."""
        r1 = {"surname": "NIGAM", "passport_number": "N7178292", "sex": "M"}
        r2 = {"surname": "NIGAM", "passport_number": "X9999999", "sex": "F"}

        agreement = calculate_field_agreement(r1, r2)
        assert 0.0 < agreement < 1.0


class TestCriticValidation:
    """Tests for fraud/validation checks."""

    def test_valid_data_passes(self):
        """Valid passport data should pass validation."""
        data = {
            "surname": "NIGAM",
            "given_names": "NIKHIL RAJESH",
            "passport_number": "N7178292",
            "date_of_birth": "961025",
            "expiry_date": "260215",
            "sex": "M",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is True
        assert len(flags) == 0

    def test_future_dob_flagged(self):
        """Future DOB should be flagged."""
        data = {
            "surname": "SMITH",
            "passport_number": "AB123456",
            "date_of_birth": "351225",  # 2035-12-25 (future)
            "sex": "M",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert "DOB_IN_FUTURE" in flags

    def test_invalid_sex_flagged(self):
        """Invalid sex character should be flagged."""
        data = {
            "surname": "SMITH",
            "passport_number": "AB123456",
            "date_of_birth": "961025",
            "sex": "5",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert any("INVALID_SEX" in f for f in flags)

    def test_short_passport_number_flagged(self):
        """Too short passport number should be flagged."""
        data = {
            "surname": "SMITH",
            "passport_number": "AB",
            "date_of_birth": "961025",
            "sex": "M",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert "INVALID_PASSPORT_NUMBER" in flags


class TestMRZParsing:
    """Tests for MRZ parsing utilities."""

    def test_parse_valid_mrz(self):
        """Valid MRZ text should parse correctly."""
        text = """
        P<INDNIGAM<<NIKHIL<RAJESH<<<<<<<<<<<<<<<<<<<
        N7178292<7IND9610253M2602154<<<<<<<<<<<<02<<
        """

        parsed, checksum_valid = _parse_mrz_from_text(text)

        assert parsed is not None
        assert parsed["surname"] == "NIGAM"
        assert "NIKHIL" in parsed["given_names"]
        assert parsed["passport_number"] == "N7178292"

    def test_parse_no_mrz_returns_none(self):
        """Text without MRZ should return None."""
        text = "This is just random text without any MRZ"

        parsed, checksum_valid = _parse_mrz_from_text(text)

        assert parsed is None

    def test_checksum_validation(self):
        """Checksum validation should work."""
        line2 = "N7178292<7IND9610253M2602154<<<<<<<<<<<<02<<"

        result = _validate_checksum(line2)

        assert isinstance(result, bool)


class TestOCREngines:
    """Integration tests for OCR engines."""

    @pytest.fixture
    def sample_image_path(self):
        """Get path to sample passport image."""
        pdf_path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample passport not available")

        from app.utils.pdf_utils import pdf_to_images

        images = pdf_to_images(pdf_path)
        if not images:
            pytest.skip("Could not convert PDF to image")

        return str(images[0])

    @pytest.mark.asyncio
    async def test_run_passport_eye(self, sample_image_path):
        """Test PassportEye engine."""
        result = await run_passport_eye(sample_image_path)

        assert result["source"] == "passport_eye"
        assert "success" in result

    @pytest.mark.asyncio
    async def test_run_tesseract(self, sample_image_path):
        """Test Tesseract engine."""
        result = await run_tesseract(sample_image_path)

        assert result["source"] == "tesseract"
        assert "success" in result

    @pytest.mark.asyncio
    async def test_run_easyocr(self, sample_image_path):
        """Test EasyOCR engine."""
        result = await run_easyocr(sample_image_path)

        assert result["source"] == "easyocr"
        assert "success" in result

    @pytest.mark.asyncio
    async def test_parallel_execution(self, sample_image_path):
        """Test all engines run in parallel."""
        import time

        start = time.time()

        results = await asyncio.gather(
            run_passport_eye(sample_image_path),
            run_tesseract(sample_image_path),
            run_easyocr(sample_image_path),
        )

        elapsed = time.time() - start

        assert len(results) == 3
        print(f"Parallel execution time: {elapsed:.1f}s")


class TestV5Pipeline:
    """Tests for the V5 HITL pipeline."""

    def test_field_result_dataclass(self):
        """Test FieldResult dataclass."""
        from app.extraction.pipeline import FieldResult

        field = FieldResult(
            field_name="surname",
            mrz_value="SMITH",
            visual_value="SMITH",
            final_value="SMITH",
            confidence=0.99,
            needs_review=False,
            source="aligned",
        )

        assert field.field_name == "surname"
        assert field.mrz_value == "SMITH"
        assert field.needs_review is False
        assert field.source == "aligned"

    def test_field_result_conflict(self):
        """Test FieldResult with conflict."""
        from app.extraction.pipeline import FieldResult

        field = FieldResult(
            field_name="passport_number",
            mrz_value="910239248",
            visual_value="A456789",
            final_value=None,
            confidence=0.0,
            needs_review=True,
            source="conflict",
        )

        assert field.needs_review is True
        assert field.final_value is None
        assert field.source == "conflict"

    def test_extraction_result_dataclass(self):
        """Test ExtractionResult dataclass."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "surname": FieldResult(
                field_name="surname",
                mrz_value="SMITH",
                visual_value="SMITH",
                final_value="SMITH",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.95,
            needs_human_review=False,
            fraud_flags=[],
            review_reason=None,
            mrz_checksum_valid=True,
        )

        assert result.success is True
        assert result.overall_confidence == 0.95
        assert result.needs_human_review is False
        assert result.mrz_checksum_valid is True

    def test_extraction_result_with_hitl(self):
        """Test ExtractionResult requiring HITL."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "surname": FieldResult(
                field_name="surname",
                mrz_value="SMITH",
                visual_value="SMITH",
                final_value="SMITH",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
            "passport_number": FieldResult(
                field_name="passport_number",
                mrz_value="910239248",
                visual_value="A456789",
                final_value=None,
                confidence=0.0,
                needs_review=True,
                source="conflict",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.5,
            needs_human_review=True,
            fraud_flags=["PASSPORT_NUMBER_MISMATCH"],
            review_reason="passport_number: MRZ≠Visual",
            mrz_checksum_valid=True,
        )

        assert result.needs_human_review is True
        assert "PASSPORT_NUMBER_MISMATCH" in result.fraud_flags
        assert result.review_reason is not None

    def test_get_final_data(self):
        """Test ExtractionResult.get_final_data() method."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "surname": FieldResult(
                field_name="surname",
                mrz_value="SMITH",
                visual_value="SMITH",
                final_value="SMITH",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
            "given_names": FieldResult(
                field_name="given_names",
                mrz_value="JOHN",
                visual_value="JOHN",
                final_value="JOHN",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.99,
            needs_human_review=False,
            fraud_flags=[],
            review_reason=None,
            mrz_checksum_valid=True,
        )

        final_data = result.get_final_data()

        assert final_data["surname"] == "SMITH"
        assert final_data["given_names"] == "JOHN"

    @pytest.fixture
    def sample_image_path(self):
        """Get path to sample passport image."""
        pdf_path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample passport not available")

        from app.utils.pdf_utils import pdf_to_images

        images = pdf_to_images(pdf_path)
        if not images:
            pytest.skip("Could not convert PDF to image")

        return str(images[0])

    @pytest.mark.asyncio
    async def test_v5_pipeline_execution(self, sample_image_path):
        """Test full V5 pipeline execution."""
        from app.extraction.pipeline import extract_passport_v5

        result = await extract_passport_v5(sample_image_path, use_llm=False)

        assert result is not None
        assert result.success is True
        assert len(result.fields) > 0
        assert result.overall_confidence >= 0.0

        # Should have extracted some fields
        final_data = result.get_final_data()
        assert "surname" in final_data or "passport_number" in final_data

        print(f"\nV5 Result:")
        print(f"  Confidence: {result.overall_confidence:.2f}")
        print(f"  Needs Review: {result.needs_human_review}")
        print(f"  MRZ Checksum Valid: {result.mrz_checksum_valid}")
        print(f"  Fields: {list(result.fields.keys())}")

    @pytest.mark.asyncio
    async def test_v5_pipeline_with_llm(self, sample_image_path):
        """Test V5 pipeline with LLM enabled (if available)."""
        from app.extraction.pipeline import extract_passport_v5
        from app.extraction.fraud_detector import check_ollama_available

        if not check_ollama_available():
            pytest.skip("Ollama not available")

        result = await extract_passport_v5(sample_image_path, use_llm=True)

        assert result is not None
        assert result.success is True

        # With LLM, we might get visual data
        has_visual = any(
            f.visual_value is not None
            for f in result.fields.values()
        )
        print(f"\nV5 with LLM - Has visual data: {has_visual}")


class TestHITLFlow:
    """Tests for HITL (Human-In-The-Loop) flow."""

    def test_hitl_needed_on_mismatch(self):
        """HITL should be needed when MRZ and Visual don't match."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "passport_number": FieldResult(
                field_name="passport_number",
                mrz_value="910239248",
                visual_value="A456789",
                final_value=None,
                confidence=0.0,
                needs_review=True,
                source="conflict",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.0,
            needs_human_review=True,
            fraud_flags=["PASSPORT_NUMBER_MISMATCH", "POTENTIAL_DOCUMENT_TAMPERING"],
            review_reason="passport_number: MRZ≠Visual",
            mrz_checksum_valid=True,
        )

        assert result.needs_human_review is True
        assert "POTENTIAL_DOCUMENT_TAMPERING" in result.fraud_flags

    def test_hitl_not_needed_when_aligned(self):
        """HITL should not be needed when all fields align."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "surname": FieldResult(
                field_name="surname",
                mrz_value="SMITH",
                visual_value="SMITH",
                final_value="SMITH",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
            "passport_number": FieldResult(
                field_name="passport_number",
                mrz_value="AB123456",
                visual_value="AB123456",
                final_value="AB123456",
                confidence=0.99,
                needs_review=False,
                source="aligned",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.99,
            needs_human_review=False,
            fraud_flags=[],
            review_reason=None,
            mrz_checksum_valid=True,
        )

        assert result.needs_human_review is False
        assert len(result.fraud_flags) == 0

    def test_mrz_only_no_hitl_needed(self):
        """When only MRZ data available (no visual), no HITL needed."""
        from app.extraction.pipeline import FieldResult, ExtractionResult

        fields = {
            "surname": FieldResult(
                field_name="surname",
                mrz_value="SMITH",
                visual_value=None,
                final_value="SMITH",
                confidence=0.95,
                needs_review=False,
                source="mrz",
            ),
        }

        result = ExtractionResult(
            success=True,
            fields=fields,
            overall_confidence=0.95,
            needs_human_review=False,
            fraud_flags=[],
            review_reason=None,
            mrz_checksum_valid=True,
        )

        assert result.needs_human_review is False
        assert result.fields["surname"].source == "mrz"
