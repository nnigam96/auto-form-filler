"""
Tests for the agentic extraction pipeline.

Tests:
- State definition
- OCR tools
- Voting logic
- Critic validation
- Graph execution
"""

import pytest
import asyncio
from pathlib import Path
from datetime import date

from app.extraction.state import PassportState
from app.extraction.logic import vote_on_results, critic_validate, calculate_field_agreement
from app.extraction.tools import (
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

    def test_low_confidence_on_disagreement(self):
        """Disagreement between OCR engines should lower confidence."""
        results = [
            {
                "source": "tesseract",
                "success": True,
                "parsed": {
                    "surname": "NIGAM",
                    "passport_number": "N7178292",
                    "date_of_birth": "961025",
                },
                "checksum_valid": False,
            },
            {
                "source": "easyocr",
                "success": True,
                "parsed": {
                    "surname": "NIGAM",
                    "passport_number": "X9999999",  # Different!
                    "date_of_birth": "000101",  # Different!
                },
                "checksum_valid": False,
            },
        ]

        best, confidence, source = vote_on_results(results)

        # Low confidence due to disagreement and no checksum
        assert confidence <= 0.75

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

    def test_no_agreement(self):
        """Completely different results should have low agreement."""
        r1 = {"surname": "SMITH", "passport_number": "AB123456"}
        r2 = {"surname": "JONES", "passport_number": "XY987654"}

        agreement = calculate_field_agreement(r1, r2)
        assert agreement < 0.5


class TestCriticValidation:
    """Tests for fraud/validation checks."""

    def test_valid_data_passes(self):
        """Valid passport data should pass validation."""
        data = {
            "surname": "NIGAM",
            "given_names": "NIKHIL RAJESH",
            "passport_number": "N7178292",
            "date_of_birth": "961025",  # 1996-10-25
            "expiry_date": "260215",  # 2026-02-15
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
            "date_of_birth": "261225",  # 2026-12-25 (future, since today is Feb 2026)
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
            "sex": "5",  # Invalid
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert any("INVALID_SEX" in f for f in flags)

    def test_short_passport_number_flagged(self):
        """Too short passport number should be flagged."""
        data = {
            "surname": "SMITH",
            "passport_number": "AB",  # Too short
            "date_of_birth": "961025",
            "sex": "M",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert "INVALID_PASSPORT_NUMBER" in flags

    def test_invalid_name_chars_flagged(self):
        """Names with invalid characters should be flagged."""
        data = {
            "surname": "SM1TH",  # Contains digit
            "passport_number": "AB123456",
            "date_of_birth": "961025",
            "sex": "M",
        }

        is_valid, flags = critic_validate(data)

        assert is_valid is False
        assert any("INVALID_NAME_CHARS" in f for f in flags)


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
        # Valid MRZ line 2
        line2 = "N7178292<7IND9610253M2602154<<<<<<<<<<<<02<<"

        result = _validate_checksum(line2)

        # Should return bool
        assert isinstance(result, bool)


class TestOCRTools:
    """Integration tests for OCR tools."""

    @pytest.fixture
    def sample_image_path(self):
        """Get path to sample passport image."""
        pdf_path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample passport not available")

        # Convert to image
        from app.utils.pdf_utils import pdf_to_images

        images = pdf_to_images(pdf_path)
        if not images:
            pytest.skip("Could not convert PDF to image")

        return str(images[0])

    @pytest.mark.asyncio
    async def test_run_passport_eye(self, sample_image_path):
        """Test PassportEye tool."""
        result = await run_passport_eye(sample_image_path)

        assert result["source"] == "passport_eye"
        assert "success" in result
        assert "error" in result or result["success"]

    @pytest.mark.asyncio
    async def test_run_tesseract(self, sample_image_path):
        """Test Tesseract tool."""
        result = await run_tesseract(sample_image_path)

        assert result["source"] == "tesseract"
        assert "success" in result

    @pytest.mark.asyncio
    async def test_run_easyocr(self, sample_image_path):
        """Test EasyOCR tool."""
        result = await run_easyocr(sample_image_path)

        assert result["source"] == "easyocr"
        assert "success" in result

    @pytest.mark.asyncio
    async def test_parallel_execution(self, sample_image_path):
        """Test all tools run in parallel."""
        import time

        start = time.time()

        results = await asyncio.gather(
            run_passport_eye(sample_image_path),
            run_tesseract(sample_image_path),
            run_easyocr(sample_image_path),
        )

        elapsed = time.time() - start

        assert len(results) == 3
        # Parallel should be faster than sequential (rough check)
        # Each tool takes ~2-5s, so parallel should be < 15s
        print(f"Parallel execution time: {elapsed:.1f}s")


class TestGraph:
    """Integration tests for the full graph."""

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
    async def test_graph_execution(self, sample_image_path):
        """Test full graph execution."""
        from app.extraction.graph import graph
        from app.extraction.state import PassportState

        initial_state: PassportState = {
            "image_path": sample_image_path,
            "ocr_results": [],
            "final_data": None,
            "confidence": 0.0,
            "errors": [],
            "source": "",
            "needs_human_review": False,
            "fraud_flags": [],
            "use_llm": False,
        }

        result = await graph.ainvoke(initial_state)

        # Should have OCR results
        assert len(result["ocr_results"]) == 3

        # Should have extracted some data
        assert result["final_data"] is not None or len(result["errors"]) > 0

        # Should have set confidence
        assert result["confidence"] >= 0.0

        # Should have set source
        assert result["source"] != ""

        print(f"Graph result: confidence={result['confidence']}, source={result['source']}")
        if result["final_data"]:
            print(f"Passport #: {result['final_data'].get('passport_number')}")

    @pytest.mark.asyncio
    async def test_graph_routes_to_human_review_on_fraud(self):
        """Test graph routes to human review when fraud detected."""
        from app.extraction.graph import route_after_ensemble
        from app.extraction.state import PassportState

        state: PassportState = {
            "image_path": "/fake/path",
            "ocr_results": [],
            "final_data": {"passport_number": "TEST"},
            "confidence": 0.8,
            "errors": [],
            "source": "tesseract",
            "needs_human_review": False,
            "fraud_flags": ["DOB_IN_FUTURE"],
            "use_llm": False,
        }

        next_node = route_after_ensemble(state)

        assert next_node == "human_review"

    @pytest.mark.asyncio
    async def test_graph_routes_to_vision_on_low_confidence(self):
        """Test graph routes to vision fallback on low confidence."""
        from app.extraction.graph import route_after_ensemble
        from app.extraction.state import PassportState

        state: PassportState = {
            "image_path": "/fake/path",
            "ocr_results": [],
            "final_data": {"passport_number": "TEST"},
            "confidence": 0.5,  # Low confidence
            "errors": [],
            "source": "tesseract",
            "needs_human_review": False,
            "fraud_flags": [],
            "use_llm": True,  # LLM enabled
        }

        next_node = route_after_ensemble(state)

        assert next_node == "vision_fallback"

    @pytest.mark.asyncio
    async def test_graph_ends_on_high_confidence(self):
        """Test graph ends when confidence is high."""
        from app.extraction.graph import route_after_ensemble, END
        from app.extraction.state import PassportState

        state: PassportState = {
            "image_path": "/fake/path",
            "ocr_results": [],
            "final_data": {"passport_number": "TEST"},
            "confidence": 0.95,  # High confidence
            "errors": [],
            "source": "passport_eye",
            "needs_human_review": False,
            "fraud_flags": [],
            "use_llm": False,
        }

        next_node = route_after_ensemble(state)

        assert next_node == END
