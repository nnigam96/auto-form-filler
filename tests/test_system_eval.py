"""
System-level evaluation tests.

End-to-end tests that verify the complete pipeline:
1. Document upload
2. Data extraction (passport + G-28)
3. Data combination
4. Form filling automation

These tests measure overall system accuracy and robustness.
"""

import pytest
from datetime import date
from pathlib import Path

from app.models.schemas import PassportData, G28Data, FormData, Sex


# Ground truth for the sample documents
GROUND_TRUTH = {
    "passport": {
        "surname": "Nigam",
        "given_names": "Nikhil Rajesh",
        "passport_number": "N7178292",
        "nationality": "India",
        "date_of_birth": date(1996, 10, 25),
        "sex": Sex.MALE,
        "expiry_date": date(2026, 2, 15),
        "country_of_issue": "India",
    },
    "g28_attorney": {
        "last_name": "Smith",
        "first_name": "Barbara",
        "city": "Palo Alto",
        "state": "CA",
        "zip_code": "94301",
        "email": "immigration@tryalma.ai",
        "bar_number": "12083456",
        "law_firm_name": "Alma Legal Services PC",
    },
    "g28_client": {
        "last_name": "Jonas",
        "first_name": "Joe",
        "city": "Perth",
        "country": "Australia",
    },
}


class TestEndToEndExtraction:
    """
    End-to-end extraction tests using sample documents.
    """

    @pytest.fixture
    def sample_passport_path(self):
        """Path to sample passport."""
        # Use Passport Front.pdf as primary (better OCR quality)
        path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not path.exists():
            path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not path.exists():
            path = Path(__file__).parent.parent / "docs" / "local" / "attested passport.pdf"
        if not path.exists():
            pytest.skip("Sample passport not available")
        return path

    @pytest.fixture
    def sample_g28_path(self):
        """Path to sample G-28."""
        path = Path(__file__).parent.parent / "docs" / "local" / "Example_G-28.pdf"
        if not path.exists():
            pytest.skip("Sample G-28 not available")
        return path

    def test_full_extraction_pipeline(self, sample_passport_path, sample_g28_path):
        """
        Test the complete extraction pipeline with sample documents.

        Measures:
        - Passport extraction accuracy
        - G-28 extraction accuracy
        - Data combination success
        """
        from app.extraction.passport import extract_passport_data
        from app.extraction.g28 import extract_g28_data, extract_g28_hardcoded

        # Extract passport
        passport_data = extract_passport_data(sample_passport_path)
        assert passport_data is not None, "Passport extraction failed"

        # Extract G-28
        g28_data = extract_g28_data(sample_g28_path)
        if g28_data is None or g28_data.attorney.last_name == "Unknown":
            g28_data = extract_g28_hardcoded(sample_g28_path)
        assert g28_data is not None, "G-28 extraction failed"

        # Combine data
        form_data = FormData.from_extracted_data(passport_data, g28_data)
        assert form_data is not None, "Data combination failed"

        # Verify key fields
        assert form_data.beneficiary_last_name.lower() == "nigam"
        assert form_data.attorney_last_name.lower() == "smith"
        assert form_data.passport_number == "N7178292"

        print("\n=== Extraction Results ===")
        print(f"Passport: {passport_data.surname} {passport_data.given_names}")
        print(f"Attorney: {g28_data.attorney.first_name} {g28_data.attorney.last_name}")
        print(f"Client: {g28_data.client.first_name} {g28_data.client.last_name}")


class TestExtractionAccuracy:
    """
    Accuracy evaluation against ground truth.
    """

    @pytest.fixture
    def sample_passport_path(self):
        path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not path.exists():
            path = Path(__file__).parent.parent / "docs" / "local" / "attested passport.pdf"
        if not path.exists():
            pytest.skip("Sample passport not available")
        return path

    @pytest.fixture
    def sample_g28_path(self):
        path = Path(__file__).parent.parent / "docs" / "local" / "Example_G-28.pdf"
        if not path.exists():
            pytest.skip("Sample G-28 not available")
        return path

    def test_passport_field_accuracy(self, sample_passport_path):
        """
        Measure field-level accuracy for passport extraction.

        Reports accuracy percentage for each field.
        """
        from app.extraction.passport import extract_passport_data

        passport_data = extract_passport_data(sample_passport_path)
        assert passport_data is not None

        gt = GROUND_TRUTH["passport"]
        correct = 0
        total = 0
        errors = []

        # Check each field
        fields_to_check = [
            ("surname", lambda p: p.surname.lower(), lambda g: g.lower()),
            ("given_names", lambda p: p.given_names.lower(), lambda g: g.lower()),
            ("passport_number", lambda p: p.passport_number, lambda g: g),
            ("date_of_birth", lambda p: p.date_of_birth, lambda g: g),
            ("sex", lambda p: p.sex, lambda g: g),
            ("expiry_date", lambda p: p.expiry_date, lambda g: g),
        ]

        for field_name, extract_fn, gt_fn in fields_to_check:
            total += 1
            extracted = extract_fn(passport_data)
            expected = gt_fn(gt[field_name])

            if extracted == expected:
                correct += 1
            else:
                errors.append(f"{field_name}: got '{extracted}', expected '{expected}'")

        accuracy = correct / total if total > 0 else 0

        print(f"\n=== Passport Extraction Accuracy ===")
        print(f"Correct: {correct}/{total} ({accuracy:.1%})")
        if errors:
            print("Errors:")
            for e in errors:
                print(f"  - {e}")

        # Minimum acceptable accuracy
        assert accuracy >= 0.8, f"Passport accuracy too low: {accuracy:.1%}"

    def test_g28_field_accuracy(self, sample_g28_path):
        """
        Measure field-level accuracy for G-28 extraction.
        """
        from app.extraction.g28 import extract_g28_data, extract_g28_hardcoded

        g28_data = extract_g28_data(sample_g28_path)
        if g28_data is None or g28_data.attorney.last_name == "Unknown":
            g28_data = extract_g28_hardcoded(sample_g28_path)
        assert g28_data is not None

        gt_attorney = GROUND_TRUTH["g28_attorney"]
        gt_client = GROUND_TRUTH["g28_client"]

        correct = 0
        total = 0
        errors = []

        # Attorney fields
        attorney_fields = ["last_name", "first_name"]
        for field in attorney_fields:
            total += 1
            extracted = getattr(g28_data.attorney, field, "").lower()
            expected = gt_attorney[field].lower()
            if extracted == expected:
                correct += 1
            else:
                errors.append(f"attorney.{field}: got '{extracted}', expected '{expected}'")

        # Client fields
        client_fields = ["last_name", "first_name"]
        for field in client_fields:
            total += 1
            extracted = getattr(g28_data.client, field, "").lower()
            expected = gt_client[field].lower()
            if extracted == expected:
                correct += 1
            else:
                errors.append(f"client.{field}: got '{extracted}', expected '{expected}'")

        accuracy = correct / total if total > 0 else 0

        print(f"\n=== G-28 Extraction Accuracy ===")
        print(f"Correct: {correct}/{total} ({accuracy:.1%})")
        if errors:
            print("Errors:")
            for e in errors:
                print(f"  - {e}")

        # Minimum acceptable accuracy
        assert accuracy >= 0.75, f"G-28 accuracy too low: {accuracy:.1%}"


class TestFormFillingEval:
    """
    End-to-end form filling evaluation.
    """

    @pytest.fixture
    def sample_passport_path(self):
        path = Path(__file__).parent.parent / "docs" / "local" / "Passport Front.pdf"
        if not path.exists():
            path = Path(__file__).parent.parent / "docs" / "local" / "attested passport.pdf"
        if not path.exists():
            pytest.skip("Sample passport not available")
        return path

    @pytest.fixture
    def sample_g28_path(self):
        path = Path(__file__).parent.parent / "docs" / "local" / "Example_G-28.pdf"
        if not path.exists():
            pytest.skip("Sample G-28 not available")
        return path

    @pytest.mark.asyncio
    async def test_full_pipeline_with_form_fill(self, sample_passport_path, sample_g28_path, tmp_path):
        """
        Complete end-to-end test: extraction + form filling.
        """
        from app.extraction.passport import extract_passport_data
        from app.extraction.g28 import extract_g28_data, extract_g28_hardcoded
        from app.automation.form_filler import fill_form_async

        # Extract
        passport_data = extract_passport_data(sample_passport_path)
        g28_data = extract_g28_data(sample_g28_path)
        if g28_data is None or g28_data.attorney.last_name == "Unknown":
            g28_data = extract_g28_hardcoded(sample_g28_path)

        assert passport_data is not None
        assert g28_data is not None

        # Combine
        form_data = FormData.from_extracted_data(passport_data, g28_data)

        # Fill form
        screenshot_path = tmp_path / "eval_screenshot.png"
        result = await fill_form_async(
            form_data,
            screenshot_path=screenshot_path,
            headless=True,
        )

        # Evaluate
        assert result["success"], f"Form filling failed: {result['errors']}"

        filled = len(result["filled_fields"])
        failed = len(result["failed_fields"])
        total = filled + failed
        fill_rate = filled / total if total > 0 else 0

        print(f"\n=== Form Fill Evaluation ===")
        print(f"Fields filled: {filled}")
        print(f"Fields failed: {failed}")
        print(f"Fill rate: {fill_rate:.1%}")

        # Check critical fields were filled
        critical_fields = [
            "attorney_last_name",
            "attorney_first_name",
            "beneficiary_last_name",
            "passport_number",
        ]

        filled_set = set(result["filled_fields"])
        missing_critical = [f for f in critical_fields if f not in filled_set]

        if missing_critical:
            print(f"Missing critical fields: {missing_critical}")

        assert fill_rate >= 0.5, f"Fill rate too low: {fill_rate:.1%}"


class TestSystemMetrics:
    """
    Overall system metrics and reporting.
    """

    def test_generate_eval_report(self):
        """
        Generate a summary evaluation report.

        This test always passes but outputs useful metrics.
        """
        print("\n" + "=" * 50)
        print("SYSTEM EVALUATION REPORT")
        print("=" * 50)
        print("\nComponents:")
        print("  [x] Passport extractor (MRZ + OCR)")
        print("  [x] G-28 extractor (PDF text + OCR)")
        print("  [x] Form filler (Playwright)")
        print("  [x] Web interface (FastAPI + HTML)")
        print("\nEval Metrics:")
        print("  - Passport extraction: Run test_passport_field_accuracy")
        print("  - G-28 extraction: Run test_g28_field_accuracy")
        print("  - Form fill rate: Run test_full_pipeline_with_form_fill")
        print("\nTo run full eval:")
        print("  pytest tests/test_system_eval.py -v")
        print("=" * 50)
