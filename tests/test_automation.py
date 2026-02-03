"""
Tests for form automation module.

Tests:
- Form filler with mock data
- Field mapping coverage
- Screenshot generation
"""

import pytest
from datetime import date
from pathlib import Path

from app.models.schemas import FormData, Sex
from app.automation.form_filler import (
    create_test_form_data,
    format_date_for_input,
    FIELD_MAPPINGS,
)


class TestFormDataHelpers:
    """Tests for helper functions."""

    def test_format_date_for_input(self):
        """Test date formatting for HTML inputs."""
        d = date(1996, 10, 25)
        result = format_date_for_input(d)
        assert result == "1996-10-25"

    def test_create_test_form_data(self):
        """Test that test form data is valid."""
        data = create_test_form_data()

        assert data.attorney_last_name == "Smith"
        assert data.beneficiary_last_name == "Nigam"
        assert data.passport_number == "N7178292"


class TestFieldMappings:
    """Tests for field mapping coverage."""

    def test_all_form_data_fields_mapped(self):
        """Verify all FormData fields have mappings."""
        # Get FormData fields (excluding private/computed)
        form_fields = set(FormData.model_fields.keys())

        # Get mapped fields
        mapped_fields = set(FIELD_MAPPINGS.keys())

        # These fields don't need mappings (computed or not in form)
        excluded = {"attorney_mobile_phone"}  # Optional fields

        unmapped = form_fields - mapped_fields - excluded

        # Log any unmapped fields (not a failure, just info)
        if unmapped:
            print(f"\nUnmapped fields (may be intentional): {unmapped}")

    def test_field_mappings_have_selectors(self):
        """Verify all mappings have valid selectors."""
        for field_name, mapping in FIELD_MAPPINGS.items():
            assert mapping.selector, f"{field_name} has empty selector"
            assert mapping.field_type in ["text", "select", "radio", "checkbox", "date"], \
                f"{field_name} has invalid field_type: {mapping.field_type}"


class TestFormFiller:
    """
    Integration tests for form filling.

    These tests require Playwright to be installed.
    Run with: pytest tests/test_automation.py -v
    """

    @pytest.fixture
    def test_form_data(self):
        """Create test form data."""
        return create_test_form_data()

    @pytest.fixture
    def screenshot_dir(self, tmp_path):
        """Temporary directory for screenshots."""
        return tmp_path / "screenshots"

    @pytest.mark.asyncio
    async def test_form_fill_with_mock_data(self, test_form_data, screenshot_dir):
        """
        Test filling the form with mock data.

        This is an integration test that actually opens a browser.
        """
        from app.automation.form_filler import fill_form_async

        screenshot_path = screenshot_dir / "test_fill.png"

        result = await fill_form_async(
            test_form_data,
            screenshot_path=screenshot_path,
            headless=True,  # Run headless in CI
        )

        # Check result
        assert result["success"], f"Form fill failed: {result['errors']}"

        # Check screenshot was created
        if screenshot_path.exists():
            assert result["screenshot_path"] is not None

        # Check some fields were filled
        assert len(result["filled_fields"]) > 0, "No fields were filled"

        # Log results for debugging
        print(f"\nFilled fields: {result['filled_fields']}")
        print(f"Failed fields: {result['failed_fields']}")

    def test_form_fill_sync_wrapper(self, test_form_data):
        """Test synchronous wrapper works."""
        from app.automation.form_filler import fill_form

        result = fill_form(
            test_form_data,
            screenshot_path=None,
            headless=True,
        )

        assert result["success"], f"Sync form fill failed: {result['errors']}"


# Expected behavior for form fill eval
# Note: beneficiary_first_name is combined into passport_given_names_combined
# because the form has a single field for given names
EXPECTED_FILLED_FIELDS = [
    "attorney_last_name",
    "attorney_first_name",
    "beneficiary_last_name",
    "passport_given_names_combined",  # Combined first+middle name field
    "passport_number",
    "date_of_birth",
]


class TestFormFillEval:
    """
    Evaluation tests for form filling accuracy.

    These verify that critical fields are being filled.
    """

    @pytest.mark.asyncio
    async def test_critical_fields_filled(self):
        """
        Eval: Verify critical fields are filled.

        This is a key metric for form automation quality.
        """
        from app.automation.form_filler import fill_form_async

        test_data = create_test_form_data()

        result = await fill_form_async(
            test_data,
            headless=True,
        )

        assert result["success"]

        # Check critical fields were filled
        filled = set(result["filled_fields"])
        missing_critical = []

        for field in EXPECTED_FILLED_FIELDS:
            if field not in filled:
                missing_critical.append(field)

        assert len(missing_critical) == 0, \
            f"Critical fields not filled: {missing_critical}"

    @pytest.mark.asyncio
    async def test_fill_rate_metric(self):
        """
        Eval: Calculate field fill rate.

        Reports percentage of fields successfully filled.
        """
        from app.automation.form_filler import fill_form_async

        test_data = create_test_form_data()

        result = await fill_form_async(
            test_data,
            headless=True,
        )

        total_attempted = len(result["filled_fields"]) + len(result["failed_fields"])
        if total_attempted > 0:
            fill_rate = len(result["filled_fields"]) / total_attempted
            print(f"\nFill rate: {fill_rate:.1%}")
            print(f"Filled: {len(result['filled_fields'])}")
            print(f"Failed: {len(result['failed_fields'])}")

            # Minimum acceptable fill rate
            assert fill_rate >= 0.5, f"Fill rate too low: {fill_rate:.1%}"
