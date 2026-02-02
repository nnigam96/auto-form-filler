"""
Tests for Pydantic schemas.

Validates:
- Schema creation with valid data
- Field validation and cleaning
- Error handling for invalid data
- FormData combination logic
"""

import pytest
from datetime import date
from pydantic import ValidationError

from app.models.schemas import (
    PassportData,
    AttorneyData,
    ClientData,
    G28Data,
    FormData,
    Sex,
    ExtractionResult,
)


class TestPassportData:
    """Tests for PassportData schema."""

    def test_valid_passport_data(self):
        """Test creating PassportData with valid fields."""
        data = PassportData(
            surname="NIGAM",
            given_names="NIKHIL RAJESH",
            passport_number="N7178292",
            nationality="INDIAN",
            date_of_birth=date(1996, 10, 25),
            sex=Sex.MALE,
            expiry_date=date(2026, 2, 15),
            country_of_issue="INDIA",
            place_of_birth="MUMBAI, MAHARASHTRA",
            issue_date=date(2016, 2, 16),
            extraction_method="mrz",
            confidence_score=0.99,
        )

        assert data.surname == "Nigam"  # Title-cased
        assert data.given_names == "Nikhil Rajesh"  # Title-cased
        assert data.passport_number == "N7178292"  # Uppercase, no spaces

    def test_passport_number_cleaning(self):
        """Test that passport numbers are cleaned."""
        data = PassportData(
            surname="Test",
            given_names="User",
            passport_number="  n7178292  ",  # Spaces and lowercase
            nationality="INDIAN",
            date_of_birth=date(1996, 10, 25),
            sex=Sex.MALE,
            expiry_date=date(2026, 2, 15),
            country_of_issue="INDIA",
        )

        assert data.passport_number == "N7178292"

    def test_name_title_casing(self):
        """Test that names are title-cased."""
        data = PassportData(
            surname="NIGAM",
            given_names="nikhil rajesh",
            passport_number="N7178292",
            nationality="INDIAN",
            date_of_birth=date(1996, 10, 25),
            sex=Sex.MALE,
            expiry_date=date(2026, 2, 15),
            country_of_issue="INDIA",
        )

        assert data.surname == "Nigam"
        assert data.given_names == "Nikhil Rajesh"

    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PassportData(
                surname="Test",
                # Missing given_names
                passport_number="N7178292",
                nationality="INDIAN",
                date_of_birth=date(1996, 10, 25),
                sex=Sex.MALE,
                expiry_date=date(2026, 2, 15),
                country_of_issue="INDIA",
            )

        assert "given_names" in str(exc_info.value)

    def test_confidence_score_bounds(self):
        """Test confidence score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            PassportData(
                surname="Test",
                given_names="User",
                passport_number="N7178292",
                nationality="INDIAN",
                date_of_birth=date(1996, 10, 25),
                sex=Sex.MALE,
                expiry_date=date(2026, 2, 15),
                country_of_issue="INDIA",
                confidence_score=1.5,  # Invalid
            )


class TestAttorneyData:
    """Tests for AttorneyData schema."""

    def test_valid_attorney_data(self):
        """Test creating AttorneyData with valid fields."""
        data = AttorneyData(
            last_name="Smith",
            first_name="Barbara",
            street_address="545 Bryant Street",
            city="Palo Alto",
            state="CA",
            zip_code="94301",
            email="immigration@tryalma.ai",
            bar_number="12083456",
            licensing_authority="State Bar of California",
            law_firm_name="Alma Legal Services PC",
        )

        assert data.state == "CA"
        assert data.last_name == "Smith"

    def test_state_conversion(self):
        """Test that full state names are converted to codes."""
        data = AttorneyData(
            last_name="Smith",
            first_name="Barbara",
            street_address="545 Bryant Street",
            city="Palo Alto",
            state="California",  # Full name
            zip_code="94301",
        )

        assert data.state == "CA"

    def test_state_already_code(self):
        """Test that state codes pass through."""
        data = AttorneyData(
            last_name="Smith",
            first_name="Barbara",
            street_address="545 Bryant Street",
            city="Palo Alto",
            state="NY",
            zip_code="10001",
        )

        assert data.state == "NY"


class TestClientData:
    """Tests for ClientData schema."""

    def test_valid_client_data(self):
        """Test creating ClientData with valid fields."""
        data = ClientData(
            last_name="Jonas",
            first_name="Joe",
            street_address="16 Anytown Street",
            city="Perth",
            state="WA",
            zip_code="6000",
            country="Australia",
            daytime_phone="+61 45453434",
            email="b.smith_00@test.ai",
        )

        assert data.last_name == "Jonas"
        assert data.country == "Australia"

    def test_minimal_client_data(self):
        """Test creating ClientData with only required fields."""
        data = ClientData(
            last_name="Jonas",
            first_name="Joe",
        )

        assert data.last_name == "Jonas"
        assert data.street_address is None


class TestG28Data:
    """Tests for G28Data schema."""

    def test_valid_g28_data(self):
        """Test creating G28Data with nested attorney and client."""
        attorney = AttorneyData(
            last_name="Smith",
            first_name="Barbara",
            street_address="545 Bryant Street",
            city="Palo Alto",
            state="CA",
            zip_code="94301",
        )

        client = ClientData(
            last_name="Jonas",
            first_name="Joe",
        )

        g28 = G28Data(
            attorney=attorney,
            client=client,
            extraction_method="pdf_text",
            confidence_score=0.95,
        )

        assert g28.attorney.last_name == "Smith"
        assert g28.client.last_name == "Jonas"


class TestFormData:
    """Tests for FormData schema and combination logic."""

    @pytest.fixture
    def sample_passport(self):
        """Sample passport data."""
        return PassportData(
            surname="Nigam",
            given_names="Nikhil Rajesh",
            passport_number="N7178292",
            nationality="Indian",
            date_of_birth=date(1996, 10, 25),
            sex=Sex.MALE,
            expiry_date=date(2026, 2, 15),
            country_of_issue="India",
            place_of_birth="Mumbai, Maharashtra",
            issue_date=date(2016, 2, 16),
            extraction_method="mrz",
        )

    @pytest.fixture
    def sample_g28(self):
        """Sample G-28 data."""
        attorney = AttorneyData(
            last_name="Smith",
            first_name="Barbara",
            street_address="545 Bryant Street",
            city="Palo Alto",
            state="CA",
            zip_code="94301",
            email="immigration@tryalma.ai",
            bar_number="12083456",
            licensing_authority="State Bar of California",
            law_firm_name="Alma Legal Services PC",
        )

        client = ClientData(
            last_name="Jonas",
            first_name="Joe",
            street_address="16 Anytown Street",
            city="Perth",
            state="WA",
            zip_code="6000",
            country="Australia",
        )

        return G28Data(
            attorney=attorney,
            client=client,
            extraction_method="pdf_text",
        )

    def test_form_data_from_extracted(self, sample_passport, sample_g28):
        """Test combining passport and G-28 data into FormData."""
        form_data = FormData.from_extracted_data(sample_passport, sample_g28)

        # Attorney fields
        assert form_data.attorney_last_name == "Smith"
        assert form_data.attorney_first_name == "Barbara"
        assert form_data.attorney_city == "Palo Alto"
        assert form_data.attorney_state == "CA"
        assert form_data.bar_number == "12083456"

        # Beneficiary/Passport fields
        assert form_data.beneficiary_last_name == "Nigam"
        assert form_data.beneficiary_first_name == "Nikhil"
        assert form_data.beneficiary_middle_name == "Rajesh"
        assert form_data.passport_number == "N7178292"
        assert form_data.nationality == "Indian"
        assert form_data.date_of_birth == date(1996, 10, 25)

    def test_form_data_single_name(self, sample_g28):
        """Test FormData when passport has single given name."""
        passport = PassportData(
            surname="Doe",
            given_names="John",  # No middle name
            passport_number="X1234567",
            nationality="American",
            date_of_birth=date(1990, 1, 1),
            sex=Sex.MALE,
            expiry_date=date(2030, 1, 1),
            country_of_issue="USA",
        )

        form_data = FormData.from_extracted_data(passport, sample_g28)

        assert form_data.beneficiary_first_name == "John"
        assert form_data.beneficiary_middle_name is None


class TestExtractionResult:
    """Tests for ExtractionResult schema."""

    def test_successful_result(self, ):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            success=True,
            errors=[],
            warnings=["Low confidence on date field"],
        )

        assert result.success is True
        assert len(result.warnings) == 1

    def test_failed_result(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            success=False,
            errors=["Could not detect MRZ", "PDF is corrupted"],
        )

        assert result.success is False
        assert len(result.errors) == 2


# Expected values for the sample documents (ground truth for eval)
EXPECTED_PASSPORT_DATA = {
    "surname": "Nigam",
    "given_names": "Nikhil Rajesh",
    "passport_number": "N7178292",
    "nationality": "Indian",
    "date_of_birth": date(1996, 10, 25),
    "sex": Sex.MALE,
    "expiry_date": date(2026, 2, 15),
    "country_of_issue": "India",
    "place_of_birth": "Mumbai, Maharashtra",
    "issue_date": date(2016, 2, 16),
}

EXPECTED_G28_ATTORNEY = {
    "last_name": "Smith",
    "first_name": "Barbara",
    "street_address": "545 Bryant Street",
    "city": "Palo Alto",
    "state": "CA",
    "zip_code": "94301",
    "email": "immigration@tryalma.ai",
    "bar_number": "12083456",
    "licensing_authority": "State Bar of California",
    "law_firm_name": "Alma Legal Services PC",
}

EXPECTED_G28_CLIENT = {
    "last_name": "Jonas",
    "first_name": "Joe",
    "street_address": "16 Anytown Street",
    "city": "Perth",
    "country": "Australia",
}
