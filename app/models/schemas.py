"""
Pydantic schemas for document data extraction.

These models define the structure of data extracted from:
- Passports (MRZ + visual zone)
- G-28 forms (attorney and client information)
- Combined form data for automation
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    OTHER = "X"


class PassportData(BaseModel):
    """Data extracted from a passport document."""

    # Core MRZ fields
    surname: str = Field(..., description="Family name / Last name")
    given_names: str = Field(..., description="Given names (first + middle)")
    passport_number: str = Field(..., description="Passport document number")
    nationality: str = Field(..., description="Nationality (country code or full name)")
    date_of_birth: date = Field(..., description="Date of birth")
    sex: Sex = Field(..., description="Sex (M/F/X)")
    expiry_date: date = Field(..., description="Passport expiration date")

    # Additional fields (from visual zone or OCR)
    country_of_issue: str = Field(..., description="Country that issued the passport")
    place_of_birth: Optional[str] = Field(None, description="City/region of birth")
    issue_date: Optional[date] = Field(None, description="Date passport was issued")

    # Extraction metadata
    extraction_method: str = Field(default="unknown", description="mrz, ocr, or llm")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator('passport_number')
    @classmethod
    def clean_passport_number(cls, v: str) -> str:
        """Remove spaces and standardize passport number."""
        return v.strip().upper().replace(" ", "")

    @field_validator('surname', 'given_names')
    @classmethod
    def clean_name(cls, v: str) -> str:
        """Clean and title-case names."""
        return v.strip().title()


class AttorneyData(BaseModel):
    """Attorney/Representative information from G-28 Part 1 & 2."""

    last_name: str = Field(..., description="Attorney family name")
    first_name: str = Field(..., description="Attorney given name")
    middle_name: Optional[str] = Field(None, description="Attorney middle name")

    # Address
    street_address: str = Field(..., description="Street number and name")
    apt_ste_flr: Optional[str] = Field(None, description="Apartment/Suite/Floor")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State (2-letter code)")
    zip_code: str = Field(..., description="ZIP code")
    country: str = Field(default="United States of America")

    # Contact
    daytime_phone: Optional[str] = Field(None, description="Daytime telephone")
    mobile_phone: Optional[str] = Field(None, description="Mobile telephone")
    email: Optional[str] = Field(None, description="Email address")
    fax_number: Optional[str] = Field(None, description="Fax number")

    # Eligibility
    licensing_authority: Optional[str] = Field(None, description="State bar")
    bar_number: Optional[str] = Field(None, description="Bar number")
    law_firm_name: Optional[str] = Field(None, description="Law firm or organization")

    @field_validator('state')
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Ensure state is uppercase 2-letter code."""
        cleaned = v.strip().upper()
        if len(cleaned) == 2:
            return cleaned
        # Common state name mappings
        state_map = {
            "CALIFORNIA": "CA",
            "NEW YORK": "NY",
            "TEXAS": "TX",
            # Add more as needed
        }
        return state_map.get(cleaned.upper(), cleaned[:2].upper())


class ClientData(BaseModel):
    """Client information from G-28 Part 3."""

    last_name: str = Field(..., description="Client family name")
    first_name: str = Field(..., description="Client given name")
    middle_name: Optional[str] = Field(None, description="Client middle name")

    # Address
    street_address: Optional[str] = Field(None, description="Street address")
    apt_ste_flr: Optional[str] = Field(None, description="Apartment/Suite/Floor")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State/Province")
    zip_code: Optional[str] = Field(None, description="ZIP/Postal code")
    country: Optional[str] = Field(None, description="Country")

    # Contact
    daytime_phone: Optional[str] = Field(None, description="Daytime telephone")
    mobile_phone: Optional[str] = Field(None, description="Mobile telephone")
    email: Optional[str] = Field(None, description="Email address")

    # Immigration
    a_number: Optional[str] = Field(None, description="Alien Registration Number")
    uscis_account_number: Optional[str] = Field(None, description="USCIS Online Account")


class G28Data(BaseModel):
    """Combined G-28 form data."""

    attorney: AttorneyData
    client: ClientData

    # Extraction metadata
    extraction_method: str = Field(default="unknown", description="pdf_text, ocr, or llm")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class FormData(BaseModel):
    """
    Combined data ready for form filling.
    Maps to the target form fields at:
    https://mendrika-alma.github.io/form-submission/
    """

    # Part 1: Attorney Information (from G-28)
    attorney_last_name: str
    attorney_first_name: str
    attorney_middle_name: Optional[str] = None
    attorney_street_address: str
    attorney_apt_ste_flr: Optional[str] = None
    attorney_city: str
    attorney_state: str
    attorney_zip_code: str
    attorney_country: str = "United States of America"
    attorney_daytime_phone: Optional[str] = None
    attorney_mobile_phone: Optional[str] = None
    attorney_email: Optional[str] = None

    # Part 2: Eligibility (from G-28)
    licensing_authority: Optional[str] = None
    bar_number: Optional[str] = None
    law_firm_name: Optional[str] = None

    # Part 3: Beneficiary/Passport Information
    beneficiary_last_name: str
    beneficiary_first_name: str
    beneficiary_middle_name: Optional[str] = None
    passport_number: str
    country_of_issue: str
    nationality: str
    date_of_birth: date
    place_of_birth: Optional[str] = None
    sex: Sex
    issue_date: Optional[date] = None
    expiration_date: date

    @classmethod
    def from_extracted_data(cls, passport: PassportData, g28: G28Data) -> "FormData":
        """Combine passport and G-28 data into form-ready format."""

        # Split given_names into first and middle
        name_parts = passport.given_names.split(" ", 1)
        first_name = name_parts[0]
        middle_name = name_parts[1] if len(name_parts) > 1 else None

        return cls(
            # Attorney info
            attorney_last_name=g28.attorney.last_name,
            attorney_first_name=g28.attorney.first_name,
            attorney_middle_name=g28.attorney.middle_name,
            attorney_street_address=g28.attorney.street_address,
            attorney_apt_ste_flr=g28.attorney.apt_ste_flr,
            attorney_city=g28.attorney.city,
            attorney_state=g28.attorney.state,
            attorney_zip_code=g28.attorney.zip_code,
            attorney_country=g28.attorney.country,
            attorney_daytime_phone=g28.attorney.daytime_phone,
            attorney_mobile_phone=g28.attorney.mobile_phone,
            attorney_email=g28.attorney.email,
            licensing_authority=g28.attorney.licensing_authority,
            bar_number=g28.attorney.bar_number,
            law_firm_name=g28.attorney.law_firm_name,

            # Beneficiary/Passport info
            beneficiary_last_name=passport.surname,
            beneficiary_first_name=first_name,
            beneficiary_middle_name=middle_name,
            passport_number=passport.passport_number,
            country_of_issue=passport.country_of_issue,
            nationality=passport.nationality,
            date_of_birth=passport.date_of_birth,
            place_of_birth=passport.place_of_birth,
            sex=passport.sex,
            issue_date=passport.issue_date,
            expiration_date=passport.expiry_date,
        )


class ExtractionResult(BaseModel):
    """Result of document extraction, including any errors."""

    success: bool
    passport_data: Optional[PassportData] = None
    g28_data: Optional[G28Data] = None
    form_data: Optional[FormData] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
