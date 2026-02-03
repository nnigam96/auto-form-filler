"""
Playwright-based form automation.

Fills the target immigration form with extracted data.
Target: https://mendrika-alma.github.io/form-submission/

Features:
- Robust field mapping with multiple selector strategies
- Screenshot capture for verification
- Does NOT submit the form (as per requirements)
"""

import logging
import asyncio
from datetime import date
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from app.models.schemas import FormData, Sex

logger = logging.getLogger(__name__)

# Target form URL
FORM_URL = "https://mendrika-alma.github.io/form-submission/"


@dataclass
class FieldMapping:
    """Maps form data fields to HTML selectors."""
    selector: str
    field_type: str = "text"  # text, select, radio, checkbox, date
    value_transform: Optional[callable] = None


# Field mappings for the target form
# Based on actual form inspection: https://mendrika-alma.github.io/form-submission/
# These are explicit mappings - auto-discovery will handle variations
FIELD_MAPPINGS: Dict[str, FieldMapping] = {
    # Part 1: Attorney Information
    "attorney_last_name": FieldMapping(
        selector="input[name='family-name'], #family-name",
        field_type="text"
    ),
    "attorney_first_name": FieldMapping(
        selector="input[name='given-name'], #given-name",
        field_type="text"
    ),
    "attorney_middle_name": FieldMapping(
        selector="input[name='middle-name'], #middle-name",
        field_type="text"
    ),
    "attorney_street_address": FieldMapping(
        selector="input[name='street-number'], #street-number",
        field_type="text"
    ),
    "attorney_city": FieldMapping(
        selector="input[name='city'], #city",
        field_type="text"
    ),
    "attorney_state": FieldMapping(
        selector="select[name='state'], #state",
        field_type="select"
    ),
    "attorney_zip_code": FieldMapping(
        selector="input[name='zip'], #zip",
        field_type="text"
    ),
    "attorney_country": FieldMapping(
        selector="input[name='country'], #country",
        field_type="text"
    ),
    "attorney_daytime_phone": FieldMapping(
        selector="input[name='daytime-phone'], #daytime-phone",
        field_type="text"
    ),
    "attorney_mobile_phone": FieldMapping(
        selector="input[name='mobile-phone'], #mobile-phone",
        field_type="text"
    ),
    "attorney_email": FieldMapping(
        selector="input[name='email'], #email",
        field_type="text"
    ),

    # Part 2: Eligibility
    "licensing_authority": FieldMapping(
        selector="input[name='licensing-authority'], #licensing-authority",
        field_type="text"
    ),
    "bar_number": FieldMapping(
        selector="input[name='bar-number'], #bar-number",
        field_type="text"
    ),
    "law_firm_name": FieldMapping(
        selector="input[name='law-firm'], #law-firm",
        field_type="text"
    ),

    # Part 3: Beneficiary/Passport Information
    "beneficiary_last_name": FieldMapping(
        selector="input[name='passport-surname'], #passport-surname",
        field_type="text"
    ),
    "beneficiary_first_name": FieldMapping(
        selector="input[name='passport-given-names'], #passport-given-names",
        field_type="text"
    ),
    # Special handling: passport-given-names field gets combined first+middle
    "passport_given_names_combined": FieldMapping(
        selector="input[name='passport-given-names'], #passport-given-names",
        field_type="text"
    ),
    # Note: passport-given-names is a single field - we'll combine first+middle
    "passport_number": FieldMapping(
        selector="input[name='passport-number'], #passport-number",
        field_type="text"
    ),
    "country_of_issue": FieldMapping(
        selector="input[name='passport-country'], #passport-country",
        field_type="text"
    ),
    "nationality": FieldMapping(
        selector="input[name='passport-nationality'], #passport-nationality",
        field_type="text"
    ),
    "date_of_birth": FieldMapping(
        selector="input[name='passport-dob'], #passport-dob",
        field_type="date"
    ),
    "place_of_birth": FieldMapping(
        selector="input[name='passport-pob'], #passport-pob",
        field_type="text"
    ),
    "sex": FieldMapping(
        selector="select[name='passport-sex'], #passport-sex",
        field_type="select"  # Changed from radio to select!
    ),
    "issue_date": FieldMapping(
        selector="input[name='passport-issue-date'], #passport-issue-date",
        field_type="date"
    ),
    "expiration_date": FieldMapping(
        selector="input[name='passport-expiry-date'], #passport-expiry-date",
        field_type="date"
    ),
}


def format_date_for_input(d: date) -> str:
    """Format date for HTML date input (YYYY-MM-DD)."""
    return d.strftime("%Y-%m-%d")


def format_sex_for_radio(sex: Sex) -> str:
    """Format sex enum for radio button value."""
    return sex.value  # M, F, or X


async def auto_discover_and_fill_fields(
    page, data_dict: Dict[str, Any], result: Dict[str, Any]
) -> list:
    """
    Auto-discover form fields by matching field names/IDs/labels.
    This makes the system more robust and general.
    
    Returns list of field names that were successfully filled.
    """
    filled = []
    
    # Field name variations for matching
    field_variations = {
        "attorney_last_name": ["family_name", "last_name", "surname", "attorney_last", "attorney_family"],
        "attorney_first_name": ["given_name", "first_name", "attorney_first", "attorney_given"],
        "attorney_middle_name": ["middle_name", "attorney_middle"],
        "attorney_street_address": ["street_address", "address", "street", "attorney_address"],
        "attorney_city": ["city", "attorney_city"],
        "attorney_state": ["state", "attorney_state"],
        "attorney_zip_code": ["zip_code", "zip", "postal_code", "attorney_zip"],
        "attorney_country": ["country", "attorney_country"],
        "attorney_daytime_phone": ["daytime_phone", "phone", "telephone", "attorney_phone"],
        "attorney_email": ["email", "attorney_email"],
        "licensing_authority": ["licensing_authority", "bar_authority"],
        "bar_number": ["bar_number", "bar_no", "bar_num"],
        "law_firm_name": ["law_firm_name", "firm_name", "firm"],
        "beneficiary_last_name": ["beneficiary_last_name", "beneficiary_last", "client_last", "last_name"],
        "beneficiary_first_name": ["beneficiary_first_name", "beneficiary_first", "client_first", "first_name", "passport-given-names"],
        "beneficiary_middle_name": ["beneficiary_middle_name", "beneficiary_middle", "middle_name"],
        "passport_given_names_combined": ["passport-given-names", "given-names", "passport_given_names"],
        "passport_number": ["passport_number", "passport_no", "passport"],
        "country_of_issue": ["country_of_issue", "issuing_country", "issue_country"],
        "nationality": ["nationality", "nationality_country"],
        "date_of_birth": ["date_of_birth", "dob", "birth_date", "birthdate"],
        "place_of_birth": ["place_of_birth", "birth_place", "birthplace"],
        "sex": ["sex", "gender"],
        "issue_date": ["issue_date", "issued_date", "passport_issue_date"],
        "expiration_date": ["expiration_date", "expiry_date", "expires", "expiration"],
    }
    
    # Get all form elements
    all_inputs = await page.locator("input, select, textarea").all()
    
    for field_name, value in data_dict.items():
        if value is None:
            continue
        
        # Skip if already in explicit mappings (those take priority)
        if field_name in FIELD_MAPPINGS:
            continue
        
        variations = field_variations.get(field_name, [])
        if not variations:
            continue
        
        # Try to find matching form field
        for element in all_inputs:
            try:
                tag = await element.evaluate("el => el.tagName.toLowerCase()")
                element_name = (await element.get_attribute("name") or "").lower()
                element_id = (await element.get_attribute("id") or "").lower()
                element_type = (await element.get_attribute("type") or "").lower()
                
                # Check if this element matches any variation
                matches = False
                for variation in variations:
                    if (variation.lower() in element_name or 
                        variation.lower() in element_id):
                        matches = True
                        break
                
                if not matches:
                    continue
                
                # Try to fill based on element type
                try:
                    if tag == "input" and element_type in ["text", "email", "tel", ""]:
                        if isinstance(value, date):
                            date_str = format_date_for_input(value)
                            await element.fill(date_str)
                        else:
                            await element.fill(str(value))
                        filled.append(field_name)
                        result["filled_fields"].append(field_name)
                        logger.debug(f"Auto-filled {field_name} via {element_name or element_id}")
                        break
                    
                    elif tag == "input" and element_type == "date":
                        if isinstance(value, date):
                            date_str = format_date_for_input(value)
                            await element.fill(date_str)
                            filled.append(field_name)
                            result["filled_fields"].append(field_name)
                            logger.debug(f"Auto-filled {field_name} (date) via {element_name or element_id}")
                            break
                    
                    elif tag == "select":
                        await element.select_option(str(value))
                        filled.append(field_name)
                        result["filled_fields"].append(field_name)
                        logger.debug(f"Auto-filled {field_name} (select) via {element_name or element_id}")
                        break
                    
                    elif tag == "input" and element_type == "radio":
                        if isinstance(value, Sex):
                            # Find radio with matching value
                            radio_value = value.value
                            radio = page.locator(f"input[type='radio'][name='{element_name}'][value='{radio_value}']").first
                            if await radio.count() > 0:
                                await radio.check()
                                filled.append(field_name)
                                result["filled_fields"].append(field_name)
                                logger.debug(f"Auto-filled {field_name} (radio) via {element_name}")
                                break
                
                except Exception as e:
                    logger.debug(f"Auto-fill attempt failed for {field_name}: {e}")
                    continue
                    
            except Exception:
                continue
    
    return filled


async def fill_form_async(
    form_data: FormData,
    screenshot_path: Optional[Path] = None,
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Fill the target form with extracted data using Playwright.

    Args:
        form_data: Combined data from passport and G-28
        screenshot_path: Path to save screenshot (optional)
        headless: Run browser in headless mode

    Returns:
        Dict with success status, screenshot path, and any errors
    """
    from playwright.async_api import async_playwright

    result = {
        "success": False,
        "screenshot_path": None,
        "filled_fields": [],
        "failed_fields": [],
        "errors": [],
    }

    async with async_playwright() as p:
        try:
            # Launch browser
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900}
            )
            page = await context.new_page()

            # Navigate to form
            logger.info(f"Navigating to {FORM_URL}")
            await page.goto(FORM_URL, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # Wait for JS to load
            
            # Wait for form to be ready
            try:
                await page.wait_for_selector("form, input, select", timeout=5000)
            except:
                logger.warning("Form elements not found immediately, continuing...")

            # Convert FormData to dict for iteration
            data_dict = form_data.model_dump()
            
            # Handle special case: passport-given-names combines first+middle
            # The form has a single field for given names, so combine them
            if "beneficiary_first_name" in data_dict and "beneficiary_middle_name" in data_dict:
                first = data_dict.get("beneficiary_first_name", "")
                middle = data_dict.get("beneficiary_middle_name", "")
                if middle:
                    data_dict["passport_given_names_combined"] = f"{first} {middle}".strip()
                else:
                    data_dict["passport_given_names_combined"] = first

            # First, try auto-discovery for fields not in FIELD_MAPPINGS
            auto_filled = await auto_discover_and_fill_fields(page, data_dict, result)
            logger.info(f"Auto-discovered and filled {len(auto_filled)} fields")

            # Then, try explicit mappings (these override auto-discovery)
            for field_name, mapping in FIELD_MAPPINGS.items():
                # Special handling for combined given names
                if field_name == "passport_given_names_combined":
                    value = data_dict.get("passport_given_names_combined")
                elif field_name == "beneficiary_first_name":
                    # Skip first name if we have combined version
                    if "passport_given_names_combined" in data_dict:
                        continue
                    value = data_dict.get(field_name)
                elif field_name == "beneficiary_middle_name":
                    # Skip middle name - it's combined into passport-given-names
                    continue
                else:
                    value = data_dict.get(field_name)

                if value is None:
                    continue

                # Skip if already filled by auto-discovery
                if field_name in auto_filled:
                    continue

                try:
                    # Handle different field types
                    if mapping.field_type == "text":
                        await fill_text_field(page, mapping.selector, str(value))
                        result["filled_fields"].append(field_name)

                    elif mapping.field_type == "select":
                        # Sex is a select dropdown, handle Sex enum
                        if field_name == "sex" and isinstance(value, Sex):
                            await fill_select_field(page, mapping.selector, value.value)
                        else:
                            await fill_select_field(page, mapping.selector, str(value))
                        result["filled_fields"].append(field_name)

                    elif mapping.field_type == "date":
                        if isinstance(value, date):
                            date_str = format_date_for_input(value)
                            await fill_text_field(page, mapping.selector, date_str)
                            result["filled_fields"].append(field_name)

                    elif mapping.field_type == "radio":
                        if isinstance(value, Sex):
                            await fill_radio_field(page, mapping.selector, value.value)
                            result["filled_fields"].append(field_name)

                except Exception as e:
                    logger.warning(f"Failed to fill {field_name}: {e}")
                    result["failed_fields"].append(field_name)

            # Take screenshot
            if screenshot_path:
                screenshot_path = Path(screenshot_path)
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(screenshot_path), full_page=True)
                result["screenshot_path"] = str(screenshot_path)
                logger.info(f"Screenshot saved to {screenshot_path}")

            # DO NOT SUBMIT - as per requirements
            logger.info("Form filled. NOT submitting as per requirements.")

            result["success"] = True
            await browser.close()

        except Exception as e:
            logger.error(f"Form filling failed: {e}")
            result["errors"].append(str(e))

    return result


async def fill_text_field(page, selector: str, value: str):
    """Fill a text input field, trying multiple selectors."""
    selectors = [s.strip() for s in selector.split(",")]

    for sel in selectors:
        try:
            element = page.locator(sel).first
            count = await element.count()
            if count > 0:
                # Wait for element to be visible and enabled
                await element.wait_for(state="visible", timeout=2000)
                await element.scroll_into_view_if_needed()
                await element.clear()
                await element.fill(value)
                # Verify value was set
                filled_value = await element.input_value()
                if filled_value != value:
                    # Try typing instead
                    await element.clear()
                    await element.type(value, delay=50)
                logger.debug(f"Filled {sel} with '{value}'")
                return
        except Exception as e:
            logger.debug(f"Selector {sel} failed: {e}")
            continue

    raise Exception(f"Could not find element for selectors: {selector}")


async def fill_select_field(page, selector: str, value: str):
    """Fill a select dropdown field."""
    selectors = [s.strip() for s in selector.split(",")]

    for sel in selectors:
        try:
            element = page.locator(sel).first
            count = await element.count()
            if count > 0:
                await element.wait_for(state="visible", timeout=2000)
                await element.scroll_into_view_if_needed()
                
                # Try exact match first
                try:
                    await element.select_option(value)
                    logger.debug(f"Selected {value} in {sel}")
                    return
                except:
                    # Try case-insensitive match
                    try:
                        await element.select_option(label=value, timeout=1000)
                        logger.debug(f"Selected {value} (case-insensitive) in {sel}")
                        return
                    except:
                        # Try partial match
                        options = await element.locator("option").all()
                        for option in options:
                            text = await option.text_content()
                            if value.lower() in text.lower() or text.lower() in value.lower():
                                option_value = await option.get_attribute("value")
                                await element.select_option(option_value)
                                logger.debug(f"Selected {value} (partial match) in {sel}")
                                return
        except Exception as e:
            logger.debug(f"Selector {sel} failed: {e}")
            continue

    raise Exception(f"Could not find select for selectors: {selector}")


async def fill_radio_field(page, selector: str, value: str):
    """Fill a radio button field."""
    try:
        # Try to find radio with matching value (exact)
        radio = page.locator(f"{selector}[value='{value}']").first
        count = await radio.count()
        if count > 0:
            await radio.scroll_into_view_if_needed()
            await radio.check()
            logger.debug(f"Checked radio {selector} with value {value}")
            return

        # Try case-insensitive value match
        radios = page.locator(selector).all()
        for r in radios:
            radio_value = await r.get_attribute("value")
            if radio_value and radio_value.upper() == value.upper():
                await r.scroll_into_view_if_needed()
                await r.check()
                logger.debug(f"Checked radio {selector} with value {radio_value} (case-insensitive)")
                return

        # Try clicking label if radio is hidden
        label = page.locator(f"label:has-text('{value}'), label[for*='{value.lower()}']").first
        count = await label.count()
        if count > 0:
            await label.scroll_into_view_if_needed()
            await label.click()
            logger.debug(f"Clicked label for radio {value}")
            return

    except Exception as e:
        logger.debug(f"Radio fill attempt failed: {e}")
    
    raise Exception(f"Could not fill radio {selector} with value {value}")


def fill_form(
    form_data: FormData,
    screenshot_path: Optional[Path] = None,
    headless: bool = True,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for fill_form_async.

    Use this from synchronous code (like FastAPI endpoints).
    """
    return asyncio.run(fill_form_async(form_data, screenshot_path, headless))


# For testing with mock data
def create_test_form_data() -> FormData:
    """Create test FormData for development/testing."""
    return FormData(
        # Attorney
        attorney_last_name="Smith",
        attorney_first_name="Barbara",
        attorney_street_address="545 Bryant Street",
        attorney_city="Palo Alto",
        attorney_state="CA",
        attorney_zip_code="94301",
        attorney_country="United States of America",
        attorney_email="immigration@tryalma.ai",
        licensing_authority="State Bar of California",
        bar_number="12083456",
        law_firm_name="Alma Legal Services PC",

        # Beneficiary
        beneficiary_last_name="Nigam",
        beneficiary_first_name="Nikhil",
        beneficiary_middle_name="Rajesh",
        passport_number="N7178292",
        country_of_issue="India",
        nationality="Indian",
        date_of_birth=date(1996, 10, 25),
        place_of_birth="Mumbai, Maharashtra",
        sex=Sex.MALE,
        issue_date=date(2016, 2, 16),
        expiration_date=date(2026, 2, 15),
    )


if __name__ == "__main__":
    # Test the form filler
    logging.basicConfig(level=logging.DEBUG)

    test_data = create_test_form_data()
    result = fill_form(
        test_data,
        screenshot_path=Path("test_screenshot.png"),
        headless=False  # Show browser for testing
    )

    print(f"\nResult: {result}")
