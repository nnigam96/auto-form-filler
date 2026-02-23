#!/usr/bin/env python3
"""
Inspect the target form and extract actual field structure.
This helps create robust, general selectors.
"""

import asyncio
import json
from playwright.async_api import async_playwright

FORM_URL = "https://mendrika-alma.github.io/form-submission/"


async def inspect_form():
    """Inspect the form and extract all field information."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()

        print(f"Navigating to {FORM_URL}...")
        await page.goto(FORM_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        print("\n" + "=" * 60)
        print("FORM INSPECTION RESULTS")
        print("=" * 60)

        # Find all form elements
        form_elements = {
            "inputs": [],
            "selects": [],
            "textareas": [],
            "radios": {},
            "checkboxes": {},
        }

        # Inspect all input fields
        inputs = await page.locator("input").all()
        print(f"\nFound {len(inputs)} input fields:")
        for i, inp in enumerate(inputs):
            input_type = await inp.get_attribute("type") or "text"
            name = await inp.get_attribute("name") or ""
            input_id = await inp.get_attribute("id") or ""
            placeholder = await inp.get_attribute("placeholder") or ""
            label = ""
            
            # Try to find associated label
            if input_id:
                label_elem = page.locator(f"label[for='{input_id}']").first
                if await label_elem.count() > 0:
                    label = await label_elem.text_content() or ""
            
            # Also check for parent label
            if not label:
                parent_label = inp.locator("xpath=ancestor::label")
                if await parent_label.count() > 0:
                    label = await parent_label.text_content() or ""

            info = {
                "index": i,
                "type": input_type,
                "name": name,
                "id": input_id,
                "placeholder": placeholder,
                "label": label.strip(),
            }
            form_elements["inputs"].append(info)
            
            print(f"  [{i}] type={input_type}, name='{name}', id='{input_id}', placeholder='{placeholder}', label='{label}'")

        # Inspect all select fields
        selects = await page.locator("select").all()
        print(f"\nFound {len(selects)} select fields:")
        for i, sel in enumerate(selects):
            name = await sel.get_attribute("name") or ""
            sel_id = await sel.get_attribute("id") or ""
            label = ""
            
            if sel_id:
                label_elem = page.locator(f"label[for='{sel_id}']").first
                if await label_elem.count() > 0:
                    label = await label_elem.text_content() or ""
            
            # Get options
            options = await sel.locator("option").all()
            option_values = []
            for opt in options:
                value = await opt.get_attribute("value") or ""
                text = await opt.text_content() or ""
                option_values.append({"value": value, "text": text.strip()})

            info = {
                "index": i,
                "name": name,
                "id": sel_id,
                "label": label.strip(),
                "options": option_values,
            }
            form_elements["selects"].append(info)
            
            print(f"  [{i}] name='{name}', id='{sel_id}', label='{label}'")
            print(f"      Options: {len(option_values)}")

        # Inspect radio buttons (grouped by name)
        radios = await page.locator("input[type='radio']").all()
        print(f"\nFound {len(radios)} radio buttons:")
        radio_groups = {}
        for radio in radios:
            name = await radio.get_attribute("name") or ""
            value = await radio.get_attribute("value") or ""
            radio_id = await radio.get_attribute("id") or ""
            
            if name not in radio_groups:
                radio_groups[name] = []
            
            label = ""
            if radio_id:
                label_elem = page.locator(f"label[for='{radio_id}']").first
                if await label_elem.count() > 0:
                    label = await label_elem.text_content() or ""
            
            radio_groups[name].append({
                "value": value,
                "id": radio_id,
                "label": label.strip(),
            })
        
        for name, buttons in radio_groups.items():
            form_elements["radios"][name] = buttons
            print(f"  Group '{name}': {len(buttons)} options")
            for btn in buttons:
                print(f"    - value='{btn['value']}', label='{btn['label']}'")

        # Save to JSON for reference
        output_file = "form_structure.json"
        with open(output_file, "w") as f:
            json.dump(form_elements, f, indent=2)
        print(f"\n✓ Form structure saved to {output_file}")

        # Generate suggested selectors
        print("\n" + "=" * 60)
        print("SUGGESTED SELECTORS")
        print("=" * 60)
        generate_selectors(form_elements)

        # Take screenshot
        await page.screenshot(path="form_inspection.png", full_page=True)
        print(f"\n✓ Screenshot saved to form_inspection.png")

        await browser.close()


def generate_selectors(form_elements):
    """Generate robust selector suggestions based on form structure."""
    
    # Map our FormData fields to form fields
    field_mapping_suggestions = {
        "attorney_last_name": ["family_name", "last_name", "surname", "attorney_last"],
        "attorney_first_name": ["given_name", "first_name", "attorney_first"],
        "attorney_middle_name": ["middle_name"],
        "attorney_street_address": ["street_address", "address", "street"],
        "attorney_city": ["city"],
        "attorney_state": ["state"],
        "attorney_zip_code": ["zip_code", "zip", "postal_code"],
        "attorney_country": ["country"],
        "attorney_daytime_phone": ["daytime_phone", "phone", "telephone"],
        "attorney_email": ["email"],
        "licensing_authority": ["licensing_authority"],
        "bar_number": ["bar_number", "bar_no"],
        "law_firm_name": ["law_firm_name", "firm_name"],
        "beneficiary_last_name": ["beneficiary_last_name", "beneficiary_last", "client_last"],
        "beneficiary_first_name": ["beneficiary_first_name", "beneficiary_first", "client_first"],
        "beneficiary_middle_name": ["beneficiary_middle_name", "beneficiary_middle"],
        "passport_number": ["passport_number", "passport_no"],
        "country_of_issue": ["country_of_issue", "issuing_country"],
        "nationality": ["nationality"],
        "date_of_birth": ["date_of_birth", "dob", "birth_date"],
        "place_of_birth": ["place_of_birth", "birth_place"],
        "sex": ["sex", "gender"],
        "issue_date": ["issue_date", "issued_date"],
        "expiration_date": ["expiration_date", "expiry_date", "expires"],
    }

    print("\nSuggested field mappings:\n")
    for our_field, possible_names in field_mapping_suggestions.items():
        matches = []
        
        # Check inputs
        for inp in form_elements["inputs"]:
            name = inp["name"].lower() if inp["name"] else ""
            input_id = inp["id"].lower() if inp["id"] else ""
            placeholder = inp["placeholder"].lower() if inp["placeholder"] else ""
            label = inp["label"].lower() if inp["label"] else ""
            
            for possible in possible_names:
                if (possible.lower() in name or 
                    possible.lower() in input_id or 
                    possible.lower() in placeholder or
                    possible.lower() in label):
                    matches.append({
                        "type": inp["type"],
                        "name": inp["name"],
                        "id": inp["id"],
                        "selector": f"input[name='{inp['name']}']" if inp["name"] else f"#{inp['id']}" if inp["id"] else "N/A"
                    })
                    break
        
        # Check selects
        for sel in form_elements["selects"]:
            name = sel["name"].lower() if sel["name"] else ""
            sel_id = sel["id"].lower() if sel["id"] else ""
            
            for possible in possible_names:
                if possible.lower() in name or possible.lower() in sel_id:
                    matches.append({
                        "type": "select",
                        "name": sel["name"],
                        "id": sel["id"],
                        "selector": f"select[name='{sel['name']}']" if sel["name"] else f"#{sel['id']}" if sel["id"] else "N/A"
                    })
                    break
        
        if matches:
            print(f"{our_field}:")
            for match in matches:
                print(f"  - {match['type']}: {match['selector']} (name='{match['name']}', id='{match['id']}')")
        else:
            print(f"{our_field}: ⚠️  NO MATCH FOUND")


if __name__ == "__main__":
    asyncio.run(inspect_form())

