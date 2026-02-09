#!/usr/bin/env python3
"""
Simple script to test OpenAI API with passport image.
Shows the raw response from GPT-4o Vision.
"""

import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
project_root = Path(__file__).parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

# Initialize client
client = OpenAI(api_key=api_key)

# Path to passport
passport_path = project_root / "docs" / "local" / "attested passport.pdf"

if not passport_path.exists():
    print(f"ERROR: Passport file not found at {passport_path}")
    exit(1)

# Convert PDF to image (first page)
print(f"Loading passport from: {passport_path}")
try:
    from pdf2image import convert_from_path
    images = convert_from_path(str(passport_path))
    if not images:
        print("ERROR: Could not convert PDF to image")
        exit(1)
    
    # Save first page as temp image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        images[0].save(tmp.name, 'JPEG')
        image_path = Path(tmp.name)
except Exception as e:
    print(f"ERROR converting PDF: {e}")
    exit(1)

# Encode image
print("Encoding image...")
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Call OpenAI
print("Sending to OpenAI GPT-4o Vision...")
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": "You are an expert OCR system for passports. Extract the data precisely into JSON format."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the following fields from this passport image: surname, given_names, passport_number, nationality, date_of_birth, sex, expiry_date, country_of_issue, place_of_birth, issue_date. Return ONLY valid JSON. For dates, use YYYY-MM-DD format (e.g., '1996-10-25'). For sex, use M, F, or X."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    content = response.choices[0].message.content
    print("\n" + "=" * 60)
    print("OPENAI RESPONSE")
    print("=" * 60)
    print(content)
    print("=" * 60)
    
    # Parse and pretty print
    try:
        data = json.loads(content)
        print("\nParsed JSON:")
        print(json.dumps(data, indent=2))
    except:
        print("\nCould not parse as JSON")
    
    # Cleanup
    image_path.unlink()
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    if image_path.exists():
        image_path.unlink()

