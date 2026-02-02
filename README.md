# Auto Form Filler

Intelligent document automation system that extracts data from immigration documents (passports, G-28 forms) and automatically populates web forms using browser automation.

## Problem Statement

Immigration law firms and HR departments process thousands of documents annually. Manual data entry from passports and legal forms (like USCIS G-28) into web applications is:

- **Time-consuming**: Each form takes 5-10 minutes of manual entry
- **Error-prone**: Typos and transcription errors lead to costly delays
- **Repetitive**: The same data is often entered across multiple systems

This system automates the extraction-to-form-filling pipeline, reducing manual effort while maintaining accuracy.

## Features

- **Multi-format Support**: Handles PDF and image uploads (JPEG, PNG)
- **Passport Extraction**: MRZ (Machine Readable Zone) parsing + OCR for visual zone data
- **G-28 Form Extraction**: Structured extraction of attorney and client information
- **Intelligent Form Filling**: Playwright-based automation that maps extracted fields to target forms
- **Dual Extraction Modes**:
  - **Traditional OCR** (default): Works offline, no API keys required
  - **LLM-enhanced** (optional): Higher accuracy with GPT-4o or Claude 3.5 Sonnet
- **Robust Error Handling**: Graceful degradation when fields are missing or unreadable

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Upload UI      │────▶│  FastAPI Backend │────▶│  Playwright     │
│  (HTML/JS)      │     │                  │     │  Automation     │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌───────────────┐         ┌───────────────┐
           │ OCR Pipeline  │         │ LLM Pipeline  │
           │ (Tesseract +  │         │ (Optional)    │
           │  PassportEye) │         │               │
           └───────────────┘         └───────────────┘
```

## Quick Start

### Local Setup (macOS)

```bash
# 1. Install system dependencies
brew install tesseract poppler

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Playwright browsers
playwright install chromium

# 5. Configure environment (optional - for LLM mode)
cp .env.example .env
# Edit .env if using LLM extraction

# 6. Run the application
uvicorn app.main:app --reload
```

### Docker Setup

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

## Usage

1. Open `http://localhost:8000` in your browser
2. Upload a passport image (JPEG/PNG) or PDF
3. Upload a G-28 form (PDF or image)
4. Review extracted data
5. Click "Fill Form" to automatically populate the target web form

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LLM_EXTRACTION` | `false` | Enable LLM-based extraction |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | - | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | - | Required if using Anthropic |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Supported Documents

### Passport
- Extracts: Full name, date of birth, nationality, passport number, expiry date
- Methods: MRZ parsing (primary), OCR (fallback), LLM (optional enhancement)

### G-28 Form (Notice of Entry of Appearance)
- Extracts: Attorney name, firm name, address, client name, A-number
- Methods: OCR with field detection, LLM (optional enhancement)

## Project Structure

```
auto-form-filler/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings management
│   ├── models/
│   │   └── schemas.py       # Pydantic data models
│   ├── extraction/
│   │   ├── passport.py      # Passport extraction logic
│   │   ├── g28.py           # G-28 extraction logic
│   │   └── llm_client.py    # LLM abstraction layer
│   ├── automation/
│   │   └── form_filler.py   # Playwright automation
│   └── utils/
│       ├── pdf_utils.py     # PDF processing
│       └── logging.py       # Structured logging
├── static/                  # Frontend assets
├── tests/                   # Test suite
├── docs/                    # Documentation
├── Dockerfile
└── docker-compose.yml
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with hot reload
uvicorn app.main:app --reload --log-level debug
```

## Technical Decisions

- **Tesseract over cloud OCR**: Privacy-first approach, works offline
- **PassportEye for MRZ**: Battle-tested library, handles various passport formats
- **Playwright over Selenium**: Faster, more reliable, better async support
- **Pydantic for validation**: Ensures extracted data is well-typed before automation
- **FastAPI**: Modern async framework with automatic OpenAPI documentation

## Limitations

- MRZ extraction accuracy depends on image quality
- Some older passport formats may require LLM fallback
- G-28 forms with handwritten entries may have lower OCR accuracy

## License

MIT
