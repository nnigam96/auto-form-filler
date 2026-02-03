# Auto Form Filler

Intelligent document automation system that extracts data from immigration documents (passports, G-28 forms) and automatically fills web forms using browser automation.

## Problem Statement

Immigration law firms process thousands of documents annually. Manual data entry from passports and legal forms into web applications is:

- **Time-consuming**: Each form takes 5-10 minutes of manual entry
- **Error-prone**: Typos and transcription errors cause costly delays
- **Repetitive**: The same data is entered across multiple systems

This system automates the extraction-to-form-filling pipeline, reducing manual effort while maintaining accuracy.

## Features

- **Multi-format Document Support**: PDF and image uploads (JPEG, PNG)
- **Passport MRZ Extraction**: Machine Readable Zone parsing with OCR fallback
- **G-28 Form Extraction**: PDF form field extraction for fillable forms
- **Automated Form Filling**: Playwright-based browser automation
- **Rotation Handling**: Automatic detection and correction of rotated scans
- **Dual Extraction Modes**:
  - **Traditional OCR** (default): Works offline, no API keys required
  - **LLM Vision** (optional): GPT-4o fallback for challenging documents

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Web UI         │────▶│  FastAPI Backend │────▶│  Playwright     │
│  (HTML/JS)      │     │                  │     │  Form Filler    │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌───────────────┐         ┌───────────────┐
           │ Passport      │         │ G-28          │
           │ Extraction    │         │ Extraction    │
           └───────┬───────┘         └───────┬───────┘
                   │                         │
        ┌──────────┼──────────┐              │
        ▼          ▼          ▼              ▼
   PassportEye   OCR       LLM         PDF Form
   (MRZ)       Service   Vision        Fields
```

## Extraction Pipeline

### Passport Extraction Strategy

The system uses a tiered approach for maximum reliability:

1. **PassportEye** (Primary): Specialized MRZ detection library with checksum validation
2. **OCR Service** (Fallback): Custom pipeline with:
   - Multi-rotation support (0°, 90°, 180°, 270°)
   - Multiple preprocessing methods (grayscale, Otsu, adaptive threshold)
   - Tesseract + EasyOCR engines
   - MRZ line detection and checksum validation
3. **LLM Vision** (Optional): GPT-4o for challenging documents

### G-28 Form Extraction

- **Fillable PDFs**: Direct form field extraction via pypdf
- **Scanned PDFs**: OCR-based text extraction with field mapping

## Quick Start

### Prerequisites

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

### Installation

```bash
# Clone and setup
git clone https://github.com/nnigam96/auto-form-filler.git
cd auto-form-filler

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium

# Optional: Configure LLM extraction
cp .env.example .env
# Edit .env with your API keys

# Run the server
python -m app.main
```

Access the application at `http://localhost:8000`

## Usage

1. Open the web interface
2. Upload a passport (PDF or image)
3. Upload a G-28 form (PDF)
4. Click **Extract Data Only** to preview extracted data
5. Click **Extract & Fill Form** to automatically populate the target form
6. Review the screenshot of the filled form

## Project Structure

```
auto-form-filler/
├── app/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── config.py               # Environment configuration
│   ├── models/
│   │   └── schemas.py          # Pydantic models (PassportData, G28Data, FormData)
│   ├── extraction/
│   │   ├── passport.py         # Passport extraction pipeline
│   │   ├── g28.py              # G-28 form extraction
│   │   ├── ocr_service.py      # Multi-engine OCR with rotation handling
│   │   └── llm_vision.py       # GPT-4o vision fallback
│   ├── automation/
│   │   └── form_filler.py      # Playwright form automation
│   └── utils/
│       └── pdf_utils.py        # PDF to image conversion
├── static/                     # Frontend (HTML, CSS, JS)
├── tests/
│   ├── test_schemas.py         # Data model validation tests
│   ├── test_extraction.py      # Extraction pipeline tests
│   ├── test_ocr_service.py     # OCR service unit tests
│   ├── test_automation.py      # Form filling tests
│   └── test_system_eval.py     # End-to-end evaluation
├── docs/
│   └── ARCHITECTURE.md         # Detailed architecture docs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LLM_EXTRACTION` | `false` | Enable LLM-based extraction |
| `OPENAI_API_KEY` | - | Required for LLM vision mode |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_ocr_service.py -v      # OCR unit tests
pytest tests/test_extraction.py -v        # Extraction tests
pytest tests/test_automation.py -v        # Form filling tests
pytest tests/test_system_eval.py -v       # End-to-end evaluation

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Tesseract + EasyOCR** | Privacy-first, works offline, no API costs |
| **PassportEye for MRZ** | Specialized library with checksum validation |
| **pypdf for G-28** | Direct form field extraction from fillable PDFs |
| **Multi-rotation OCR** | Handles rotated scans automatically |
| **Playwright** | Fast, reliable browser automation with async support |
| **Pydantic** | Runtime validation ensures data integrity |
| **FastAPI** | Modern async framework with automatic OpenAPI docs |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/extract` | POST | Extract data from documents |
| `/fill-form` | POST | Extract and fill target form |
| `/screenshots/{filename}` | GET | Retrieve form screenshots |

## Limitations

- MRZ extraction accuracy depends on scan quality
- Heavily rotated or skewed scans may require manual correction
- Handwritten entries on G-28 forms have lower OCR accuracy
- LLM mode requires API keys and incurs costs

## Future Improvements

- [ ] Additional document types (I-94, I-20, DS-160)
- [ ] Batch processing for multiple documents
- [ ] Confidence scoring with manual review workflow
- [ ] Integration with case management systems

## License

MIT
