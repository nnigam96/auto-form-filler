# Auto Form Filler

Intelligent document automation system that extracts data from immigration documents (passports, G-28 forms) and automatically fills web forms using browser automation.

## Problem Statement

Immigration law firms process thousands of documents annually. Manual data entry from passports and legal forms into web applications is:

- **Time-consuming**: Each form takes 5-10 minutes of manual entry
- **Error-prone**: Typos and transcription errors cause costly delays
- **Repetitive**: The same data is entered across multiple systems

This system automates the extraction-to-form-filling pipeline, reducing manual effort while maintaining accuracy.

## Features

- **LangGraph Agentic Pipeline**: Orchestrated extraction with parallel OCR, cross-validation, and HITL
- **Multi-format Document Support**: PDF and image uploads (JPEG, PNG)
- **Passport MRZ Extraction**: Machine Readable Zone parsing with checksum validation
- **Fraud Detection**: Cross-validates MRZ (machine-readable) vs Visual (printed) text
- **Human-In-The-Loop (HITL)**: Interactive review for conflicting field values
- **G-28 Form Extraction**: PDF form field extraction for fillable forms
- **Automated Form Filling**: Playwright-based browser automation
- **LLM Vision**: Local Ollama (llama3.2-vision) for visual text extraction

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────────┐
│  Web UI         │────▶│  FastAPI Backend                             │
│  (HTML/JS)      │     │                                              │
│  + HITL Modal   │◀────│  /extract/passport    → LangGraph Pipeline   │
└─────────────────┘     │  /extract/passport/confirm → HITL Confirm    │
                        │  /upload/g28          → G-28 Extraction      │
                        │  /fill-form           → Full Automation      │
                        └──────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
        ┌───────────────────────────┐                   ┌───────────────────┐
        │  LangGraph Pipeline       │                   │  G-28 Extraction  │
        │  (Passport Extraction)    │                   │  (PDF Form Fields)│
        └───────────┬───────────────┘                   └───────────────────┘
                    │
    ┌───────────────┼───────────────┬───────────────────┐
    ▼               ▼               ▼                   ▼
┌────────┐    ┌──────────┐    ┌──────────┐      ┌─────────────┐
│Passport│    │Tesseract │    │ EasyOCR  │      │ LLM Vision  │
│  Eye   │    │  OCR     │    │          │      │ (Ollama)    │
└────────┘    └──────────┘    └──────────┘      └─────────────┘
    │               │               │                   │
    └───────────────┴───────────────┘                   │
                    │                                   │
                    ▼                                   ▼
            ┌───────────────┐                   ┌───────────────┐
            │  MRZ Data     │                   │  Visual Data  │
            │  (checksummed)│                   │  (printed)    │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
                    └─────────────┬─────────────────────┘
                                  ▼
                        ┌─────────────────┐
                        │ Cross-Validate  │
                        │ MRZ vs Visual   │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌───────────────┐         ┌───────────────┐
            │   Aligned     │         │   Mismatch    │
            │ High Confidence│        │ HITL Required │
            └───────────────┘         └───────────────┘
```

## Agentic Pipeline Design

The extraction system uses **LangGraph** for workflow orchestration with several agentic patterns:

### 1. Parallel Tool Execution
Three OCR engines run concurrently for speed and redundancy:
- **PassportEye**: Specialized MRZ library with checksum validation
- **Tesseract**: General-purpose OCR with preprocessing
- **EasyOCR**: Deep learning-based text recognition

### 2. Field-Level Aggregation
Instead of picking one "winner" result, the system aggregates the best value per field:
- MRZ-encoded fields (passport number, dates) trust checksummed sources
- Name fields use majority voting across engines
- Low-confidence fields are flagged for review

### 3. Cross-Validation (Fraud Detection)
The key insight: **MRZ data is cryptographically verified, Visual data is what humans see.**
- LLM Vision extracts printed text (surname, dates, etc.)
- System compares MRZ vs Visual for each field
- Mismatches trigger fraud flags and HITL review

### 4. Human-In-The-Loop (HITL)
When MRZ and Visual data conflict:
- UI displays both values side-by-side
- User selects correct value for each field
- System continues with human-verified data

### 5. Graceful Degradation
- If LLM refuses (safety guardrails) → Fall back to MRZ-only
- If OCR fails → Try alternative engines
- If checksum invalid → Lower confidence, flag for review

## System Considerations

### Why MRZ is Ground Truth
- MRZ contains check digits that validate data integrity
- A valid checksum means the data hasn't been corrupted
- Visual text can be altered; MRZ tampering breaks checksums

### LLM Hallucination Handling
Vision models sometimes fabricate data. Mitigations:
- Never trust LLM over checksummed MRZ
- Detect refusal patterns ("I cannot assist with...")
- Use LLM only for cross-validation, not as primary source

### Confidence Scoring
- `0.99`: MRZ and Visual aligned
- `0.95`: MRZ only, valid checksum
- `0.80`: Visual only (fields not in MRZ)
- `0.00`: Conflict requiring HITL

## Quick Start

### Prerequisites

```bash
# macOS
brew install tesseract poppler ollama

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
# Install Ollama from https://ollama.ai

# Pull vision model
ollama pull llama3.2-vision
ollama serve  # Start in background
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

# Run the server
uvicorn app.main:app --reload --port 8000
```

Access the application at `http://localhost:8000`

## Usage

1. Open the web interface
2. Upload a passport (PDF or image)
3. Optionally upload a G-28 form (PDF)
4. Toggle "Use LLM Vision" for fraud detection
5. Click **Extract Data Only**
6. If conflicts detected → Review in HITL modal → Confirm selections
7. Click **Extract & Fill Form** to populate the target form

## Project Structure

```
auto-form-filler/
├── app/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── config.py               # Environment configuration
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── extraction/
│   │   ├── pipeline.py         # LangGraph extraction pipeline
│   │   ├── state.py            # Pipeline state definition
│   │   ├── ocr_engines.py      # PassportEye, Tesseract, EasyOCR
│   │   ├── voting.py           # Consensus & validation logic
│   │   ├── aggregator.py       # Field-level aggregation
│   │   ├── fraud_detector.py   # LLM visual extraction
│   │   ├── llm_vision.py       # Ollama vision interface
│   │   ├── passport.py         # Legacy extraction
│   │   ├── g28.py              # G-28 form extraction
│   │   └── reflection_agent.py # LLM reflection for errors
│   ├── automation/
│   │   └── form_filler.py      # Playwright form automation
│   └── utils/
│       └── pdf_utils.py        # PDF to image conversion
├── research/                   # Benchmarking & experiments
│   ├── benchmark.py            # OCR method comparison
│   ├── run_benchmark.py        # Benchmark runner
│   └── ocr_*.py                # Individual OCR implementations
├── static/                     # Frontend (HTML, CSS, JS)
│   ├── index.html              # Main UI with HITL modal
│   ├── style.css
│   └── app.js                  # HITL flow logic
├── tests/
│   ├── test_agentic.py         # Pipeline & HITL tests
│   ├── test_extraction.py      # Extraction tests
│   └── ...
├── _archive/                   # Archived code (V1-V4 pipelines)
└── docs/
    └── ARCHITECTURE.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/extract/passport` | POST | Extract passport with HITL support |
| `/extract/passport/confirm` | POST | Confirm HITL selections |
| `/upload/g28` | POST | Extract G-28 form data |
| `/extract` | POST | Combined passport + G-28 extraction |
| `/fill-form` | POST | Extract and fill target form |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `llama3.2-vision` | Vision model for fraud detection |
| `USE_LLM_EXTRACTION` | `false` | Enable LLM-based extraction |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run pipeline tests
pytest tests/test_agentic.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Improvements

### Completed
- [x] Parallel OCR execution (3 engines)
- [x] Field-level aggregation (vs result-level voting)
- [x] LLM Vision cross-validation
- [x] HITL modal for conflict resolution
- [x] Fraud detection flags
- [x] LLM refusal handling

### Planned
- [ ] Persist HITL corrections for model fine-tuning
- [ ] Batch processing for multiple documents
- [ ] Face matching (photo vs database)
- [ ] Additional document types (I-94, I-20, DS-160)
- [ ] Integration with case management systems
- [ ] Cloud deployment with GPU for faster LLM inference
- [ ] Confidence calibration based on historical accuracy

### Research Ideas
- [ ] Train custom MRZ detection model on edge cases
- [ ] Ensemble LLM (multiple models for consensus)
- [ ] Active learning from HITL corrections

## License

MIT
