# OCR Research Directory

This directory contains experimental OCR implementations and evaluation tools.

## Structure

- `ocr_preprocessing.py` - Image preprocessing methods (otsu, adaptive, morphology, etc.)
- `ocr_tesseract.py` - Tesseract OCR implementations with different PSM modes
- `ocr_easyocr.py` - EasyOCR implementations with different preprocessing
- `ocr_metrics.py` - Metrics tracking for OCR performance evaluation
- `ocr_eval.py` - Evaluation script to test all configurations and find the best
- `ocr_production.py` - Production OCR using the best evaluated configuration

## Usage

### 1. Run Evaluation

Evaluate all OCR configurations to find the best one:

```python
from app.extraction.research.ocr_eval import find_best_ocr_config
from pathlib import Path

# Your test passport image
image_path = Path("docs/local/attested passport.pdf")

# Expected data for validation
expected = {
    "surname": "Nigam",
    "given_names": "Nikhil Rajesh",
    "passport_number": "N7178292",
    "date_of_birth": date(1996, 10, 25),
    "expiry_date": date(2026, 2, 15),
    # ... other fields
}

# Find best config
best_config = find_best_ocr_config(image_path, expected)
print(f"Best OCR method: {best_config}")
```

### 2. Update Production Config

After evaluation, update the best config in `ocr_production.py`:

```python
from app.extraction.research.ocr_production import update_best_config

# Update with best config from evaluation
update_best_config("tesseract", {"psm_mode": 6, "preprocess": "otsu"})
```

Or manually edit `BEST_OCR_CONFIG` in `ocr_production.py`.

### 3. Production Use

The production code in `app/extraction/passport.py` automatically uses the best config from `ocr_production.py`.

## Evaluation Results

The evaluation script tests:
- Tesseract with PSM modes: 6, 11, 13
- Tesseract with preprocessing: otsu, adaptive
- EasyOCR with preprocessing: none, otsu, adaptive

Results are tracked with metrics including:
- Success rate
- Extraction time
- Checksum validation
- Field accuracy

