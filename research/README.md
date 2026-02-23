# OCR Research Directory

This directory contains OCR experimentation and benchmarking tools.
Research results inform production configuration.

## Flow: Research → Production

```
┌─────────────────────────────────────────────────────────────┐
│                      RESEARCH                                │
│  Run benchmarks → Compare methods → Export best config      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                   best_config.json
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      PRODUCTION                              │
│  ocr_service.py reads best_config.json                      │
└─────────────────────────────────────────────────────────────┘
```

## Structure

```
research/
├── test_data/                # Test images (gitignored)
│   ├── .gitignore
│   └── ground_truth.json     # Expected values for each image
├── ocr_preprocessing.py      # Image preprocessing methods
├── ocr_tesseract.py          # Tesseract experiments
├── ocr_easyocr.py            # EasyOCR experiments
├── ocr_passporteye.py        # PassportEye experiments
├── ocr_ensemble.py           # Parallel voting (agentic)
├── ocr_llm_vision.py         # Ollama LLM Vision
├── benchmark.py              # Main benchmark system
├── run_benchmark.py          # CLI entry point
├── best_config.json          # OUTPUT: winning config (auto-generated)
└── README.md
```

## Quick Start

### 1. Add Test Images

Place test passport images in `test_data/`:
```bash
cp /path/to/passport.png test_data/passport_valid.png
```

### 2. Update Ground Truth

Edit `test_data/ground_truth.json`:
```json
{
  "images": {
    "passport_valid.png": {
      "expected": {
        "surname": "SMITH",
        "passport_number": "AB123456",
        "date_of_birth": "1990-05-15",
        ...
      },
      "expects_human_review": false
    }
  }
}
```

### 3. Run Benchmark

```bash
# Run all methods on all images
python -m app.extraction.research.run_benchmark

# Run specific method
python -m app.extraction.research.run_benchmark --method tesseract

# Enable memory profiling
python -m app.extraction.research.run_benchmark --memory

# Output to JSON/CSV
python -m app.extraction.research.run_benchmark --output all
```

### 4. Results

- Console output shows method comparison table
- `best_config.json` is auto-generated with winning config
- Production (`ocr_service.py`) automatically uses best config

## OCR Methods

| Method | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| Tesseract | Traditional OCR with PSM modes | Fast | Good |
| EasyOCR | Deep learning OCR | Medium | Good |
| PassportEye | Specialized MRZ detector | Fast | Excellent |
| Ensemble | Parallel + voting | Slow | Best |
| LLM Vision | Ollama (llava:7b) | Slowest | Fallback |

## Metrics Tracked

- **Success Rate**: MRZ found / total attempts
- **Checksum Pass Rate**: Valid MRZ checksums
- **Accuracy Score**: Exact + fuzzy match to ground truth
- **Latency**: Extraction time in ms
- **Memory** (optional): Peak memory usage in MB
- **Human Review Detection**: Fraud flag accuracy

## Adding New Methods

1. Create `ocr_newmethod.py` with:
   - `extract_with_newmethod(image_path) -> Dict`
   - `get_all_configs() -> List[Dict]`

2. Add to `benchmark.py`:
   - Import in `_get_all_method_configs()`
   - Add case in `run_single_method()`

## Adversarial Testing

For manipulated images that should trigger human review:

```json
{
  "passport_adversarial.png": {
    "expects_human_review": true,
    "expected_fraud_flags": ["MRZ_VISUAL_MISMATCH"]
  }
}
```

The benchmark tracks whether methods correctly flag these images.
