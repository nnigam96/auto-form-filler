# Architecture

## Overview

This document describes the system architecture for the document automation pipeline.

## Components

### 1. Upload Interface
- Simple HTML/JS frontend
- Accepts PDF and image files
- Validates file types before upload

### 2. Extraction Pipeline
- **Traditional OCR** (default): Tesseract + PassportEye
- **LLM-enhanced** (optional): GPT-4o or Claude 3.5 Sonnet

### 3. Form Automation
- Playwright-based browser automation
- Field mapping from extracted data to form inputs

## Data Flow

```
Upload → Validate → Extract → Transform → Automate
```

## Design Decisions

See inline comments in source code for implementation details.
