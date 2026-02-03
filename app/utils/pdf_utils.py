"""
PDF processing utilities.

Handles:
- PDF to image conversion (for OCR)
- PDF text extraction (for fillable forms like G-28)
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: Path,
    dpi: int = 300,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Convert PDF pages to images for OCR processing.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (higher = better OCR, slower)
        output_dir: Directory to save images (uses temp dir if None)

    Returns:
        List of paths to generated images
    """
    try:
        from pdf2image import convert_from_path

        # Use temp directory if no output specified
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="pdf_images_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=dpi)

        image_paths = []
        for i, image in enumerate(images):
            image_path = output_dir / f"page_{i + 1}.png"
            image.save(str(image_path), "PNG")
            image_paths.append(image_path)
            logger.debug(f"Saved page {i + 1} to {image_path}")

        logger.info(f"Converted {len(image_paths)} pages from {pdf_path}")
        return image_paths

    except ImportError:
        logger.error("pdf2image not installed. Run: pip install pdf2image")
        return []
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        return []


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using pdfplumber.

    Best for fillable PDFs like G-28 where text is embedded.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content
    """
    try:
        import pdfplumber

        text_parts = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from {pdf_path}")
        return full_text

    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return ""


def extract_pdf_form_fields(pdf_path: Path) -> dict:
    """
    Extract form field data from fillable PDFs.

    Some PDFs have actual form fields (not just text) that can be read directly.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dictionary of field names to values
    """
    try:
        import pdfplumber

        fields = {}

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # Try to get form fields if available
                if hasattr(page, 'annots') and page.annots:
                    for annot in page.annots:
                        if annot.get('field_name'):
                            fields[annot['field_name']] = annot.get('field_value', '')

        logger.info(f"Extracted {len(fields)} form fields from {pdf_path}")
        return fields

    except Exception as e:
        logger.error(f"PDF form field extraction failed: {e}")
        return {}


def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Detect if a PDF is scanned (image-based) vs native text.

    Scanned PDFs need OCR; native PDFs can use text extraction.

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if PDF appears to be scanned/image-based
    """
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return True

            # Check first page
            page = pdf.pages[0]
            text = page.extract_text() or ""

            # If very little text relative to page size, likely scanned
            # Typical text page has thousands of chars
            if len(text.strip()) < 100:
                logger.info(f"{pdf_path} appears to be scanned (minimal text)")
                return True

            logger.info(f"{pdf_path} appears to have native text")
            return False

    except Exception as e:
        logger.warning(f"Could not determine PDF type: {e}")
        return True  # Assume scanned as fallback
