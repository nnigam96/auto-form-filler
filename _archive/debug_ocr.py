import logging
import time
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from app.extraction.research.ocr_easyocr import extract_mrz_with_easyocr, get_easyocr_reader
from app.extraction.research.ocr_preprocessing import PREPROCESSING_METHODS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_easyocr_tournament(pdf_path):
    """
    Run EasyOCR tournament on a multi-page PDF with High-Res conversion.
    """
    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"=== 🏆 HIGH-RES OCR TOURNAMENT: {path.name} ===\n")
    
    # 1. Convert PDF to Images at 300 DPI (Crucial for MRZ)
    print("Converting PDF to 300 DPI images...")
    try:
        images = convert_from_path(str(path), dpi=300)
        print(f"Converted {len(images)} pages.\n")
    except Exception as e:
        print(f"❌ PDF Conversion failed: {e}")
        return

    # Initialize reader once
    reader = get_easyocr_reader()
    if not reader:
        print("❌ Could not initialize EasyOCR")
        return

    # We test these 3 winning strategies
    # 'upscale' might be overkill at 300 DPI, so 'none' (grayscale) usually wins
    strategies = ["none", "adaptive", "upscale"]

    for page_num, pil_image in enumerate(images):
        print(f"--- 📄 Processing Page {page_num + 1} ---")
        
        # Save debug image
        debug_filename = f"debug_page_{page_num+1}.png"
        pil_image.save(debug_filename)
        
        for method in strategies:
            print(f"   [Strategy: {method}]")
            start = time.time()
            
            try:
                # 1. Preprocess
                if method == "none":
                    # Just convert PIL to OpenCV format
                    # EasyOCR expects BGR for colored images
                    img_np = np.array(pil_image)
                    image_input = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    func = PREPROCESSING_METHODS[method]
                    # We pass the saved debug file to reuse your existing functions
                    processed_pil = func(debug_filename)
                    img_np = np.array(processed_pil)
                    if len(img_np.shape) == 2: # Grayscale
                        image_input = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                    else:
                        image_input = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 2. Run EasyOCR
                results = reader.readtext(
                    image_input,
                    detail=0,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< ' 
                )
                
                # 3. Show retrieved lines (first 20 lines)
                print(f"      📝 Retrieved {len(results)} lines from OCR")
                print(f"      Showing first 20 lines:")
                for i, line in enumerate(results[:20], 1):
                    clean = line.replace(" ", "").upper()
                    # Mark potential MRZ lines
                    is_mrz_candidate = (
                        len(clean) > 20 and 
                        ("<<" in clean or "P<" in clean or clean.startswith(("P<", "I<", "A<")))
                    )
                    marker = "🔍 MRZ?" if is_mrz_candidate else "  "
                    print(f"      {marker} [{i:2d}] {line[:60]}{'...' if len(line) > 60 else ''}")
                    if is_mrz_candidate:
                        print(f"           Clean: {clean[:70]}{'...' if len(clean) > 70 else ''}")
                
                if len(results) > 20:
                    print(f"      ... ({len(results) - 20} more lines)")
                
                # 4. Scan for MRZ candidates
                mrz_candidates = []
                for line in results:
                    clean = line.replace(" ", "").upper()
                    if len(clean) > 20 and ("<<" in clean or clean.startswith(("P<", "I<", "A<"))):
                        mrz_candidates.append(clean)
                
                if mrz_candidates:
                    print(f"      ✅ Found {len(mrz_candidates)} MRZ candidate(s):")
                    for i, candidate in enumerate(mrz_candidates[:5], 1):  # Show first 5
                        print(f"         [{i}] {candidate[:80]}{'...' if len(candidate) > 80 else ''}")
                else:
                    print(f"      ❌ No MRZ candidates found")
                    
                # 5. Show lines sorted by length (longest first, likely MRZ)
                if results:
                    sorted_by_length = sorted(results, key=lambda x: len(x.replace(" ", "")), reverse=True)
                    print(f"      📏 Top 5 longest lines (likely MRZ):")
                    for i, line in enumerate(sorted_by_length[:5], 1):
                        clean = line.replace(" ", "").upper()
                        print(f"         [{i}] ({len(clean)} chars) {clean[:70]}{'...' if len(clean) > 70 else ''}")

            except Exception as e:
                print(f"      ❌ CRASHED: {e}")
            
            print(f"      ⏱️ {time.time() - start:.2f}s")
        print("\n")

if __name__ == "__main__":
    # Use the PDF directly this time
    TEST_FILE = "/Users/nnigam/Personal Projects/Personal/auto-form-filler/docs/local/Passport Front.pdf"
    run_easyocr_tournament(TEST_FILE)