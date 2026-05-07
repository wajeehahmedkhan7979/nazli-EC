"""Quick test: extract page 1 from the most recently uploaded PDF and print raw OCR output."""
import sys, json, logging
from pathlib import Path

# Enable DEBUG so we see the raw OCR text on failures
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s — %(message)s")

# Find the most recent cached result to get the PDF path, or use CLI arg
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    # Use fitz to test with a dummy — we need a real PDF
    print("Usage: python scripts/test_single_page.py <path_to_pdf>")
    print("\nRunning inline diagnostic on debug crops instead...\n")
    
    import pytesseract
    from PIL import Image
    
    debug_dir = Path("static/debug")
    if not debug_dir.exists():
        print("No debug crops found. Run the server and upload a PDF first.")
        sys.exit(1)
    
    for name in ["crop_chassis.png", "crop_issue.png", "crop_expiry.png"]:
        p = debug_dir / name
        if p.exists():
            img = Image.open(p)
            
            if "chassis" in name:
                config = "--oem 1 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            else:
                config = "--oem 1 --psm 6 -l eng"
            
            text = pytesseract.image_to_string(img, config=config).strip()
            print(f"{name}: {repr(text)}")
    
    sys.exit(0)

# Full page test
from src.extractor import extract_pdf
result = extract_pdf(pdf_path)
print(json.dumps(result.as_dict, indent=2, ensure_ascii=False))
