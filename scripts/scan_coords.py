import fitz
import re
import sys

def scan(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    words = page.get_text("words")
    
    print(f"Page size: {page.rect}")
    print("-" * 40)
    
    chassis_pattern = re.compile(r"[A-Z0-9]+-[0-9]{5,}")
    date_pattern = re.compile(r"20\d{2}") # Look for years first
    
    for x0, y0, x1, y1, word, block_no, line_no, word_no in words:
        if chassis_pattern.search(word):
            print(f"Found Chassis candidate: '{word}' at ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f})")
        if date_pattern.search(word):
            print(f"Found Year candidate: '{word}' at ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f})")
            # Print surrounding words to find the full date
            # (In a real scenario we'd look for MM and DD nearby)
    
    # Also dump all text to see the structure
    print("-" * 40)
    print("Full text dump:")
    print(page.get_text())

if __name__ == "__main__":
    scan(sys.argv[1])
