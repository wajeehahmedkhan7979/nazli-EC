"""
ROI probe — run this to find/verify field coordinates on any EC PDF page.

Usage:
    python scripts/probe_rois.py path/to/file.pdf [page_number]

Saves a debug image with the current ROI boxes drawn on it, and prints
the fractional coordinates. Adjust ROIS in src/extractor.py to match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import ROIS, RENDER_DPI, _preprocess, _crop, _ocr_chassis, _ocr_date, _parse_chassis, _parse_date


COLOURS = {
    "chassis_no": "red",
    "issue_date":  "blue",
    "expiry_date": "green",
}


def probe(pdf_path: str, page_num: int = 1):
    path = Path(pdf_path)
    with fitz.open(str(path)) as doc:
        page = doc[page_num - 1]
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    W, H = img.size
    draw = ImageDraw.Draw(img)

    print(f"\nPage {page_num} — {W}×{H} px at {RENDER_DPI} DPI")
    print("-" * 60)

    for field, roi in ROIS.items():
        x1 = int(roi["x"] * W)
        y1 = int(roi["y"] * H)
        x2 = int((roi["x"] + roi["w"]) * W)
        y2 = int((roi["y"] + roi["h"]) * H)
        draw.rectangle([x1, y1, x2, y2], outline=COLOURS[field], width=6)
        draw.text((x1 + 6, y1 + 6), field, fill=COLOURS[field])

        crop = _preprocess(_crop(img.convert("RGB"), roi))

        if field == "chassis_no":
            raw = _ocr_chassis(crop)
            parsed = _parse_chassis(raw)
        else:
            raw = _ocr_date(crop)
            parsed = _parse_date(raw)

        print(f"  [{field}]")
        print(f"    ROI (frac): x={roi['x']} y={roi['y']} w={roi['w']} h={roi['h']}")
        print(f"    ROI  (px) : ({x1},{y1}) → ({x2},{y2})")
        print(f"    OCR raw   : {repr(raw[:120])}")
        print(f"    Parsed    : {parsed}")
        print()

    out = Path(f"probe_page{page_num}.png")
    # Save at reduced size for easier viewing
    img.thumbnail((2000, 2000))
    img.save(out)
    print(f"Debug image saved to: {out.resolve()}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf:
        print("Usage: python scripts/probe_rois.py path/to/file.pdf [page_number]")
        sys.exit(1)
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    probe(pdf, page)
