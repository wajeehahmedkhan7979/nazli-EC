"""
ROI probe — run this to find/verify field coordinates on any EC PDF page.

Usage:
    python scripts/probe_rois.py path/to/file.pdf [page_number]

Saves a debug image with the current ROI boxes drawn on it, and prints
the OCR results. Adjust ROIS in src/extractor.py to match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.extractor import (
    ROIS, RENDER_DPI,
    _preprocess, _run_ocr,
    _parse_chassis, _parse_date,
)


COLOURS = {
    "chassis_no": "red",
    "issue_date":  "blue",
    "expiry_date": "green",
}


def probe(pdf_path: str, page_num: int = 1):
    path = Path(pdf_path)
    with fitz.open(str(path)) as doc:
        page = doc[page_num - 1]

        # Dynamic reference from the actual page bounding box
        rect = page.rect
        ref_w = rect.width
        ref_h = rect.height

        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    W, H = img.size
    scale_x = W / ref_w
    scale_y = H / ref_h
    draw = ImageDraw.Draw(img)

    print(f"\nPage {page_num} — {W}×{H} px at {RENDER_DPI} DPI")
    print(f"PDF Points: {ref_w:.1f}×{ref_h:.1f}")
    print(f"Scale: x={scale_x:.4f}, y={scale_y:.4f}")
    print("-" * 60)

    for field_name, roi in ROIS.items():
        # Scale Points to Pixels
        x1 = max(0, int(roi["x1"] * scale_x))
        y1 = max(0, int(roi["y1"] * scale_y))
        x2 = min(W, int(roi["x2"] * scale_x))
        y2 = min(H, int(roi["y2"] * scale_y))

        colour = COLOURS.get(field_name, "yellow")
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=6)
        draw.text((x1 + 6, y1 + 6), field_name, fill=colour)

        # Crop, preprocess, OCR
        crop = img.crop((x1, y1, x2, y2))
        processed = _preprocess(crop)
        processed.save(f"debug_crop_{field_name}.png")

        if field_name == "chassis_no":
            raw = _run_ocr(processed, mode="chassis")
            parsed = _parse_chassis(raw)
        else:
            raw = _run_ocr(processed, mode="date")
            parsed = _parse_date(raw)

        print(f"  [{field_name}]")
        print(f"    ROI (pts): ({roi['x1']},{roi['y1']}) → ({roi['x2']},{roi['y2']})")
        print(f"    ROI (px) : ({x1},{y1}) → ({x2},{y2})")
        print(f"    OCR raw  : {repr(raw[:120])}")
        print(f"    Parsed   : {parsed}")
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
