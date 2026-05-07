"""
EC PDF Extractor — Calibrated for Landscape Export Certificates.

Coordinate System:
  - All ROI coordinates are in PDF Points (1 pt = 1/72 inch).
  - A4 Landscape = 842 × 595 points.
  - At 300 DPI rendering: scale = 300/72 = 4.1667x → 3508 × 2479 pixels.
  - The _extract_page() function dynamically computes scale from page.rect.

OCR Engine:
  - Primary: Tesseract 5.x (stable, local, no GPU required)
  - The PaddlePaddle 3.3.1 detection model has a known PIR crash
    (ConvertPirAttribute2RuntimeAttribute) that is unfixable without
    rebuilding the paddle wheel.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 300

# ---------------------------------------------------------------------------
# CALIBRATED COORDINATES (PDF Points, 72 DPI)
# ---------------------------------------------------------------------------
# Based on user-annotated visual evidence for Landscape A4 (842×595 points).
#
# Chassis No:  Row 1, rightmost column ("車台番号 / Maker's serial number")
#              e.g. "NKE165-7242932"
#              Position: ~70-99% width, ~7-14% height
#
# Expiry Date: "輸出予定日" row ("Export scheduled day")
#              e.g. "令和8年 8月 6日 / 2026 year 8 month 6 day"
#              Position: ~21-51% width, ~58-64% height
#
# Issue Date:  Official stamp at bottom (Director-General date)
#              e.g. "2026 year 4 month 9 day"
#              Position: ~29-58% width, ~91-95% height
# ---------------------------------------------------------------------------
ROIS = {
    # Chassis No: Under "Maker's serial number"
    "chassis_no":  {"x1": 510, "y1": 70, "x2": 750, "y2": 100},
    
    # Expiry Date: "Export scheduled day" row
    "expiry_date": {"x1": 210, "y1": 350, "x2": 400, "y2": 380},
    
    # Issue Date: Official stamp at bottom, left of "Director-General"
    "issue_date":  {"x1": 180, "y1": 535, "x2": 325, "y2": 575},
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_number: int
    chassis_no:  Optional[str] = None
    issue_date:  Optional[str] = None
    expiry_date: Optional[str] = None
    warnings:    list[str] = field(default_factory=list)
    raw_text:    str = field(default="", repr=False)
    elapsed_ms:  float = 0.0


@dataclass
class ExtractionResult:
    file_name:  str
    page_count: int
    results:    list[PageResult]

    @property
    def as_dict(self) -> dict:
        return {
            "file_name":  self.file_name,
            "page_count": self.page_count,
            "results": [
                {
                    "page":        r.page_number,
                    "chassis_no":  r.chassis_no,
                    "issue_date":  r.issue_date,
                    "expiry_date": r.expiry_date,
                    "warnings":    r.warnings,
                    "elapsed_ms":  round(r.elapsed_ms, 1),
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# OCR Engine — Tesseract (primary, stable)
# ---------------------------------------------------------------------------

def _run_ocr(img: Image.Image, mode: str = "line") -> str:
    """
    Run Tesseract OCR on a preprocessed PIL Image.

    Args:
        img:  A preprocessed (binarized) PIL Image.
        mode: "chassis" for single-line alphanumeric,
              "date" for block-mode date extraction,
              "line" for generic single-line.
    """
    import pytesseract

    if mode == "chassis":
        # Single text line, alphanumeric + hyphen only
        config = (
            "--oem 1 --psm 7 -l eng "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )
    elif mode == "date":
        # Uniform block — expect digits, Japanese era markers, spaces
        config = "--oem 1 --psm 6 -l eng"
    else:
        config = "--oem 1 --psm 7 -l eng"

    try:
        text = pytesseract.image_to_string(img, config=config).strip()
        return text
    except Exception as exc:
        logger.error("Tesseract OCR error: %s", exc)
        return f"[OCR_ERROR: {exc}]"


# ---------------------------------------------------------------------------
# Image Handling & "Surgical Crop"
# ---------------------------------------------------------------------------

def _crop(img: Image.Image, roi: dict) -> Image.Image:
    """Crops the image using absolute pixel coordinates."""
    return img.crop((roi["x1"], roi["y1"], roi["x2"], roi["y2"]))


def _preprocess(img: Image.Image) -> Image.Image:
    """Cleans the small ROI crop for better OCR."""
    arr = np.array(img.convert("L"))
    # Upscale 2x for small text recognition
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Light denoise
    arr = cv2.medianBlur(arr, 3)
    # Otsu binarization
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# Field Parsers
# ---------------------------------------------------------------------------

def _parse_chassis(text: str) -> Optional[str]:
    """Extract chassis number pattern: LETTERS-DIGITS (e.g. NKE165-7242932)."""
    text = unicodedata.normalize("NFKC", text).upper().replace(" ", "")
    m = re.search(r"([A-Z0-9]{2,10})-(\d{5,10})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Fallback: return cleaned text if it has enough alphanumeric chars
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text)
    return cleaned if len(cleaned) >= 6 else None


def _parse_date(text: str) -> Optional[str]:
    """
    Extract date from OCR text.
    Handles formats like:
      - "2026 year 8 month 6 day"
      - "2026 8 6"
      - "令和 8 年 8 月 6 日"
    Returns: "YYYY-MM-DD" string or None.
    """
    nums = re.findall(r"\d+", text)
    if len(nums) >= 3:
        year, month, day = int(nums[0]), int(nums[1]), int(nums[2])
        # Handle Japanese era (Reiwa): small year number
        if year < 100:
            year += 2018  # Reiwa era offset
        # Basic sanity check
        if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year}-{month:02d}-{day:02d}"
    return None


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _save_crop_debug(name: str, img: Image.Image) -> None:
    """Save debug crop with naming compatible with the static UI."""
    try:
        d = Path("static/debug")
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"crop_{name}.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page Extraction
# ---------------------------------------------------------------------------

def _extract_page(page: fitz.Page, page_number: int) -> PageResult:
    t0 = time.monotonic()

    # 1. Render full page at high DPI
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 2. Dynamic scale from page's actual bounding box (handles non-standard PDFs)
    ref_w = page.rect.width
    ref_h = page.rect.height
    W, H = full_img.size
    scale_x = W / ref_w
    scale_y = H / ref_h

    # 3. Extract each field via surgical ROI
    results = {}
    raw_texts = []

    for field_name, roi in ROIS.items():
        # Scale PDF Points → rendered Pixels
        px_roi = {
            "x1": max(0, int(roi["x1"] * scale_x)),
            "y1": max(0, int(roi["y1"] * scale_y)),
            "x2": min(W, int(roi["x2"] * scale_x)),
            "y2": min(H, int(roi["y2"] * scale_y)),
        }

        cropped = _crop(full_img, px_roi)
        processed = _preprocess(cropped)

        # Save debug crops for page 1
        if page_number == 1:
            short_name = field_name.replace("_no", "").replace("_date", "")
            _save_crop_debug(short_name, processed)

        # OCR with field-specific mode
        if field_name == "chassis_no":
            raw = _run_ocr(processed, mode="chassis")
        else:
            raw = _run_ocr(processed, mode="date")

        raw_texts.append(f"{field_name}: {raw}")

        # Parse
        if field_name == "chassis_no":
            results[field_name] = _parse_chassis(raw)
        else:
            results[field_name] = _parse_date(raw)

    chassis_no  = results.get("chassis_no")
    expiry_date = results.get("expiry_date")
    issue_date  = results.get("issue_date")

    warnings: list[str] = []
    if chassis_no  is None: warnings.append("PARSE_FAIL_CHASSIS")
    if issue_date  is None: warnings.append("PARSE_FAIL_ISSUE_DATE")
    if expiry_date is None: warnings.append("PARSE_FAIL_EXPIRY_DATE")

    elapsed = (time.monotonic() - t0) * 1000
    return PageResult(
        page_number=page_number,
        chassis_no=chassis_no,
        issue_date=issue_date,
        expiry_date=expiry_date,
        warnings=warnings,
        raw_text=" | ".join(raw_texts),
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Document Entry Point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    path = Path(pdf_path)
    results: list[PageResult] = []

    with fitz.open(str(path)) as doc:
        page_count = len(doc)
        logger.info("Extracting %d pages from %s", page_count, path.name)

        for idx, page in enumerate(doc):
            pn = idx + 1
            try:
                result = _extract_page(page, pn)
            except Exception as exc:
                logger.error("p%d failed: %s", pn, exc)
                result = PageResult(
                    page_number=pn,
                    chassis_no=None, issue_date=None, expiry_date=None,
                    warnings=[f"PAGE_FAILED: {exc}"],
                )
            results.append(result)
            logger.info(
                "p%d → chassis=%s  issue=%s  expiry=%s  warn=%s  (%.0f ms)",
                pn, result.chassis_no, result.issue_date,
                result.expiry_date, result.warnings, result.elapsed_ms,
            )

    return ExtractionResult(
        file_name=path.name,
        page_count=page_count,
        results=results,
    )


# ---------------------------------------------------------------------------
# Compatibility Exports for probe_rois.py
# ---------------------------------------------------------------------------

def _ocr_chassis(img: Image.Image) -> str:
    return _run_ocr(img, mode="chassis")

def _ocr_date(img: Image.Image) -> str:
    return _run_ocr(img, mode="date")