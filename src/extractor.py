"""
EC PDF Extractor — Surgical ROI with PaddleOCR.

These dimensions were manually verified by the user:
  Chassis:     (434, 61)   to (600, 81)
  Expiry Date: (182, 300) to (350, 315)
  Issue Date:  (160, 450) to (273, 470)

The system renders at 300 DPI, crops these regions, cleans them, and runs PaddleOCR.
"""
from __future__ import annotations

import os
# Disable unstable Paddle PIR compiler and oneDNN to fix crash on some systems
os.environ["PADDLE_PIR_ENABLE"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import re
import time
import unicodedata
import functools
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import fitz
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 300

# Latest calibrated Point coordinates for Landscape A4 (842x595)
ROIS = {
    "chassis_no":  {"x1": 435, "y1": 185, "x2": 650, "y2": 215},
    "expiry_date": {"x1": 165, "y1": 60,  "x2": 380, "y2": 95},
    "issue_date":  {"x1": 110, "y1": 535, "x2": 320, "y2": 565},
}


@dataclass
class PageResult:
    page_number: int
    chassis_no:  Optional[str]
    issue_date:  Optional[str]
    expiry_date: Optional[str]
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
# OCR Engine (Lazy Loaded PaddleOCR)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_ocr():
    """Initialises PaddleOCR only once."""
    try:
        from paddleocr import PaddleOCR
        # Using English and Japanese for mixed EC documents
        # Using mobile model + disabling MKLDNN to kill OneDNN errors
        return PaddleOCR(
            lang="en", 
            enable_mkldnn=False
        )
    except ImportError:
        logger.error("PaddleOCR not installed. Falling back to dummy.")
        return None


def _run_ocr(img: Image.Image) -> str:
    """Runs PaddleOCR on a PIL Image and returns joined text lines."""
    ocr = get_ocr()
    if not ocr:
        return ""
    
    # Convert PIL to BGR for Paddle
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    try:
        try:
            # Attempt recognition-only mode (det=False)
            result = ocr.ocr(arr, det=False, cls=True)
        except (TypeError, ValueError):
            # Fallback for older/different PaddleOCR versions that don't support 'det' in .ocr()
            logger.warning("PaddleOCR does not support 'det' argument, falling back to full mode.")
            result = ocr.ocr(arr)

        if not result or not result[0] or not result[0][0]:
            return ""
        
        # Handle different return formats (det=False vs full mode)
        if isinstance(result[0][0], (list, tuple)):
            # With det=False: [[('text', confidence)]]
            return str(result[0][0][0])
        else:
            # Full mode: [[ [box, [text, conf]], ... ]]
            return " ".join([line[1][0] for line in result[0]])
    except Exception as exc:
        logger.error("PaddleOCR error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Image Handling & "Surgical Crop"
# ---------------------------------------------------------------------------

def _crop(img: Image.Image, roi: dict) -> Image.Image:
    """Crops the image using absolute pixel coordinates."""
    return img.crop((roi["x1"], roi["y1"], roi["x2"], roi["y2"]))


def _preprocess(img: Image.Image) -> Image.Image:
    """Cleans the small ROI crop for better OCR."""
    arr = np.array(img.convert("L"))
    # Upscale slightly for small text
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Denoise and threshold
    arr = cv2.medianBlur(arr, 3)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# Field Parsers
# ---------------------------------------------------------------------------

def _parse_chassis(text: str) -> Optional[str]:
    """Cleans and validates chassis number."""
    text = unicodedata.normalize("NFKC", text).upper().replace(" ", "")
    # Expecting: [PREFIX]-[SUFFIX] e.g. NKE165-7242861
    m = re.search(r"([A-Z0-9]+)-([0-9]{5,10})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def _parse_date(text: str) -> Optional[str]:
    """Parses date from OCR string (expects YYYY MM DD)."""
    nums = re.findall(r"\d+", text)
    if len(nums) < 3:
        # Fallback for YYYY MM (if day missing)
        if len(nums) == 2 and 2000 <= int(nums[0]) <= 2100:
             return f"{nums[0]}-{int(nums[1]):02d}-01"
        return None
    
    y = int(nums[0])
    m = int(nums[1])
    d = int(nums[2])
    
    # Simple fix for common OCR digit errors in years
    if 9000 <= y <= 9999: y = 2000 + (y % 100)
    
    if not (2000 <= y <= 2100): return None
    if not (1 <= m <= 12):      return None
    if not (1 <= d <= 31):      d = 1 # clamp
    
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Compatibility Exports for probe_rois.py
# ---------------------------------------------------------------------------

def _ocr_chassis(img: Image.Image) -> str: return _run_ocr(img)
def _ocr_date(img: Image.Image) -> str:    return _run_ocr(img)


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def _extract_page(page: fitz.Page, page_number: int) -> PageResult:
    t0 = time.monotonic()

    # 1. Render full page in memory
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 2. Extract Fields via Surgical ROI
    # Use dynamic page size (points) as reference for scaling
    ref_w = page.rect.width
    ref_h = page.rect.height
    
    W, H = full_img.size
    scale_x = W / ref_w
    scale_y = H / ref_h

    results = {}
    raw_texts = []
    
    for field_name, roi in ROIS.items():
        # Scale Points to Pixels
        px_roi = {
            "x1": int(roi["x1"] * scale_x),
            "y1": int(roi["y1"] * scale_y),
            "x2": int(roi["x2"] * scale_x),
            "y2": int(roi["y2"] * scale_y),
        }
        
        # Crop and Preprocess
        cropped = _crop(full_img, px_roi)
        processed = _preprocess(cropped)
        
        # Save debug crops for Page 1
        if page_number == 1:
            _save_crop_debug(field_name, processed)
        
        # OCR
        raw = _run_ocr(processed)
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


def _save_crop_debug(name: str, img: Image.Image) -> None:
    try:
        d = Path("static/debug")
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"crop_{name}.png")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Document entry point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    path = Path(pdf_path)
    results: list[PageResult] = []

    with fitz.open(str(path)) as doc:
        page_count = len(doc)
        logger.info("Extracting %d pages from %s using ROI-PaddleOCR", page_count, path.name)

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

    return ExtractionResult(file_name=path.name, page_count=page_count, results=results)