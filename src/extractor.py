"""
EC PDF Extractor — zero AI, zero paid services.

HYBRID strategy based on actual document analysis:
  - Chassis:     Full-page English OCR + regex (layout-independent, proven reliable)
  - Expiry Date: ROI crop of "Export scheduled day" section (middle of page)
  - Issue Date:  ROI crop of "Date of Application" section (bottom of page)

Pipeline per page:
  PyMuPDF render → full-page OCR for chassis → ROI crops for dates → regex parse
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 300

# ---------------------------------------------------------------------------
# ROI definitions for date fields (fractional coordinates)
# Measured from the actual annotated Export Certificate document.
# Page: 987 × 693 pt landscape → 4113 × 2888 px at 300 DPI
# ---------------------------------------------------------------------------
ROIS = {
    # Chassis: Top right area
    "chassis_no": dict(x=0.60, y=0.05, w=0.35, h=0.10),
    # Expiry: Shifted DOWN (y=0.64) to capture 'Export scheduled day'
    "expiry_date": dict(x=0.08, y=0.64, w=0.50, h=0.08),
    # Issue: Confirmed bottom area 'Date of Application'
    "issue_date":  dict(x=0.03, y=0.86, w=0.55, h=0.12),
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_number: int
    chassis_no:  Optional[str]
    issue_date:  Optional[str]   # ISO YYYY-MM-DD
    expiry_date: Optional[str]   # ISO YYYY-MM-DD
    warnings:    list[str] = field(default_factory=list)
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
                    "page":       r.page_number,
                    "chassis_no": r.chassis_no,
                    "issue_date": r.issue_date,
                    "expiry_date":r.expiry_date,
                    "warnings":   r.warnings,
                    "elapsed_ms": round(r.elapsed_ms, 1),
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_fullpage(img: Image.Image) -> Image.Image:
    """Grayscale + Otsu for full-page OCR (no upscale needed at 300 DPI)."""
    arr = np.array(img.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def _preprocess_crop(crop: Image.Image) -> Image.Image:
    """Enhanced preprocessing for small, stamped text areas."""
    # Convert to grayscale
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    
    # 3x Upscale using Cubic Interpolation for smoother character edges
    arr = cv2.resize(arr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Denoise to handle stamp artifacts
    arr = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
    
    # Otsu's threshold to create a clean black-and-white mask
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _ocr_fullpage(img: Image.Image) -> str:
    """Full-page English OCR for chassis extraction."""
    import pytesseract
    return pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng").strip()


def _ocr_date_crop(img: Image.Image) -> str:
    import pytesseract
    # PSM 7: Treat the image as a single text line.
    # OEM 1: Neural nets LSTM engine only.
    custom_config = r'--oem 1 --psm 7 -l eng'
    return pytesseract.image_to_string(img, config=custom_config).strip()


# ---------------------------------------------------------------------------
# ROI crop helper
# ---------------------------------------------------------------------------

def _crop(img: Image.Image, roi: dict) -> Image.Image:
    W, H = img.size
    x1 = int(roi["x"] * W)
    y1 = int(roi["y"] * H)
    x2 = int((roi["x"] + roi["w"]) * W)
    y2 = int((roi["y"] + roi["h"]) * H)
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Chassis: letters + digits + dash + digits
_CHASSIS_RE = re.compile(
    r"([A-Z]{2,10}[0-9]{2,6}[A-Z]?)\s*[-—–]\s*([0-9]{5,10})"
)

# Separator noise in OCR output
_SEP = r"[\s._,;:|]*"

# Updated Regex to allow 9, 8, or 2 as the leading digit
_DATE_ENG_RE = re.compile(
    r"([298]\s*0\s*\d\s*\d)" + _SEP + 
    r"[A-Za-z]{2,5}" + _SEP + 
    r"(\d{1,2})" + _SEP + 
    r"[A-Za-z]{2,6}" + _SEP + 
    r"(\d{1,2})",
    re.IGNORECASE,
)

# Fallback: "Bday" where B=8 (digit merged into word)
_DATE_ENG_RE2 = re.compile(
    r"([2980]\s*0\s*\d\s*\d)" + _SEP +
    r"[A-Za-z]{2,5}" + _SEP +
    r"(\d{1,2})" + _SEP +
    r"[A-Za-z]{2,6}" + _SEP +
    r"([B8])\s*[dD][aeio]*[yY]",
    re.IGNORECASE,
)

# Japanese era date: 令和8 (2026) 4月 9日 — garbled through English OCR
# Matches: (YYYY) then digit(s) then letter then digit(s)
_DATE_JP_GARBLED_RE = re.compile(
    r"\(\s*(\d{4})\s*\)" + _SEP +
    r"[#]?\s*(\d{1,2})\s*[A-Za-z#]+" + _SEP +
    r"(\d{1,2})"
)

# ISO format: 2026-03-27 or 2026/03/27
_DATE_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")


def _clean_year(raw: str) -> int:
    """Convert OCR year like '9026' or '20 26' to int 2026."""
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) == 4:
        # If it ends in '026' but starts with an OCR error (9, 8, 0)
        if digits.endswith("026") and digits[0] in "980":
            return 2026
    return int(digits)


# ---------------------------------------------------------------------------
# Chassis parser
# ---------------------------------------------------------------------------

def _fix_chassis_ocr(prefix: str, suffix: str) -> Optional[str]:
    """Apply common OCR corrections to a chassis number."""
    suffix = (suffix.replace('L', '1').replace('S', '5').replace('O', '0')
              .replace('I', '1').replace('B', '8').replace('G', '6').replace('Z', '2'))

    m = re.match(r'([A-Z]+)([0-9A-Z]+)', prefix)
    if m:
        letters = m.group(1)
        digits = (m.group(2).replace('L', '1').replace('S', '5').replace('O', '0')
                  .replace('I', '1').replace('B', '8').replace('G', '6').replace('Z', '2'))
        prefix = letters + digits

    if len(suffix) > 7:
        suffix = suffix[:7]

    if len(prefix) < 3 or len(prefix) > 10 or len(suffix) < 5:
        return None

    return f"{prefix}-{suffix}"


def _extract_chassis(text: str) -> Optional[str]:
    """Find chassis number in full-page OCR text."""
    text = unicodedata.normalize("NFKC", text).upper()
    for m in _CHASSIS_RE.finditer(text):
        chassis = _fix_chassis_ocr(m.group(1), m.group(2))
        if chassis:
            return chassis
    return None


# ---------------------------------------------------------------------------
# Date parser — tries multiple patterns on a date-crop OCR text
# ---------------------------------------------------------------------------

def _parse_date(text: str) -> Optional[str]:
    """Extract a date from OCR text using multiple fallback patterns."""
    text = unicodedata.normalize("NFKC", text)

    # Pattern 1: English "2026 year 3 month 27 day" (fuzzy)
    for pattern in [_DATE_ENG_RE, _DATE_ENG_RE2]:
        m = pattern.search(text)
        if m:
            try:
                year = _clean_year(m.group(1))
                month = int(m.group(2))
                day_raw = m.group(3)
                day = int(day_raw.replace('B', '8').replace('b', '8'))
                d = date(year, month, day)
                if 2000 <= d.year <= 2100:
                    return d.isoformat()
            except (ValueError, OverflowError):
                pass

    # Pattern 2: Garbled Japanese era "(2026) 4A 9" from English OCR
    m = _DATE_JP_GARBLED_RE.search(text)
    if m:
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d = date(year, month, day)
            if 2000 <= d.year <= 2100:
                return d.isoformat()
        except (ValueError, OverflowError):
            pass

    # Pattern 3: ISO date
    m = _DATE_ISO_RE.search(text)
    if m:
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d = date(year, month, day)
            if 2000 <= d.year <= 2100:
                return d.isoformat()
        except (ValueError, OverflowError):
            pass

    return None


# ---------------------------------------------------------------------------
# Single page extraction
# ---------------------------------------------------------------------------

def _extract_page(page: fitz.Page, page_number: int) -> PageResult:
    t0 = time.monotonic()
    warnings: list[str] = []

    # 1 — Render to PIL Image at 300 DPI
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 2 — ROI crops
    chassis_raw = _crop(img, ROIS["chassis_no"])
    expiry_raw  = _crop(img, ROIS["expiry_date"])
    issue_raw   = _crop(img, ROIS["issue_date"])

    chassis_proc = _preprocess_crop(chassis_raw)
    expiry_proc  = _preprocess_crop(expiry_raw)
    issue_proc   = _preprocess_crop(issue_raw)

    # 3 — OCR
    import pytesseract
    try:
        chassis_text = pytesseract.image_to_string(chassis_proc, config="--oem 1 --psm 6 -l eng").strip()
    except Exception as exc:
        logger.error("p%d chassis OCR failed: %s", page_number, exc)
        chassis_text = ""
        warnings.append("OCR_FAILED_CHASSIS")

    chassis_no = _extract_chassis(chassis_text)

    try:
        expiry_text = _ocr_date_crop(expiry_proc)
    except Exception as exc:
        expiry_text = ""
        warnings.append("OCR_FAILED_EXPIRY")

    try:
        issue_text = _ocr_date_crop(issue_proc)
    except Exception as exc:
        issue_text = ""
        warnings.append("OCR_FAILED_ISSUE")

    expiry_date = _parse_date(expiry_text)
    issue_date  = _parse_date(issue_text)

    # 4 — Debug output for first 3 pages
    if page_number <= 3:
        try:
            debug_dir = Path("static/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            expiry_raw.save(debug_dir / f"raw_expiry_p{page_number}.png")
            issue_raw.save(debug_dir / f"raw_issue_p{page_number}.png")
            chassis_raw.save(debug_dir / f"raw_chassis_p{page_number}.png")
            expiry_proc.save(debug_dir / f"proc_expiry_p{page_number}.png")
            issue_proc.save(debug_dir / f"proc_issue_p{page_number}.png")
            chassis_proc.save(debug_dir / f"proc_chassis_p{page_number}.png")
            (debug_dir / f"ocr_expiry_p{page_number}.txt").write_text(expiry_text, encoding="utf-8")
            (debug_dir / f"ocr_issue_p{page_number}.txt").write_text(issue_text, encoding="utf-8")
            (debug_dir / f"ocr_chassis_p{page_number}.txt").write_text(chassis_text, encoding="utf-8")
            
            if page_number == 1:
                chassis_proc.save(debug_dir / "crop_chassis.png")
                issue_proc.save(debug_dir / "crop_issue.png")
                expiry_proc.save(debug_dir / "crop_expiry.png")
        except Exception:
            pass

    # 5 — Warnings
    if chassis_no is None:
        warnings.append("PARSE_FAIL_CHASSIS")
    if issue_date is None:
        warnings.append("PARSE_FAIL_ISSUE_DATE")
    if expiry_date is None:
        warnings.append("PARSE_FAIL_EXPIRY_DATE")

    if issue_date and expiry_date and expiry_date < issue_date:
        warnings.append("DATE_INCONSISTENT")

    elapsed = (time.monotonic() - t0) * 1000
    return PageResult(
        page_number=page_number,
        chassis_no=chassis_no,
        issue_date=issue_date,
        expiry_date=expiry_date,
        warnings=warnings,
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Document-level entry point
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    """Extract chassis_no, issue_date, expiry_date from every page."""
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
                result = PageResult(page_number=pn, chassis_no=None,
                                    issue_date=None, expiry_date=None,
                                    warnings=[f"PAGE_FAILED: {exc}"])
            results.append(result)
            logger.info(
                "p%d → chassis=%s  issue=%s  expiry=%s  warn=%s  %.0fms",
                pn, result.chassis_no, result.issue_date,
                result.expiry_date, result.warnings, result.elapsed_ms,
            )

    return ExtractionResult(file_name=path.name, page_count=page_count, results=results)
