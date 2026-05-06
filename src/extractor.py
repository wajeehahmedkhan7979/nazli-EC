"""
EC PDF Extractor — zero AI, zero paid services.

Root-cause diagnosis (v3 rewrite):
  The previous ROI-crop approach had two fatal bugs:
    1. Chassis ROI: small crop OCR garbled spaced characters → 22% success.
    2. Expiry ROI: PSM-7 single-line on a 2-line Japanese+English block → empty → 33% success.
    3. Chassis regex: [0-9]{5,10} in suffix didn't allow OCR noise chars (O,I,B etc) → 78% miss.

FIX — Single full-page OCR pass + anchor-based extraction:
  Full-page OCR (PSM 6, English) returns all three values in its output.
  Proven by debug files. We extract fields by anchoring on known English label text.
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

import fitz
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 300


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
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_fullpage(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# OCR — one call per page
# ---------------------------------------------------------------------------

def _ocr_fullpage(img: Image.Image) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, config="--oem 1 --psm 6 -l eng")


# ---------------------------------------------------------------------------
# Chassis
# ---------------------------------------------------------------------------

_CHASSIS_RE = re.compile(
    r"([A-Z]{2,10}[0-9A-Z]{2,6}[A-Z]?)\s*[-—–_]{1,3}\s*([0-9A-Z\s]{5,15})\b"
)
_OCR_DIGIT_MAP = str.maketrans("LSOIBGZ", "1501862")


def _fix_chassis(prefix: str, suffix: str) -> Optional[str]:
    suffix = suffix.replace(" ", "")
    if prefix.startswith("I") and len(prefix) > 2:
        prefix = prefix[1:]
    m = re.match(r"([A-Z]+)([0-9A-Z]+)", prefix)
    if m:
        prefix = m.group(1) + m.group(2).translate(_OCR_DIGIT_MAP)
    suffix = suffix.translate(_OCR_DIGIT_MAP)[:7]
    result = f"{prefix}-{suffix}"
    if re.match(r"^[A-Z]{2,10}\d{2,6}[A-Z]?-\d{5,7}$", result):
        return result
    return None


def extract_chassis(text: str) -> Optional[str]:
    text = unicodedata.normalize("NFKC", text).upper()
    for m in _CHASSIS_RE.finditer(text):
        result = _fix_chassis(m.group(1), m.group(2))
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Expiry date  (Export Scheduled Day)
# ---------------------------------------------------------------------------

def extract_expiry(text: str) -> Optional[str]:
    """
    Anchor: 'Export scheduled day'
    Date format in OCR output: YYYY [garbled-word] M [garbled-word] D [garbled-word]
    Year OCR variants: '2026', '20 26', '9026', '8026'
    Day OCR variant:   'Bday' where B = 8
    """
    m_anchor = re.search(r"(?:export|cxport|expart|born|cxpert|expcrt)\s+(?:scheduled|acheduled|schedu[a-z]*|sch[\w]+)\b", text, re.IGNORECASE)
    if not m_anchor:
        return None
    idx = m_anchor.start()
    chunk = text[idx: idx + 150]
    chunk = re.sub(r"\bB(?=[dDfF])", "8", chunk)  # Bday → 8day
    nums = re.findall(r"\d+", chunk)

    year: Optional[int] = None
    j = 0
    while j < len(nums):
        v = int(nums[j])
        if 2000 <= v <= 2100:
            year = v; j += 1; break
        if v in (20, 80, 90) and j + 1 < len(nums):
            nv = int(nums[j + 1])
            if 20 <= nv <= 30:
                year = 2000 + nv if v in (80, 90) else int(f"{v}{nv}")
                j += 2; break
        j += 1

    if not year:
        return None

    md = [int(n) for n in nums[j:] if 1 <= int(n) <= 31]
    if len(md) < 2:
        return None
    try:
        return date(year, md[0], md[1]).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Issue date  (Date of Application stamp at the bottom)
# ---------------------------------------------------------------------------

def extract_issue(text: str) -> Optional[str]:
    """
    Anchor: parenthesised Gregorian year  (YYYY)  in the bottom stamp area.
    The garbled Japanese date 'sm8 (2026) #4H 98H' gives year=2026, month=4, day=9.
    '98' = day-digit '9' + garbled '日' glyph '8' → clamp to first digit if > 31.
    Skip any (YYYY) that appears in a mileage/km context.
    """
    best: Optional[str] = None
    for m in re.finditer(r"[\(\[\{]\s*(\d{4})\s*[\)\]\}]", text):
        y = int(m.group(1))
        if not (2000 <= y <= 2100):
            continue
        after = text[m.end(): m.end() + 30]
        if re.search(r"[A-Z]\)|\bkm\b", after, re.IGNORECASE):
            continue
        md_nums = re.findall(r"\d+", after)
        if len(md_nums) < 2:
            continue
        mo = int(md_nums[0])
        da = int(md_nums[1])
        if da > 31:
            da = int(str(da)[0])
        if 1 <= mo <= 12 and 1 <= da <= 31:
            try:
                best = date(y, mo, da).isoformat()
            except ValueError:
                pass
    return best


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def _extract_page(page: fitz.Page, page_number: int) -> PageResult:
    t0 = time.monotonic()

    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    proc = _preprocess_fullpage(img)

    try:
        raw_text = _ocr_fullpage(proc)
    except Exception as exc:
        logger.error("p%d OCR failed: %s", page_number, exc)
        return PageResult(
            page_number=page_number,
            chassis_no=None, issue_date=None, expiry_date=None,
            warnings=["OCR_FAILED"],
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    chassis_no  = extract_chassis(raw_text)
    expiry_date = extract_expiry(raw_text)
    issue_date  = extract_issue(raw_text)

    warnings: list[str] = []
    if chassis_no  is None: warnings.append("PARSE_FAIL_CHASSIS")
    if issue_date  is None: warnings.append("PARSE_FAIL_ISSUE_DATE")
    if expiry_date is None: warnings.append("PARSE_FAIL_EXPIRY_DATE")
    if issue_date and expiry_date and expiry_date < issue_date:
        warnings.append("DATE_INCONSISTENT")

    if page_number <= 3:
        _save_debug(page_number, img, raw_text)

    elapsed = (time.monotonic() - t0) * 1000
    return PageResult(
        page_number=page_number,
        chassis_no=chassis_no,
        issue_date=issue_date,
        expiry_date=expiry_date,
        warnings=warnings,
        raw_text=raw_text,
        elapsed_ms=elapsed,
    )


def _save_debug(pn: int, img: Image.Image, raw_text: str) -> None:
    try:
        d = Path("static/debug")
        d.mkdir(parents=True, exist_ok=True)
        thumb = img.copy()
        thumb.thumbnail((1800, 1800))
        thumb.save(d / f"page_{pn}.png")
        (d / f"ocr_fullpage_p{pn}.txt").write_text(raw_text, encoding="utf-8")
    except Exception as exc:
        logger.debug("Debug save failed: %s", exc)


# ---------------------------------------------------------------------------
# Document entry point
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

    return ExtractionResult(file_name=path.name, page_count=page_count, results=results)