import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
from functools import lru_cache

import cv2
import fitz
import numpy as np
from PIL import Image

@lru_cache(maxsize=1)
def get_ocr():
    from paddleocr import PaddleOCR
    # Mute PaddleOCR's noisy console output
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    # Initialize with stable settings
    return PaddleOCR(use_angle_cls=False, lang='japan', show_log=False)

logger = logging.getLogger(__name__)

RENDER_DPI = 300

@dataclass
class PageResult:
    page_number: int
    chassis_no: Optional[str]
    issue_date: Optional[str]
    expiry_date: Optional[str]
    warnings: List[str]
    elapsed_ms: float

@dataclass
class ExtractionResult:
    file_name: str
    page_count: int
    results: List[PageResult]

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "page_count": self.page_count,
            "results": [
                {
                    "page": r.page_number,
                    "chassis_no": r.chassis_no,
                    "issue_date": r.issue_date,
                    "expiry_date": r.expiry_date,
                    "warnings": r.warnings,
                    "elapsed_ms": round(r.elapsed_ms, 2),
                }
                for r in self.results
            ]
        }

def _get_preprocessed_roi(img: Image.Image, roi_coords: tuple, c_val: int, blur: bool = True) -> Image.Image:
    w, h = img.size
    crop = img.crop((int(roi_coords[0]*w), int(roi_coords[1]*h), int(roi_coords[2]*w), int(roi_coords[3]*h)))
    arr = np.array(crop.convert("L"))
    arr = cv2.medianBlur(arr, 3)
    if blur:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, c_val
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(binary)

def safe_ocr(arr):
    try:
        ocr = get_ocr()
        return ocr.ocr(arr, cls=False)
    except NotImplementedError as e:
        logger.error("OCR engine failure - fallback triggered", exc_info=True)
        return []

def _ocr_fullpage(img: Image.Image) -> str:
    arr = np.array(img.convert('RGB'))
    arr = arr[:, :, ::-1] 
    result = safe_ocr(arr)
    if not result or not result[0]:
        return ""
    return " ".join([line[1][0] for line in result[0]])

_CHASSIS_RE = re.compile(r"([A-Z\s]{2,12}[0-9A-Z\s]{2,8}[A-Z]?)\s*[-—–_=]{1,3}\s*([0-9A-Z\s]{5,15})\b")
_OCR_DIGIT_MAP = str.maketrans("LSOIBGZ", "1501862")

def _fix_chassis(prefix: str, suffix: str) -> Optional[str]:
    prefix = re.sub(r"\s+", "", prefix)
    suffix = re.sub(r"\s+", "", suffix)
    match = re.search(r"([A-Z]{2,6}\d{1,4}[A-Z]?)", prefix)
    if match: prefix = match.group(1)
    suffix = suffix.translate(_OCR_DIGIT_MAP)[:7]
    result = f"{prefix}-{suffix}"
    if re.match(r"^[A-Z]{2,10}\d{2,6}[A-Z]?-\d{5,7}$", result):
        return result
    return None

def extract_chassis(text: str) -> Optional[str]:
    text = unicodedata.normalize("NFKC", text).upper()
    for m in _CHASSIS_RE.finditer(text):
        res = _fix_chassis(m.group(1), m.group(2))
        if res: return res
    return None

def _extract_date_from_chunk(text: str) -> Optional[str]:
    # Reiwa pattern
    for m in re.finditer(r"R(?:eiwa|e)?\s*(\d{1,2})\s*[/年\-.]\s*(\d{1,2})\s*[/月\-.]\s*(\d{1,2})", text, re.IGNORECASE):
        y, m_v, d_v = int(m.group(1)) + 2018, int(m.group(2)), int(m.group(3))
        if 1 <= m_v <= 12 and 1 <= d_v <= 31:
            return f"{y:04d}-{m_v:02d}-{d_v:02d}"
    
    nums = re.findall(r"\d+", text)
    if len(nums) >= 3:
        y = int(nums[0])
        if 20 <= y <= 35: y += 2000 
        if 1980 <= y <= 2100:
            return f"{y:04d}-{int(nums[1]):02d}-{int(nums[2]):02d}"
    return None

def _extract_page(page: fitz.Page, page_number: int) -> PageResult:
    t0 = time.monotonic()
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    chassis_no = extract_chassis(_ocr_fullpage(_get_preprocessed_roi(img, (0.58, 0.07, 0.90, 0.16), 12)))
    expiry_date = _extract_date_from_chunk(_ocr_fullpage(_get_preprocessed_roi(img, (0.24, 0.55, 0.50, 0.64), 7, blur=False)))
    issue_date = _extract_date_from_chunk(_ocr_fullpage(_get_preprocessed_roi(img, (0.20, 0.88, 0.40, 0.97), 14)))

    elapsed = (time.monotonic() - t0) * 1000
    warns = []
    if not chassis_no: warns.append("PARSE_FAIL_CHASSIS")
    if not issue_date: warns.append("PARSE_FAIL_ISSUE_DATE")
    if not expiry_date: warns.append("PARSE_FAIL_EXPIRY_DATE")

    return PageResult(page_number, chassis_no, issue_date, expiry_date, warns, elapsed)

def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    pdf_path = Path(pdf_path)
    results = []
    doc = fitz.open(pdf_path)
    for i in range(doc.page_count):
        res = _extract_page(doc[i], i + 1)
        results.append(res)
    doc.close()
    return ExtractionResult(pdf_path.name, len(results), results)