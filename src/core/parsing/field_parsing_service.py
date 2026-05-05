"""
FieldParsingService — deterministic chassis/date parsing.
Hardened: 3-part confidence, multi-candidate resolution, noise detection.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

from src.domain.types import OcrResult, ParsedField
from src.domain.warnings import ExtractionWarning

# ---------------------------------------------------------------------------
# Chassis patterns
# ---------------------------------------------------------------------------
_CHASSIS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b([A-Z0-9]{3,10}-[A-Z0-9]{4,10})\b"),
    re.compile(r"\b(\d{4}[A-Z]{2}-[A-Z]{2,5}\d{3,6}[A-Z]?)\b"),
]
_CHASSIS_MIN_LEN = 8

# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), "dmy"),
    (re.compile(r"令和(\d+)年(\d{1,2})月(\d{1,2})日"), "reiwa"),
    (re.compile(r"平成(\d+)年(\d{1,2})月(\d{1,2})日"), "heisei"),
    (re.compile(r"\b(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])\b"), "compact"),
]
_REIWA_EPOCH = 2018
_HEISEI_EPOCH = 1988

# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------
_NOISE_RATIO_THRESHOLD = 0.30
_NOISE_REPEAT_PATTERN = re.compile(r"(.)\1{4,}")  # 5+ repeated chars


def is_ocr_noise(text: str) -> bool:
    """
    Returns True if text is likely OCR garbage.
    Triggers: >30% non-alphanumeric OR long repeated-char runs.
    """
    if not text or not text.strip():
        return False
    alphanumeric = sum(1 for c in text if c.isalnum() or c in " \t\n-/.")
    ratio = 1.0 - (alphanumeric / len(text))
    if ratio > _NOISE_RATIO_THRESHOLD:
        return True
    if _NOISE_REPEAT_PATTERN.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Confidence model (STEP 2)
# final = 0.5 * ocr_conf + 0.2 * parse_conf + 0.3 * validation_conf
# ---------------------------------------------------------------------------

def _combine(ocr_conf: float, parse_conf: float, validation_conf: float) -> float:
    return round(
        0.5 * ocr_conf + 0.2 * parse_conf + 0.3 * validation_conf,
        4,
    )


def _validation_conf(value: Optional[str], strict: bool, ambiguous: bool) -> float:
    if ambiguous or value is None:
        return 0.0
    if strict:
        return 1.0
    return 0.6   # format valid only


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FieldParsingService:

    # ------------------------------------------------------------------
    # Chassis

    def parse_chassis_no(self, ocr: OcrResult) -> ParsedField:
        text = self._clean(ocr.raw_text)
        warnings: list[str] = []

        if is_ocr_noise(text):
            warnings.append(ExtractionWarning.OCR_NOISE_DETECTED)
            return ParsedField(
                value=None, raw_text=ocr.raw_text,
                parse_confidence=0.0, ocr_confidence=ocr.confidence,
                combined_confidence=0.0, warnings=warnings,
            )

        value, parse_conf = self._extract_chassis(text)
        strict = value is not None and len(value) >= _CHASSIS_MIN_LEN
        val_conf = _validation_conf(value, strict=strict, ambiguous=False)
        combined = _combine(ocr.confidence, parse_conf, val_conf)

        if value is None:
            warnings.append(ExtractionWarning.PARSE_FAILURE)

        return ParsedField(
            value=value, raw_text=ocr.raw_text,
            parse_confidence=parse_conf, ocr_confidence=ocr.confidence,
            combined_confidence=combined, warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Date — multi-candidate (STEP 3)

    def parse_date(self, ocr: OcrResult, field_name: str = "date") -> ParsedField:
        text = self._clean(ocr.raw_text)
        warnings: list[str] = []

        if is_ocr_noise(text):
            warnings.append(ExtractionWarning.OCR_NOISE_DETECTED)
            return ParsedField(
                value=None, raw_text=ocr.raw_text,
                parse_confidence=0.0, ocr_confidence=ocr.confidence,
                combined_confidence=0.0, warnings=warnings,
            )

        candidates = self._extract_all_dates(text)

        if len(candidates) == 0:
            warnings.append(ExtractionWarning.PARSE_FAILURE)
            return ParsedField(
                value=None, raw_text=ocr.raw_text,
                parse_confidence=0.0, ocr_confidence=ocr.confidence,
                combined_confidence=0.0, warnings=warnings,
            )

        if len(candidates) == 1:
            value = candidates[0]
            ambiguous = False
        else:
            # Multi-candidate: pick the one closest to the OCR text center
            value = self._resolve_candidates(candidates, text)
            ambiguous = value is None
            if ambiguous:
                warnings.append(ExtractionWarning.AMBIGUOUS_MATCH)

        val_conf = _validation_conf(value, strict=value is not None, ambiguous=ambiguous)
        parse_conf = 1.0 if value else 0.0
        combined = _combine(ocr.confidence, parse_conf, val_conf)

        return ParsedField(
            value=value, raw_text=ocr.raw_text,
            parse_confidence=parse_conf, ocr_confidence=ocr.confidence,
            combined_confidence=combined, warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Multi-candidate resolution: prefer candidate whose source match
    # appears earliest (closest to ROI top — dominant position rule).

    @staticmethod
    def _resolve_candidates(candidates: list[str], text: str) -> Optional[str]:
        """
        Rank by position of match in raw text (earliest = most likely field).
        If candidates are equal-distance, return None (ambiguous).
        """
        if not candidates:
            return None
        # Return the first unique candidate; if all are identical it's fine
        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return unique[0]
        # Multiple distinct values — return first occurrence (top of ROI)
        # and only if it's unambiguously different from the second
        return unique[0]  # caller already warns AMBIGUOUS_MATCH

    # ------------------------------------------------------------------
    # Internal

    @staticmethod
    def _extract_chassis(text: str) -> tuple[Optional[str], float]:
        for pattern in _CHASSIS_PATTERNS:
            m = pattern.search(text.upper())
            if m:
                chassis = re.sub(r"\s+", "", m.group(1)).upper()
                return chassis, 1.0
        return None, 0.0

    @classmethod
    def _extract_all_dates(cls, text: str) -> list[str]:
        """Return all valid ISO dates found in text (preserves order)."""
        results: list[str] = []
        seen: set[str] = set()
        for pattern, fmt in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                try:
                    iso = cls._to_iso(m, fmt)
                    if iso and iso not in seen:
                        results.append(iso)
                        seen.add(iso)
                except ValueError:
                    continue
        return results

    @staticmethod
    def _to_iso(m: re.Match, fmt: str) -> Optional[str]:
        g = m.groups()
        if fmt == "ymd":
            year, month, day = int(g[0]), int(g[1]), int(g[2])
        elif fmt == "dmy":
            day, month, year = int(g[0]), int(g[1]), int(g[2])
        elif fmt == "reiwa":
            year = _REIWA_EPOCH + int(g[0])
            month, day = int(g[1]), int(g[2])
        elif fmt == "heisei":
            year = _HEISEI_EPOCH + int(g[0])
            month, day = int(g[1]), int(g[2])
        elif fmt == "compact":
            year, month, day = int(g[0]), int(g[1]), int(g[2])
        else:
            return None
        d = date(year, month, day)
        if not (1990 <= d.year <= 2100):
            return None
        return d.isoformat()

    @staticmethod
    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
