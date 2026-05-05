"""
Domain types for the EC PDF extraction pipeline.
All types are pure dataclasses — no framework dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Field-level confidence bundle
# ---------------------------------------------------------------------------

@dataclass
class FieldConfidence:
    chassis_no: float = 0.0
    issue_date: float = 0.0
    expiration_date: float = 0.0


# ---------------------------------------------------------------------------
# ROI definition (fractional coordinates, 0.0–1.0 relative to page size)
# ---------------------------------------------------------------------------

@dataclass
class RoiDef:
    field: str
    x: float
    y: float
    width: float
    height: float

    def to_pixels(self, page_w: int, page_h: int) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in pixel space."""
        x1 = int(self.x * page_w)
        y1 = int(self.y * page_h)
        x2 = int((self.x + self.width) * page_w)
        y2 = int((self.y + self.height) * page_h)
        return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Template anchor for detection
# ---------------------------------------------------------------------------

@dataclass
class AnchorDef:
    name: str
    text: str
    min_confidence: float = 0.70


# ---------------------------------------------------------------------------
# Full template configuration
# ---------------------------------------------------------------------------

@dataclass
class TemplateConfig:
    template_id: str
    document_type: str
    page_size: str
    rotation: int
    anchors: dict[str, dict]    # name -> anchor config
    field_configs: dict[str, dict] # field_name -> field config
    fields: dict[str, RoiDef]  # field_name -> base RoiDef
    match_criteria: dict = field(default_factory=dict)
    roi_map_version: str = "1.0"


# ---------------------------------------------------------------------------
# OCR result for a single region
# ---------------------------------------------------------------------------

@dataclass
class OcrResult:
    raw_text: str
    confidence: float
    engine: str  # "paddleocr" | "tesseract"
    bounding_box: Optional[tuple[int, int, int, int]] = None  # x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Parsed field value
# ---------------------------------------------------------------------------

@dataclass
class ParsedField:
    value: Optional[str]           # normalized value or None
    raw_text: str                  # original OCR text
    parse_confidence: float        # 0.0–1.0
    ocr_confidence: float          # 0.0–1.0
    combined_confidence: float     # weighted combination
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-page extraction result
# ---------------------------------------------------------------------------

ExtractionMethod = Literal["template_roi_ocr", "fallback_full_page_ocr", "failed"]

@dataclass
class PageResult:
    page_number: int               # 1-indexed
    chassis_no: Optional[str]
    issue_date: Optional[str]      # ISO 8601 YYYY-MM-DD
    expiration_date: Optional[str] # ISO 8601 YYYY-MM-DD
    confidence: FieldConfidence
    method: ExtractionMethod
    template_id: Optional[str]
    roi_map_version: Optional[str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Full document extraction result
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    file_id: str
    source_path: str
    status: Literal["completed", "partial", "failed"]
    page_count: int
    results: list[PageResult]
    processed_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    @property
    def summary(self) -> dict:
        success = sum(
            1 for p in self.results
            if p.chassis_no or p.issue_date or p.expiration_date
        )
        fallback = sum(
            1 for p in self.results if p.method == "fallback_full_page_ocr"
        )
        failed = sum(1 for p in self.results if p.method == "failed")
        confidences = [
            (
                p.confidence.chassis_no
                + p.confidence.issue_date
                + p.confidence.expiration_date
            ) / 3
            for p in self.results
            if p.method != "failed"
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "success_pages": success,
            "failed_pages": failed,
            "fallback_pages": fallback,
            "average_confidence": round(avg_conf, 4),
        }
