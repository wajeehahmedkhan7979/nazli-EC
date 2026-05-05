"""
AnchorRoiService — minimal anchor-based ROI shift.

If a template defines an anchor, this service:
1. Searches for the anchor text in its anchor_region via OCR
2. Computes the offset between expected and found position
3. Returns a shifted RoiDef

Falls back silently to the static RoiDef if anchor not found.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from PIL import Image

from src.domain.types import RoiDef, TemplateConfig

logger = logging.getLogger(__name__)


@dataclass
class AnchorResult:
    found: bool
    dx_frac: float = 0.0   # fractional shift x (relative to page width)
    dy_frac: float = 0.0   # fractional shift y (relative to page height)
    drift_detected: bool = False
    retry_used: bool = False


class AnchorRoiService:
    """
    Detects anchor text position and shifts field ROIs accordingly.
    Intended as an optional layer wrapping RoiMappingService.
    """

    def __init__(self, ocr_lang: str = "japan") -> None:
        self._ocr_lang = ocr_lang
        self._paddle: Optional[object] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Remove punctuation, collapse whitespace, full-width handling."""
        import re
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[^\w]', '', text)
        return text.upper()

    # ------------------------------------------------------------------

    def resolve_roi(
        self,
        image: Image.Image,
        template: TemplateConfig,
        field_name: str,
    ) -> RoiDef:
        """
        Return a (possibly shifted) RoiDef for field_name.
        If no anchor is configured or anchor not found → return static ROI.
        """
        static_roi = template.fields.get(field_name)
        if static_roi is None:
            raise KeyError(f"Field '{field_name}' not in template")

        # 1. Try anchor (strict region)
        # In the new schema, anchor_cfg has expected coords, and field has offset.
        field_cfg = template.field_configs.get(field_name, {})
        anchor_cfg = getattr(template, "anchors", {}).get(field_cfg.get("anchor"))
        if not anchor_cfg:
            anchor_cfg = getattr(template, "anchor_refs", {}).get(field_name)
            
        if not anchor_cfg:
            return static_roi, False

        anchor_result = self._detect_anchor_with_retry(image, anchor_cfg)
        if not anchor_result.found:
            logger.debug("Anchor not found for %s — using static ROI", field_name)
            return static_roi, False

        dx = anchor_result.dx_frac
        dy = anchor_result.dy_frac
        
        # Base expected anchor coords + field offset + drift dx/dy
        offset = field_cfg.get("offset", {"dx": 0.0, "dy": 0.0})
        expected = anchor_cfg.get("expected", {"x": 0.0, "y": 0.0})
        
        base_x = expected.get("x", 0.0) + offset.get("dx", 0.0)
        base_y = expected.get("y", 0.0) + offset.get("dy", 0.0)
        
        new_x = base_x + dx
        new_y = base_y + dy
        
        width = static_roi.width
        height = static_roi.height
        
        shifted = RoiDef(
            field=static_roi.field,
            x=max(0.0, min(1.0 - width, new_x)),
            y=max(0.0, min(1.0 - height, new_y)),
            width=width,
            height=height,
        )
        logger.info(
            "Anchor shift for %s: dx=%.4f dy=%.4f (retry=%s)",
            field_name, dx, dy, anchor_result.retry_used
        )
        # Record metrics
        try:
            from src.core.output.metrics import metrics
            if hasattr(metrics, "record_anchor"):
                metrics.record_anchor(found=True, retry_used=anchor_result.retry_used, drift=anchor_result.drift_detected)
        except ImportError:
            pass

        return shifted, anchor_result.drift_detected

    # ------------------------------------------------------------------

    def _detect_anchor_with_retry(self, image: Image.Image, anchor_cfg: dict) -> AnchorResult:
        # 1. Try anchor (strict region)
        res = self._detect_anchor(image, anchor_cfg)
        if res.found:
            return res
            
        # 2. Retry anchor (expanded region +20%)
        expanded_cfg = dict(anchor_cfg)
        r = expanded_cfg.get("search_region", expanded_cfg.get("region", {}))
        expanded_cfg["search_region"] = {
            "x": max(0.0, r.get("x", 0) - 0.1),
            "y": max(0.0, r.get("y", 0) - 0.1),
            "w": min(1.0, r.get("w", 0.2) + 0.2),
            "h": min(1.0, r.get("h", 0.1) + 0.2),
        }
        res2 = self._detect_anchor(image, expanded_cfg)
        if res2.found:
            res2.retry_used = True
            return res2
            
        # 3. If still fail
        try:
            from src.core.output.metrics import metrics
            if hasattr(metrics, "record_anchor"):
                metrics.record_anchor(found=False, retry_used=True, drift=False)
        except ImportError:
            pass
            
        return AnchorResult(found=False)

    def _detect_anchor(self, image: Image.Image, anchor_cfg: dict) -> AnchorResult:
        """
        anchor_cfg = {
          "text": "車台番号",
          "expected": {"x": 0.30, "y": 0.28},   # expected fractional position
          "region":   {"x": 0.20, "y": 0.22, "width": 0.25, "height": 0.12}
        }
        """
        try:
            ocr = self._get_paddle()
            w, h = image.size
            region = anchor_cfg.get("search_region", anchor_cfg.get("region", {}))
            w_key = "w" if "w" in region else "width"
            h_key = "h" if "h" in region else "height"
            
            x1 = int(region.get("x", 0) * w)
            y1 = int(region.get("y", 0) * h)
            x2 = int((region.get("x", 0) + region.get(w_key, 1)) * w)
            y2 = int((region.get("y", 0) + region.get(h_key, 1)) * h)
            crop = image.crop((x1, y1, x2, y2))

            result = ocr.ocr(np.array(crop))
            target = anchor_cfg.get("text", "").upper()
            expected = anchor_cfg.get("expected", {})
            exp_x = expected.get("x", 0.0)
            exp_y = expected.get("y", 0.0)

            target_norm = self._normalize_text(target)

            if result:
                page_res = result[0] if isinstance(result, list) and result else []
                if isinstance(page_res, list):
                    for line in page_res:
                        if isinstance(line, list) and len(line) >= 2:
                            box, content = line[0], line[1]
                            if isinstance(content, (list, tuple)) and len(content) >= 2:
                                text_norm = self._normalize_text(str(content[0]))
                                line_conf = float(content[1])
                                
                                if target_norm in text_norm and line_conf > 0.80:
                                    # Midpoint of found bounding box (relative to crop)
                                    cx = float(np.mean([p[0] for p in box])) / (x2 - x1)
                                    cy = float(np.mean([p[1] for p in box])) / (y2 - y1)
                                    # Convert crop-relative to page-relative
                                    found_x = region.get("x", 0) + cx * region.get(w_key, 1)
                                    found_y = region.get("y", 0) + cy * region.get(h_key, 1)
                                    
                                    dx = found_x - exp_x
                                    dy = found_y - exp_y
                                    
                                    drift_magnitude = (dx**2 + dy**2) ** 0.5
                                    drift_detected = drift_magnitude > 0.04
                                    return AnchorResult(found=True, dx_frac=dx, dy_frac=dy, drift_detected=drift_detected)
        except Exception as exc:
            logger.warning("Anchor detection error: %s", exc)

        return AnchorResult(found=False)

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR  # type: ignore
            self._paddle = PaddleOCR(
                use_angle_cls=False, lang=self._ocr_lang
            )
        return self._paddle
