"""
RoiMappingService — converts fractional template ROIs to pixel crops.
"""
from __future__ import annotations

from PIL import Image

from src.domain.types import RoiDef, TemplateConfig
from src.core.template.anchor_roi_service import AnchorRoiService

# Target fields — only these are ever extracted
TARGET_FIELDS = ("chassis_no", "issue_date", "expiration_date")


class RoiMappingService:
    """Maps template ROI definitions to cropped PIL Images."""

    # Expansion factor for fallback (percentage of dimension)
    EXPAND_FRACTION = 0.15

    def __init__(self, anchor_roi_service: AnchorRoiService | None = None):
        self._anchor_service = anchor_roi_service or AnchorRoiService()

    def get_field_roi(
        self,
        template: TemplateConfig,
        field_name: str,
    ) -> RoiDef:
        roi = template.fields.get(field_name)
        if roi is None:
            raise KeyError(f"Field '{field_name}' not in template '{template.template_id}'")
        return roi

    def crop_field(
        self,
        image: Image.Image,
        roi: RoiDef,
        expand: bool = False,
    ) -> Image.Image:
        """
        Crop the ROI from the image.
        If expand=True, grow the bounding box by EXPAND_FRACTION on each side
        (clamped to image bounds) — used in fallback retries.
        """
        w, h = image.size
        x1, y1, x2, y2 = roi.to_pixels(w, h)

        if expand:
            dx = int((x2 - x1) * self.EXPAND_FRACTION)
            dy = int((y2 - y1) * self.EXPAND_FRACTION)
            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(w, x2 + dx)
            y2 = min(h, y2 + dy)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid ROI crop for field '{roi.field}': "
                f"({x1},{y1})→({x2},{y2}) on {w}×{h} image"
            )

        return image.crop((x1, y1, x2, y2))

    def crop_all_fields(
        self,
        image: Image.Image,
        template: TemplateConfig,
        expand: bool = False,
    ) -> tuple[dict[str, Image.Image], list[str]]:
        """Return ({field_name: cropped_image}, warnings) for all target fields."""
        crops: dict[str, Image.Image] = {}
        warnings: list[str] = []
        drift_flagged = False
        
        for field_name in TARGET_FIELDS:
            try:
                roi, drift_detected = self._anchor_service.resolve_roi(image, template, field_name)
                if drift_detected and not drift_flagged:
                    warnings.append("TEMPLATE_DRIFT_DETECTED")
                    drift_flagged = True
                    
                crops[field_name] = self.crop_field(image, roi, expand=expand)
            except Exception:
                pass  # orchestrator will handle missing crops as fallback trigger
        return crops, warnings
