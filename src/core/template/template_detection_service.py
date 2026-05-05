"""
TemplateDetectionService — identifies which template a page belongs to
by running a lightweight OCR pass over anchor regions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from src.domain.types import AnchorDef, RoiDef, TemplateConfig

logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads and caches TemplateConfig objects from YAML files."""

    def __init__(self, registry_path: str | Path) -> None:
        self._registry_path = Path(registry_path)
        self._cache: dict[str, TemplateConfig] = {}
        self._load_all()

    def _load_all(self) -> None:
        with open(self._registry_path) as f:
            registry = yaml.safe_load(f)

        base_dir = self._registry_path.parent
        for entry in registry.get("templates", []):
            if not entry.get("enabled", True):
                continue
            
            try:
                if "path" in entry:
                    tpl_path = Path(entry["path"])
                    if not tpl_path.is_absolute():
                        tpl_path = base_dir.parent / tpl_path
                    with open(tpl_path) as tf:
                        tpl_data = yaml.safe_load(tf)
                else:
                    tpl_data = entry
                    
                tpl = self._parse_template_dict(tpl_data)
                self._cache[tpl.template_id] = tpl
                logger.info("Loaded template: %s", tpl.template_id)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", entry.get("id"), exc)

    def get(self, template_id: str) -> Optional[TemplateConfig]:
        return self._cache.get(template_id)

    def all(self) -> list[TemplateConfig]:
        return list(self._cache.values())

    @staticmethod
    def _parse_template_dict(data: dict) -> TemplateConfig:
        
        raw_anchors = data.get("anchors", {})
        raw_fields = data.get("fields", {})

        fields = {}
        for field_name, f_cfg in raw_fields.items():
            anchor_name = f_cfg.get("anchor")
            anchor_cfg = raw_anchors.get(anchor_name)
            
            if anchor_cfg and "expected" in anchor_cfg:
                # Calculate base ROI from anchor expected + offset
                exp = anchor_cfg["expected"]
                off = f_cfg.get("offset", {"dx": 0.0, "dy": 0.0})
                size = f_cfg.get("size", {"w": 0.1, "h": 0.05})
                
                fields[field_name] = RoiDef(
                    field=field_name,
                    x=exp["x"] + off.get("dx", 0.0),
                    y=exp["y"] + off.get("dy", 0.0),
                    width=size.get("w", 0.1),
                    height=size.get("h", 0.05),
                )
            elif "x" in f_cfg:
                # Fallback to old schema if explicit x, y are present
                fields[field_name] = RoiDef(
                    field=field_name,
                    x=f_cfg["x"],
                    y=f_cfg["y"],
                    width=f_cfg["width"],
                    height=f_cfg["height"],
                )

        return TemplateConfig(
            template_id=data.get("id", data.get("template_id")),
            document_type=data.get("document_type", "export_certificate"),
            page_size=data.get("page_size", "A4"),
            rotation=data.get("rotation", 0),
            anchors=raw_anchors,
            field_configs=raw_fields,
            fields=fields,
            match_criteria=data.get("match", {}),
        )


class TemplateDetectionService:
    """
    Detects the template for a rendered page image.
    Strategy: quick full-page OCR → anchor text match.
    Falls back to the highest-priority template if OCR is unavailable.
    """

    def __init__(self, loader: TemplateLoader) -> None:
        self._loader = loader

    def detect(self, image: Image.Image, hint: Optional[str] = None) -> Optional[TemplateConfig]:
        """
        Return the matching TemplateConfig or None if no match.
        If hint is provided, try that template first.
        """
        templates = self._loader.all()
        if not templates:
            return None

        # Honour caller hint (e.g. user-supplied document_type)
        if hint:
            for tpl in templates:
                if tpl.template_id == hint or tpl.document_type == hint:
                    return tpl

        # Anchor-text matching via lightweight OCR
        matched = self._match_by_anchor(image, templates)
        if matched:
            return matched

        # If only one template registered, default to it with a warning
        if len(templates) == 1:
            logger.warning(
                "No anchor match; defaulting to sole template: %s",
                templates[0].template_id,
            )
            return templates[0]

        return None

    # ------------------------------------------------------------------
    # Internal

    def _match_by_anchor(
        self, image: Image.Image, templates: list[TemplateConfig]
    ) -> Optional[TemplateConfig]:
        """
        Run PaddleOCR on the full page and look for anchor strings.
        Import is local to avoid startup cost when template hint is used.
        """
        try:
            from paddleocr import PaddleOCR  # type: ignore
            ocr = PaddleOCR(
                use_angle_cls=False,
                lang="japan",
            )
            import numpy as np
            result = ocr.ocr(np.array(image))
            texts: list[str] = []
            if result:
                page_res = result[0] if isinstance(result, list) and result else []
                if isinstance(page_res, list):
                    for line in page_res:
                        if isinstance(line, list) and len(line) >= 2:
                            content = line[1]
                            if isinstance(content, (list, tuple)) and len(content) >= 2:
                                texts.append(str(content[0]).upper())
            full_text = " ".join(texts)
        except Exception as exc:
            logger.warning("Anchor OCR failed: %s", exc)
            return None

        for tpl in templates:
            if self._anchors_match(full_text, tpl):
                logger.info("Template matched: %s", tpl.template_id)
                return tpl

        return None

    @staticmethod
    def _anchors_match(full_text: str, tpl: TemplateConfig) -> bool:
        """At least one anchor text from match_criteria or anchors must appear in the page OCR output."""
        from src.core.template.anchor_roi_service import AnchorRoiService
        norm_full = AnchorRoiService._normalize_text(full_text)
        logger.info("Matching anchors. Full text (norm): %s...", norm_full[:100])
        
        # 1. Check explicit match criteria
        match_texts = tpl.match_criteria.get("anchor_text", [])
        for text in match_texts:
            target = AnchorRoiService._normalize_text(text)
            if target in norm_full:
                logger.info("Matched by criteria: %s", text)
                return True
                
        # 2. Check individual field anchors as secondary signal
        for name, cfg in tpl.anchors.items():
            target_raw = cfg.get("text", "")
            if target_raw:
                target = AnchorRoiService._normalize_text(target_raw)
                if target in norm_full:
                    logger.info("Matched by field anchor: %s", name)
                    return True
                
        return False
