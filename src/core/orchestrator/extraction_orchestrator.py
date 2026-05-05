"""
ExtractionOrchestrator — coordinates the full pipeline per page and document.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from src.core.ocr.ocr_service import OcrService
from src.core.parsing.field_parsing_service import FieldParsingService
from src.core.preprocess.image_preprocess_service import ImagePreprocessService
from src.core.renderer.pdf_render_service import PdfRenderService
from src.core.template.roi_mapping_service import RoiMappingService, TARGET_FIELDS
from src.core.template.template_detection_service import TemplateDetectionService, TemplateLoader
from src.core.parsing.field_parsing_service import is_ocr_noise
from src.core.output.metrics import metrics
from src.core.output.structured_logger import log_page_result
from src.domain.types import (
    ExtractionResult,
    FieldConfidence,
    OcrResult,
    PageResult,
    TemplateConfig,
)
from src.domain.warnings import ExtractionWarning

logger = logging.getLogger(__name__)

# Confidence threshold below which a field value is nulled out
_ACCEPT_THRESHOLD = 0.90
_FALLBACK_TRIGGER = 0.80   # STEP 5: raised from 0.60 — trigger earlier


class ExtractionOrchestrator:
    """Drives the full pipeline: PDF → per-page JSON."""

    def __init__(
        self,
        renderer: PdfRenderService,
        preprocessor: ImagePreprocessService,
        template_detector: TemplateDetectionService,
        roi_mapper: RoiMappingService,
        ocr_service: OcrService,
        field_parser: FieldParsingService,
        accept_threshold: float = _ACCEPT_THRESHOLD,
        fallback_trigger: float = _FALLBACK_TRIGGER,
    ) -> None:
        self._renderer = renderer
        self._preproc = preprocessor
        self._detector = template_detector
        self._roi_mapper = roi_mapper
        self._ocr = ocr_service
        self._parser = field_parser
        self._accept = accept_threshold
        self._fallback_trigger = fallback_trigger

    # ------------------------------------------------------------------
    # Public API

    def process_document(
        self,
        pdf_path: str | Path,
        file_id: Optional[str] = None,
        template_hint: Optional[str] = None,
    ) -> ExtractionResult:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        path = Path(pdf_path)
        file_id = file_id or str(uuid.uuid4())
        page_results: list[PageResult] = []

        page_count = self._renderer.page_count(path)
        logger.info("[%s] Processing %d pages from %s", file_id, page_count, path.name)

        futures = []
        with ThreadPoolExecutor(max_workers=1) as executor:  # Set to 1 for stability; PaddleOCR is not thread-safe.
            for rendered in self._renderer.render_all_pages(path):
                f = executor.submit(
                    self._process_page_with_retry,
                    rendered.image,
                    rendered.page_number,
                    file_id,
                    template_hint,
                )
                futures.append((rendered.page_number, f))

            for page_number, future in futures:
                try:
                    # Large timeout for the future itself; per-page watchdog is 30s inside the task.
                    result = future.result(timeout=600.0)
                except TimeoutError:
                    logger.error("[%s] p%d: Timeout exceeded", file_id, page_number)
                    result = self._failed_page(page_number, "timeout_exceeded", time.monotonic())
                except Exception as exc:
                    result = self._failed_page(page_number, str(exc), time.monotonic())

                page_results.append(result)

        failed = sum(1 for p in page_results if p.method == "failed")
        status = (
            "completed" if failed == 0
            else "partial" if failed < page_count
            else "failed"
        )

        return ExtractionResult(
            file_id=file_id,
            source_path=str(path),
            status=status,
            page_count=page_count,
            results=page_results,
        )

    # ------------------------------------------------------------------
    # Per-page pipeline

    def _process_page_with_retry(self, image, page_number, file_id, template_hint) -> PageResult:
        for attempt in range(2):
            t_start = time.monotonic()
            try:
                result = self._process_page(image, page_number, file_id, template_hint)
                if result.elapsed_ms > 180000:
                    logger.error("[%s] p%d timeout exceeded 180s budget", file_id, page_number)
                    result = self._failed_page(page_number, "timeout_exceeded", t_start)
                
                if result.method == "failed" and attempt == 0:
                    continue  # Retry on failure

                # Log to structured logger
                avg_conf = (result.confidence.chassis_no + result.confidence.issue_date + result.confidence.expiration_date) / 3.0
                log_page_result(
                    file_id=file_id,
                    page_number=page_number,
                    template_id=result.template_id,
                    field="all",
                    ocr_conf=avg_conf,  # approx
                    final_conf=avg_conf,
                    fallback_used=result.method == "fallback_full_page_ocr",
                    latency_ms=result.elapsed_ms,
                    warnings=result.warnings + result.errors
                )
                
                # Update metrics
                metrics.record_page(
                    latency_ms=result.elapsed_ms,
                    fallback=result.method == "fallback_full_page_ocr",
                    low_conf=avg_conf < self._accept,
                    mismatch=result.template_id is None
                )

                return result
            except Exception as e:
                if attempt == 0:
                    logger.warning("[%s] p%d attempt 1 failed: %s. Retrying...", file_id, page_number, e)
                    continue
                return self._failed_page(page_number, f"error: {e}", t_start)
        return self._failed_page(page_number, "max_retries_exceeded", time.monotonic())

    # ------------------------------------------------------------------
    # Per-page pipeline

    def _process_page(
        self,
        raw_image,
        page_number: int,
        file_id: str,
        template_hint: Optional[str],
    ) -> PageResult:
        t0 = time.monotonic()

        # 1. Preprocess
        try:
            page_image = self._preproc.prepare_for_ocr(raw_image)
        except Exception as exc:
            logger.error("[%s] Preprocess error p%d: %s", file_id, page_number, exc)
            return self._failed_page(page_number, str(exc), t0)

        # 2. Template detection
        template = self._detector.detect(page_image, hint=template_hint)
        if template is None:
            logger.warning("[%s] p%d: no template match", file_id, page_number)
            return self._run_fallback(page_image, page_number, t0, "no_template_match")

        # 3. ROI extraction (primary path)
        result = self._run_roi_extraction(page_image, page_number, template, t0)

        # 4. Fallback if any field confidence is too low
        needs_fallback = self._any_below_fallback(result)
        if needs_fallback:
            logger.info("[%s] p%d: triggering fallback", file_id, page_number)
            fallback_result = self._run_fallback(
                page_image, page_number, t0, "low_confidence_roi",
                template=template,
            )
            result = self._merge_best(result, fallback_result)

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    def _run_roi_extraction(
        self,
        page_image,
        page_number: int,
        template: TemplateConfig,
        t0: float,
        expand: bool = False,
    ) -> PageResult:
        warnings: list[str] = []

        try:
            crops, drift_warnings = self._roi_mapper.crop_all_fields(page_image, template, expand=expand)
            warnings.extend(drift_warnings)
        except Exception as exc:
            logger.error("ROI crop error p%d: %s", page_number, exc)
            return self._failed_page(page_number, str(exc), t0)

        # OCR each crop
        ocr_results: dict[str, OcrResult] = {}
        for field_name in TARGET_FIELDS:
            crop = crops.get(field_name)
            if crop is None:
                warnings.append(ExtractionWarning.ROI_EMPTY)
                ocr_results[field_name] = OcrResult("", 0.0, "paddleocr")
                continue
            roi_img = self._preproc.prepare_roi(crop)
            ocr_results[field_name] = self._ocr.recognize_with_fallback(
                roi_img, threshold=self._fallback_trigger
            )

        # Parse fields
        chassis_f = self._parser.parse_chassis_no(ocr_results["chassis_no"])
        issue_f = self._parser.parse_date(ocr_results["issue_date"], "issue_date")
        exp_f = self._parser.parse_date(ocr_results["expiration_date"], "expiration_date")

        warnings += chassis_f.warnings + issue_f.warnings + exp_f.warnings

        # Null out low-confidence values
        chassis_val = chassis_f.value if chassis_f.combined_confidence >= self._accept else None
        issue_val = issue_f.value if issue_f.combined_confidence >= self._accept else None
        exp_val = exp_f.value if exp_f.combined_confidence >= self._accept else None

        if chassis_f.value and chassis_val is None:
            warnings.append(ExtractionWarning.LOW_CONFIDENCE)
        if issue_f.value and issue_val is None:
            warnings.append(ExtractionWarning.LOW_CONFIDENCE)
        if exp_f.value and exp_val is None:
            warnings.append(ExtractionWarning.LOW_CONFIDENCE)

        # STEP 1 — Cross-field date validation
        exp_conf = exp_f.combined_confidence
        if issue_val and exp_val and exp_val < issue_val:
            warnings.append(ExtractionWarning.DATE_INCONSISTENT)
            exp_conf = round(exp_conf * 0.5, 4)  # penalise but don't discard
            logger.warning(
                "DATE_INCONSISTENT p%d: issue=%s exp=%s",
                page_number, issue_val, exp_val,
            )

        return PageResult(
            page_number=page_number,
            chassis_no=chassis_val,
            issue_date=issue_val,
            expiration_date=exp_val,
            confidence=FieldConfidence(
                chassis_no=chassis_f.combined_confidence,
                issue_date=issue_f.combined_confidence,
                expiration_date=exp_conf,
            ),
            method="template_roi_ocr",
            template_id=template.template_id,
            roi_map_version=template.roi_map_version,
            warnings=warnings,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Fallback path

    def _run_fallback(
        self,
        page_image,
        page_number: int,
        t0: float,
        reason: str,
        template: Optional[TemplateConfig] = None,
    ) -> PageResult:
        """
        Fallback sequence:
        1. Expanded ROI (if template available)
        2. Full-page OCR
        """
        warnings = [ExtractionWarning.FALLBACK_TRIGGERED]

        # Step 1 — expanded ROI
        if template is not None:
            expanded = self._run_roi_extraction(
                page_image, page_number, template, t0, expand=True
            )
            expanded.method = "fallback_full_page_ocr"
            expanded.warnings = warnings + expanded.warnings
            # If we got at least one field, return this
            if expanded.chassis_no or expanded.issue_date or expanded.expiration_date:
                expanded.elapsed_ms = (time.monotonic() - t0) * 1000
                return expanded

        # Step 2 — full-page OCR
        try:
            full_ocr = self._ocr.recognize(page_image)
        except Exception as exc:
            warnings.append(f"full-page OCR error: {exc}")
            return self._failed_page(page_number, "; ".join(warnings), t0)

        chassis_f = self._parser.parse_chassis_no(full_ocr)
        issue_f = self._parser.parse_date(full_ocr, "issue_date")
        exp_f = self._parser.parse_date(full_ocr, "expiration_date")
        warnings += chassis_f.warnings + issue_f.warnings + exp_f.warnings

        return PageResult(
            page_number=page_number,
            chassis_no=chassis_f.value,
            issue_date=issue_f.value,
            expiration_date=exp_f.value,
            confidence=FieldConfidence(
                chassis_no=chassis_f.combined_confidence,
                issue_date=issue_f.combined_confidence,
                expiration_date=exp_f.combined_confidence,
            ),
            method="fallback_full_page_ocr",
            template_id=template.template_id if template else None,
            roi_map_version=template.roi_map_version if template else None,
            warnings=warnings,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Helpers

    def _any_below_fallback(self, result: PageResult) -> bool:
        """True if any field confidence is below threshold OR noise was detected."""
        confs = [
            result.confidence.chassis_no,
            result.confidence.issue_date,
            result.confidence.expiration_date,
        ]
        if any(c < self._fallback_trigger for c in confs):
            return True
        # Also trigger if noise warning is present
        return ExtractionWarning.OCR_NOISE_DETECTED in result.warnings

    @staticmethod
    def _merge_best(primary: PageResult, fallback: PageResult) -> PageResult:
        """Take best value per field from primary vs fallback."""
        def pick(p_val, p_conf, f_val, f_conf):
            if p_val is not None and p_conf >= f_conf:
                return p_val, p_conf
            if f_val is not None:
                return f_val, f_conf
            return p_val, p_conf

        chassis, ch_conf = pick(
            primary.chassis_no, primary.confidence.chassis_no,
            fallback.chassis_no, fallback.confidence.chassis_no,
        )
        issue, is_conf = pick(
            primary.issue_date, primary.confidence.issue_date,
            fallback.issue_date, fallback.confidence.issue_date,
        )
        exp, ex_conf = pick(
            primary.expiration_date, primary.confidence.expiration_date,
            fallback.expiration_date, fallback.confidence.expiration_date,
        )

        primary.chassis_no = chassis
        primary.issue_date = issue
        primary.expiration_date = exp
        primary.confidence = FieldConfidence(ch_conf, is_conf, ex_conf)
        primary.warnings += fallback.warnings
        if fallback.method == "fallback_full_page_ocr":
            primary.method = "fallback_full_page_ocr"
        return primary

    @staticmethod
    def _failed_page(page_number: int, error: str, t0: float) -> PageResult:
        return PageResult(
            page_number=page_number,
            chassis_no=None,
            issue_date=None,
            expiration_date=None,
            confidence=FieldConfidence(),
            method="failed",
            template_id=None,
            roi_map_version=None,
            errors=[error],
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# Factory helper

def build_orchestrator(
    registry_path: str = "configs/templates/template_registry.yaml",
    thresholds: Optional[dict] = None,
) -> ExtractionOrchestrator:
    """Build a fully wired orchestrator from config paths."""
    th = thresholds or {}
    renderer = PdfRenderService(dpi=th.get("render", {}).get("dpi", 300))
    preprocessor = ImagePreprocessService(
        deskew_threshold_deg=th.get("preprocessing", {}).get("deskew_threshold_deg", 1.0),
        denoise_strength=th.get("preprocessing", {}).get("denoise_strength", 3),
        contrast_clip_limit=th.get("preprocessing", {}).get("contrast_clip_limit", 2.0),
        contrast_tile_grid=th.get("preprocessing", {}).get("contrast_tile_grid", 8),
    )
    loader = TemplateLoader(registry_path)
    detector = TemplateDetectionService(loader)
    roi_mapper = RoiMappingService()
    ocr_service = OcrService(
        primary_engine=th.get("ocr", {}).get("primary_engine", "paddleocr"),
        fallback_engine=th.get("ocr", {}).get("fallback_engine", "tesseract"),
        paddle_lang=th.get("ocr", {}).get("paddleocr_lang", "japan"),
        tess_lang=th.get("ocr", {}).get("tesseract_lang", "jpn+eng"),
        tess_config=th.get("ocr", {}).get("tesseract_config", "--oem 1 --psm 7"),
    )
    field_parser = FieldParsingService()
    conf = th.get("confidence", {})
    return ExtractionOrchestrator(
        renderer=renderer,
        preprocessor=preprocessor,
        template_detector=detector,
        roi_mapper=roi_mapper,
        ocr_service=ocr_service,
        field_parser=field_parser,
        accept_threshold=conf.get("accept", 0.90),
        fallback_trigger=conf.get("fallback_trigger", 0.80),
    )
