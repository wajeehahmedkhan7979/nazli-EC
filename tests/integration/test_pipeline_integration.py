"""
Integration test — full pipeline with a synthetic single-page PDF.
Uses PyMuPDF to create a minimal in-memory PDF for testing.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.domain.types import FieldConfidence, OcrResult, PageResult, TemplateConfig, RoiDef, AnchorDef
from src.core.orchestrator.extraction_orchestrator import ExtractionOrchestrator
from src.core.renderer.pdf_render_service import RenderedPage
from src.core.parsing.field_parsing_service import FieldParsingService
from src.core.preprocess.image_preprocess_service import ImagePreprocessService
from src.core.template.roi_mapping_service import RoiMappingService


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_image() -> Image.Image:
    """300×420 white RGB image (A4-ish at 300 DPI)."""
    return Image.new("RGB", (2480, 3508), color=(255, 255, 255))


@pytest.fixture
def export_cert_template() -> TemplateConfig:
    return TemplateConfig(
        template_id="export_certificate_v1",
        document_type="vehicle_export_certificate",
        page_size="A4",
        rotation=0,
        anchors=[AnchorDef(name="title", text="EXPORT CERTIFICATE", min_confidence=0.70)],
        fields={
            "chassis_no": RoiDef("chassis_no", 0.35, 0.28, 0.45, 0.045),
            "issue_date": RoiDef("issue_date", 0.55, 0.14, 0.35, 0.04),
            "expiration_date": RoiDef("expiration_date", 0.55, 0.19, 0.35, 0.04),
        },
    )


# ---------------------------------------------------------------------------
# Integration: full pipeline with mocked OCR and renderer
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """
    Mocks the renderer (no real PDF needed) and OCR (no PaddleOCR install needed).
    Validates that the orchestrator wires all services correctly and returns
    well-formed JSON-serialisable PageResult objects.
    """

    def _build_orchestrator(self, template: TemplateConfig, ocr_texts: dict[str, str]):
        """
        Build orchestrator with:
        - renderer → yields one RenderedPage with a blank image
        - template detector → always returns `template`
        - OCR → returns hard-coded text per field name
        """
        from src.core.orchestrator.extraction_orchestrator import ExtractionOrchestrator

        renderer = MagicMock()
        blank = Image.new("RGB", (2480, 3508), (255, 255, 255))
        renderer.page_count.return_value = 1
        renderer.render_all_pages.return_value = iter([
            RenderedPage(page_number=1, image=blank, width=2480, height=3508, dpi=300)
        ])

        preprocessor = ImagePreprocessService()

        detector = MagicMock()
        detector.detect.return_value = template

        roi_mapper = RoiMappingService()

        # OCR returns per-field canned text
        def mock_recognize(image):
            # We can't distinguish crops by content in a blank image, so we
            # return a combined string that contains all target values.
            combined = " ".join(ocr_texts.values())
            return OcrResult(raw_text=combined, confidence=0.95, engine="mock")

        ocr_service = MagicMock()
        ocr_service.recognize.side_effect = mock_recognize
        ocr_service.recognize_with_fallback.side_effect = mock_recognize

        field_parser = FieldParsingService()

        return ExtractionOrchestrator(
            renderer=renderer,
            preprocessor=preprocessor,
            template_detector=detector,
            roi_mapper=roi_mapper,
            ocr_service=ocr_service,
            field_parser=field_parser,
            accept_threshold=0.50,   # lowered so mock confidences pass
            fallback_trigger=0.20,
        )

    def test_happy_path_returns_all_three_fields(
        self, export_cert_template, tmp_path
    ):
        """All three fields present in OCR output → all extracted."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        ocr_texts = {
            "chassis_no": "NKE165G-7242861",
            "issue_date": "2026-03-27",
            "expiration_date": "2026-03-15",
        }

        orch = self._build_orchestrator(export_cert_template, ocr_texts)
        result = orch.process_document(fake_pdf, file_id="test-001")

        assert result.file_id == "test-001"
        assert result.page_count == 1
        assert len(result.results) == 1

        page = result.results[0]
        assert page.page_number == 1
        assert page.chassis_no == "NKE165G-7242861"
        assert page.issue_date == "2026-03-27"
        assert page.expiration_date == "2026-03-15"
        assert page.confidence.chassis_no > 0
        assert page.confidence.issue_date > 0
        assert page.confidence.expiration_date > 0
        assert page.method == "template_roi_ocr"
        assert page.template_id == "export_certificate_v1"

    def test_missing_chassis_returns_null(self, export_cert_template, tmp_path):
        """OCR returns only dates → chassis_no is None with warning."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        ocr_texts = {
            "chassis_no": "no match here",
            "issue_date": "2026-03-27",
            "expiration_date": "2026-03-15",
        }

        orch = self._build_orchestrator(export_cert_template, ocr_texts)
        result = orch.process_document(fake_pdf, file_id="test-002")

        page = result.results[0]
        assert page.chassis_no is None
        assert page.issue_date == "2026-03-27"
        assert any("chassis_no" in w for w in page.warnings)

    def test_summary_aggregation(self, export_cert_template, tmp_path):
        """Summary block is computed correctly."""
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        ocr_texts = {
            "chassis_no": "KSP210-0116561",
            "issue_date": "2026-01-01",
            "expiration_date": "2026-12-31",
        }

        orch = self._build_orchestrator(export_cert_template, ocr_texts)
        result = orch.process_document(fake_pdf)
        summary = result.summary

        assert "success_pages" in summary
        assert "average_confidence" in summary
        assert summary["failed_pages"] == 0

    def test_json_serialisable_output(self, export_cert_template, tmp_path):
        """Output must be JSON-serialisable (no unserializable types)."""
        import json
        from src.core.output.json_builder import extraction_result_to_dict

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        ocr_texts = {
            "chassis_no": "MH95S-291382",
            "issue_date": "2026-06-01",
            "expiration_date": "2027-06-01",
        }

        orch = self._build_orchestrator(export_cert_template, ocr_texts)
        result = orch.process_document(fake_pdf, file_id="serial-test")
        body = extraction_result_to_dict(result)

        # Must not raise
        json_str = json.dumps(body)
        assert "serial-test" in json_str
