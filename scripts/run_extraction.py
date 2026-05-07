import os
import sys
import time

from src.core.orchestrator.extraction_orchestrator import ExtractionOrchestrator
from src.core.template.template_detection_service import TemplateLoader, TemplateDetectionService
from src.core.template.roi_mapping_service import RoiMappingService
from src.core.template.anchor_roi_service import AnchorRoiService
from src.core.ocr.ocr_service import OcrService
from src.core.ocr.ocr_cache import OcrCache
from src.core.renderer.pdf_render_service import PdfRenderService
from src.core.preprocess.image_preprocess_service import ImagePreprocessService
from src.core.output.metrics import metrics
from src.core.parsing.field_parsing_service import FieldParsingService

def main(pdf_path: str):
    print(f"Initializing pipeline...")
    
    renderer = PdfRenderService()
    preprocessor = ImagePreprocessService()
    
    loader = TemplateLoader(Path("configs/templates/export_certificate_v1.yaml"))
    detector = TemplateDetectionService(loader)
    
    anchor_service = AnchorRoiService()
    roi_mapper = RoiMappingService(anchor_service)
    
    ocr_service = OcrService()
    
    parser = FieldParsingService()
    
    orchestrator = ExtractionOrchestrator(
        renderer=renderer,
        preprocessor=preprocessor,
        template_detector=detector,
        roi_mapper=roi_mapper,
        ocr_service=ocr_service,
        field_parser=parser,
        fallback_trigger=0.80,
        accept_threshold=0.92
    )
    
    print(f"Processing PDF: {pdf_path}")
    t0 = time.time()
    
    result = orchestrator.process_document(pdf_path, file_id="test_doc", template_hint=None)
    
    print(f"Finished in {time.time() - t0:.2f}s")
    print(f"Status: {result.status}")
    print(f"Pages: {result.page_count}")
    
    print("\n## Extraction Results")
    print("| Page | Chassis No | Issue Date | Expiration Date | Confidence | Method | Warnings |")
    print("|---|---|---|---|---|---|---|")
    
    for r in result.results:
        conf = f"{r.confidence.chassis_no:.2f}/{r.confidence.issue_date:.2f}/{r.confidence.expiration_date:.2f}"
        warnings = ", ".join(r.warnings) if r.warnings else "-"
        print(f"| {r.page_number} | {r.chassis_no or '-'} | {r.issue_date or '-'} | {r.expiration_date or '-'} | {conf} | {r.method} | {warnings} |")
        
    print("\n## Metrics Snapshot")
    snap = metrics.snapshot()
    for k, v in snap.items():
        print(f"{k}: {v}")
        
    if "failed" in result.status or "partial" in result.status:
        print("\n[WARNING] Some pages failed or fell back. Review the table above.")

if __name__ == "__main__":
    from pathlib import Path
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/home/morrty00/Documents/Files/EC 4.30.pdf"
    main(pdf)
