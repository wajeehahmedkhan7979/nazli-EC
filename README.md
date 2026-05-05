# EC PDF Field Extractor

This is a standalone Python service for extracting exactly three fields (`chassis_no`, `issue_date`, `expiration_date`) from image-based PDF "Export Certificate" (EC) documents.

## Architecture Highlights
- **ROI-First OCR:** Uses PyMuPDF to render pages to images, then crops target regions before running OCR. This minimizes noise and improves parsing accuracy.
- **Dual OCR Engines:** Primary is PaddleOCR (fast, local, good for Japanese/English mixed). Falls back to Tesseract if confidence is low.
- **Hardened Parsing:** Deterministic regex parsing for chassis numbers and multi-format date resolution (ISO, compact, Japanese era).
- **Cross-Field Validation:** Validates `expiration_date >= issue_date`.
- **Anchor-Based ROI Support:** Can dynamically shift static ROI crops based on OCR-detected anchor texts to handle scan drift.

## Setup

Requires Python 3.10+.

### System Dependencies
You will need Tesseract and its Japanese language pack, as well as OpenCV dependencies:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-jpn libgl1-mesa-glx
```

### Python Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running the API

```bash
python main.py
```
This starts a FastAPI server on `http://0.0.0.0:8080`.

## API Usage

**Endpoint:** `POST /v1/extract`
**Content-Type:** `multipart/form-data`

```bash
curl -X POST http://localhost:8080/v1/extract \
  -F "file=@/path/to/your/document.pdf" \
  -F "template_hint=export_certificate_v1"
```

The response is a JSON object summarizing the extraction results per page, including confidence scores and any validation warnings.

## Configuration

- **Thresholds & OCR config:** `configs/thresholds.yaml`
- **Template definitions (ROI coordinates):** `configs/templates/`

### Template Probe Utility
To calibrate ROI coordinates for a new or modified template, run the interactive probe:
```bash
python scripts/template_probe.py /path/to/sample.pdf 1
```
This lets you draw bounding boxes and prints out the fractional `(x, y, width, height)` values needed for the template YAML files.

## Testing

Run the full test suite (unit and mocked integration tests):
```bash
pytest tests/ -v
```
