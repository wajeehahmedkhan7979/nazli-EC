# EC PDF Extractor v2

Extracts `chassis_no`, `issue_date`, `expiry_date` from all pages of a Japanese
MLIT Export Certificate PDF.

**Zero AI. Zero paid services. Runs entirely on your own machine.**

---

## How it works

```
Upload PDF → PyMuPDF renders each page → Fixed ROI crops → Tesseract OCR → Regex parse → JSON
```

No ML models, no cloud APIs. Each page takes ~0.5–1.5 s depending on your CPU.

---

## Prerequisites

### 1. Python 3.10+
### 2. Tesseract with Japanese language pack

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
and add it to your PATH. Install the `jpn` and `eng` data files.

---

## Installation

```bash
git clone https://github.com/wajeehahmedkhan7979/nazli-EC.git
cd nazli-EC
pip install -e .
```

---

## Running

```bash
# Copy and edit the env file
cp .env.example .env
# edit FORWARD_URL if you want results auto-sent to your ERP/frontend

python main.py
# → http://localhost:8080
```

---

## API Usage

### Extract a PDF

```bash
curl -X POST http://localhost:8080/v1/extract \
  -F "file=@/path/to/export_certificate.pdf"
```

**Response:**
```json
{
  "file_id": "3fa85f64-...",
  "file_name": "EC_4_30.pdf",
  "page_count": 51,
  "summary": {
    "total_pages": 51,
    "fully_extracted": 50,
    "partial": 1,
    "failed": 0,
    "success_rate": "98.0%"
  },
  "results": [
    {
      "page": 1,
      "chassis_no": "NKE165-7242932",
      "issue_date": "2026-03-27",
      "expiry_date": "2026-08-06",
      "warnings": [],
      "elapsed_ms": 820.3
    }
  ]
}
```

### Retrieve a previous result

```bash
curl http://localhost:8080/v1/results/{file_id}
```

### Health check

```bash
curl http://localhost:8080/health
```

---

## Forwarding to your ERP / frontend

Set `FORWARD_URL` in your `.env`:
```
FORWARD_URL=https://your-erp.com/api/ec-results
FORWARD_API_KEY=your-bearer-token
```

After every extraction the service will POST the full JSON payload to that URL.
Your frontend endpoint just needs to accept a POST with a JSON body.

---

## Docker

```bash
docker build -t ec-extractor .
docker run -p 8080:8080 \
  -e FORWARD_URL=https://your-erp.com/api/ec-results \
  -e FORWARD_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  ec-extractor
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## ROI Calibration (if layout changes)

If you receive a different form variant, edit the `ROIS` dict at the top of
`src/extractor.py`. Values are fractional (0.0–1.0 relative to page size):

```python
ROIS = {
    "chassis_no": dict(x=0.58, y=0.115, w=0.38, h=0.060),
    "issue_date":  dict(x=0.26, y=0.115, w=0.24, h=0.060),
    "expiry_date": dict(x=0.22, y=0.575, w=0.38, h=0.085),
}
```

To find new coordinates, run the probe script:
```bash
python scripts/probe_rois.py /path/to/new_form.pdf
```
