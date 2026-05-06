"""
EC Extractor API

POST /v1/extract
  - Accepts multipart PDF upload
  - Runs local extraction (no AI, no paid services)
  - Returns JSON
  - Optionally forwards results to ERP/frontend via FORWARD_URL env var

GET /health
GET /v1/results/{file_id}   — retrieve a previous result from disk cache
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.extractor import extract_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
FORWARD_URL       = os.getenv("FORWARD_URL", "")          # e.g. https://your-erp.com/api/ec-results
FORWARD_API_KEY   = os.getenv("FORWARD_API_KEY", "")      # Bearer token for the ERP
RESULTS_DIR       = Path(os.getenv("RESULTS_DIR", "data/results"))
MAX_UPLOAD_BYTES  = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_PAGES         = int(os.getenv("MAX_PAGES", "200"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EC PDF Extractor",
    version="2.0.0",
    description="Extracts chassis_no, issue_date, expiry_date from EC PDFs. No AI, no paid services.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your ERP domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ec-extractor", "version": "2.0.0"}


@app.post("/v1/extract")
async def extract(file: UploadFile = File(...)):
    """
    Upload a PDF → receive JSON with all extracted records.

    If FORWARD_URL is set, the result is also POSTed to that URL
    so your ERP/frontend receives it automatically.
    """
    # --- Validate ---
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(400, "Not a valid PDF.")

    # Quick page-count check via fitz before writing to disk
    import fitz
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception:
        raise HTTPException(400, "Could not open PDF.")

    if page_count > MAX_PAGES:
        raise HTTPException(400, f"PDF has {page_count} pages; max is {MAX_PAGES}.")

    # --- Save temp file ---
    file_id  = str(uuid.uuid4())
    tmp_path = Path(tempfile.gettempdir()) / f"ec_{file_id}.pdf"
    try:
        tmp_path.write_bytes(content)

        # --- Extract ---
        extraction = extract_pdf(tmp_path)
        result_dict = extraction.as_dict

        # --- Cache result to disk ---
        cache_path = RESULTS_DIR / f"{file_id}.json"
        cache_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2))

        # --- Forward to ERP/frontend if configured ---
        forward_status = None
        if FORWARD_URL:
            forward_status = await _forward_result(file_id, result_dict)

        response = {
            "file_id":   file_id,
            "file_name": extraction.file_name,
            "page_count": extraction.page_count,
            "results":   result_dict["results"],
            "summary":   _summary(result_dict["results"]),
        }
        if forward_status is not None:
            response["forwarded"] = forward_status

        return JSONResponse(response)

    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/v1/results/{file_id}")
def get_result(file_id: str):
    """Retrieve a previously extracted result from the local disk cache."""
    # Sanitise: only allow UUID-shaped IDs
    import re
    if not re.fullmatch(r"[0-9a-f\-]{36}", file_id):
        raise HTTPException(400, "Invalid file_id format.")
    cache_path = RESULTS_DIR / f"{file_id}.json"
    if not cache_path.exists():
        raise HTTPException(404, "Result not found.")
    return JSONResponse(json.loads(cache_path.read_text()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _forward_result(file_id: str, payload: dict) -> dict:
    """POST the extraction result to the configured ERP/frontend URL."""
    headers = {"Content-Type": "application/json"}
    if FORWARD_API_KEY:
        headers["Authorization"] = f"Bearer {FORWARD_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(FORWARD_URL, json={"file_id": file_id, **payload}, headers=headers)
        logger.info("Forwarded to %s → HTTP %s", FORWARD_URL, resp.status_code)
        return {"url": FORWARD_URL, "status_code": resp.status_code, "ok": resp.is_success}
    except Exception as exc:
        logger.warning("Forward failed: %s", exc)
        return {"url": FORWARD_URL, "error": str(exc), "ok": False}


def _summary(results: list[dict]) -> dict:
    total = len(results)
    success = sum(
        1 for r in results
        if r["chassis_no"] and r["issue_date"] and r["expiry_date"]
    )
    partial = sum(
        1 for r in results
        if any([r["chassis_no"], r["issue_date"], r["expiry_date"]])
        and not (r["chassis_no"] and r["issue_date"] and r["expiry_date"])
    )
    failed = total - success - partial
    return {
        "total_pages":    total,
        "fully_extracted": success,
        "partial":        partial,
        "failed":         failed,
        "success_rate":   f"{round(success / total * 100, 1)}%" if total else "0%",
    }
