"""
FastAPI application — POST /v1/extract and GET /health endpoints.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import yaml
import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from src.core.orchestrator.extraction_orchestrator import build_orchestrator
from src.core.output.json_builder import extraction_result_to_dict
from src.core.jobs.job_queue import job_queue
from src.core.output.metrics import metrics
from src.infrastructure.storage_service import storage_service

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "configs/thresholds.yaml")
_REGISTRY_PATH = os.getenv("REGISTRY_PATH", "configs/templates/template_registry.yaml")
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit
_MAX_PAGES = 100

with open(_THRESHOLDS_PATH) as _f:
    _THRESHOLDS = yaml.safe_load(_f)

_orchestrator = build_orchestrator(
    registry_path=_REGISTRY_PATH,
    thresholds=_THRESHOLDS,
)

app = FastAPI(
    title="EC PDF Field Extractor",
    version="1.0.0",
    description="Extracts chassis_no, issue_date, expiration_date from image-based PDFs.",
)

# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "ec-pdf-extractor", "version": "1.0.0"}

@app.get("/metrics")
def get_metrics():
    return JSONResponse(content=metrics.snapshot())

@app.get("/v1/extract/{job_id}")
def get_job(job_id: str):
    job = job_queue.get_job(job_id)
    if not job:
        result = storage_service.load_result(job_id)
        if result:
            return JSONResponse(content={"job_id": job_id, "status": result.get("status"), "results": result})
        raise HTTPException(status_code=404, detail="Job not found")
        
    response = {"job_id": job.job_id, "status": job.status, "progress": job.progress}
    if job.error:
        response["error"] = job.error
    if job.results:
        response["results"] = extraction_result_to_dict(job.results)
    return JSONResponse(content=response)


@app.post("/v1/extract")
async def extract(
    file: UploadFile = File(...),
    document_type: str = Form(default=None),
    template_hint: str = Form(default=None),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max 10MB.")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        if doc.is_encrypted:
            raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported.")
        if doc.page_count > _MAX_PAGES:
            raise HTTPException(status_code=400, detail=f"Too many pages. Max {_MAX_PAGES}.")
        doc.close()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    file_id = str(uuid.uuid4())
    tmp_path = Path(tempfile.gettempdir()) / f"{file_id}.pdf"
    with open(tmp_path, "wb") as tmp:
        tmp.write(content)

    storage_service.save_raw_pdf(file_id, tmp_path)

    job = job_queue.create_job(file_id)

    async def _extraction_task():
        try:
            result = _orchestrator.process_document(
                tmp_path,
                file_id=file_id,
                template_hint=template_hint or document_type,
            )
            job.results = result
            job.status = result.status
            storage_service.save_result_json(file_id, extraction_result_to_dict(result))
        except Exception as exc:
            logger.exception("Extraction failed for file_id=%s", file_id)
            job.status = "failed"
            job.error = str(exc)
        finally:
            tmp_path.unlink(missing_ok=True)

    job_queue.enqueue(file_id, _extraction_task)

    return JSONResponse(status_code=202, content={"job_id": file_id, "status": "queued", "message": "Extraction job submitted."})
