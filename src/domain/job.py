"""
Async Job System Models.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime

from src.domain.types import ExtractionResult

JobStatus = Literal["queued", "processing", "completed", "failed", "partial"]

@dataclass
class ExtractionJob:
    job_id: str
    status: JobStatus = "queued"
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    results: Optional[ExtractionResult] = None
    error: Optional[str] = None
