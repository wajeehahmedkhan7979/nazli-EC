"""
Minimal Async Job Queue.
Uses asyncio.create_task for background execution. In a real distributed setup,
this would be replaced by Redis/BullMQ.
"""
import asyncio
import logging
from typing import Callable, Coroutine, Dict
from src.domain.job import ExtractionJob

logger = logging.getLogger(__name__)

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, ExtractionJob] = {}

    def create_job(self, job_id: str) -> ExtractionJob:
        job = ExtractionJob(job_id=job_id)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> ExtractionJob | None:
        return self._jobs.get(job_id)

    def enqueue(self, job_id: str, task_func: Callable[[], Coroutine]):
        job = self.get_job(job_id)
        if not job:
            return

        async def worker():
            job.status = "processing"
            try:
                await task_func()
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                job.status = "failed"
                job.error = str(e)

        asyncio.create_task(worker())

job_queue = JobQueue()
