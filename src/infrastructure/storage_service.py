"""
Result persistence layer.
"""
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from src.domain.types import ExtractionResult

class StorageService:
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = Path(base_dir)
        self.pdfs_dir = self.base_dir / "pdfs"
        self.results_dir = self.base_dir / "results"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_pdf(self, file_id: str, source_path: Path):
        dest = self.pdfs_dir / f"{file_id}.pdf"
        shutil.copy2(source_path, dest)

    def save_result_json(self, file_id: str, result_dict: dict):
        dest = self.results_dir / f"{file_id}.json"
        with open(dest, "w") as f:
            json.dump(result_dict, f, indent=2)

    def load_result(self, file_id: str) -> Optional[dict]:
        dest = self.results_dir / f"{file_id}.json"
        if not dest.exists():
            return None
        with open(dest, "r") as f:
            return json.load(f)

storage_service = StorageService()
