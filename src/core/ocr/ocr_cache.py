"""
Lightweight in-memory cache for OCR results based on image hashing.
"""
import hashlib
from typing import Optional

from PIL import Image
from src.domain.types import OcrResult

class OcrCache:
    def __init__(self):
        self._cache: dict[str, OcrResult] = {}

    def _hash_image(self, image: Image.Image) -> str:
        # Simple perceptual/data hash for exact crops
        return hashlib.md5(image.tobytes()).hexdigest()

    def get(self, image: Image.Image) -> Optional[OcrResult]:
        h = self._hash_image(image)
        return self._cache.get(h)

    def set(self, image: Image.Image, result: OcrResult):
        h = self._hash_image(image)
        self._cache[h] = result

ocr_cache = OcrCache()
