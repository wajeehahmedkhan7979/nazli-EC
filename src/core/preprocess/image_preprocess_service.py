"""
ImagePreprocessService — deskew, normalize, denoise for OCR preparation.
Conservative pipeline: avoid over-processing that harms faint text.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessService:
    """Prepares a PIL Image for ROI-based OCR."""

    def __init__(
        self,
        deskew_threshold_deg: float = 1.0,
        denoise_strength: int = 3,
        contrast_clip_limit: float = 2.0,
        contrast_tile_grid: int = 8,
    ) -> None:
        self._deskew_thresh = deskew_threshold_deg
        self._denoise_h = denoise_strength
        self._clip = contrast_clip_limit
        self._tile = contrast_tile_grid

    # ------------------------------------------------------------------
    # Public API

    def prepare_for_ocr(self, image: Image.Image) -> Image.Image:
        """Full preprocessing chain for a page image."""
        arr = self._to_gray(image)
        arr = self._normalize_contrast(arr)
        arr = self._denoise(arr)
        angle = self._detect_skew(arr)
        if abs(angle) > self._deskew_thresh:
            arr = self._rotate(arr, angle)
        return Image.fromarray(arr)

    def prepare_roi(self, roi_image: Image.Image) -> Image.Image:
        """Lighter processing for an already-cropped ROI."""
        arr = self._to_gray(roi_image)
        arr = self._normalize_contrast(arr)
        return Image.fromarray(arr)

    def deskew(self, image: Image.Image) -> tuple[Image.Image, float]:
        """Return deskewed image and detected angle."""
        arr = self._to_gray(image)
        angle = self._detect_skew(arr)
        if abs(angle) > self._deskew_thresh:
            arr = self._rotate(arr, angle)
        return Image.fromarray(arr), angle

    # ------------------------------------------------------------------
    # Internal

    @staticmethod
    def _to_gray(image: Image.Image) -> np.ndarray:
        if image.mode != "L":
            return np.array(image.convert("L"))
        return np.array(image)

    def _normalize_contrast(self, arr: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self._clip,
            tileGridSize=(self._tile, self._tile),
        )
        return clahe.apply(arr)

    def _denoise(self, arr: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(arr, h=self._denoise_h)

    def _detect_skew(self, arr: np.ndarray) -> float:
        """Estimate skew angle via Hough line transform (degrees)."""
        edges = cv2.Canny(arr, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return 0.0
        angles = []
        for line in lines[:30]:
            rho, theta = line[0]
            angle_deg = math.degrees(theta) - 90
            if abs(angle_deg) < 45:
                angles.append(angle_deg)
        return float(np.median(angles)) if angles else 0.0

    @staticmethod
    def _rotate(arr: np.ndarray, angle: float) -> np.ndarray:
        h, w = arr.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            arr, mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
