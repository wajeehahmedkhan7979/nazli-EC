"""
OcrService — PaddleOCR primary, Tesseract fallback.
Returns OcrResult with raw text + confidence.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from src.domain.types import OcrResult
from src.core.ocr.ocr_cache import ocr_cache

logger = logging.getLogger(__name__)


class OcrService:
    """
    Primary: PaddleOCR (PP-OCRv5, local, multilingual).
    Fallback: Tesseract (pytesseract).
    """

    def __init__(
        self,
        primary_engine: str = "paddleocr",
        fallback_engine: str = "tesseract",
        paddle_lang: str = "japan",
        tess_lang: str = "jpn+eng",
        tess_config: str = "--oem 1 --psm 7",
    ) -> None:
        self._primary = primary_engine
        self._fallback = fallback_engine
        self._paddle_lang = paddle_lang
        self._tess_lang = tess_lang
        self._tess_config = tess_config
        self._paddle_ocr: Optional[object] = None  # lazy init

    # ------------------------------------------------------------------
    # Public API

    def recognize(self, image: Image.Image) -> OcrResult:
        """Run primary OCR engine. Does NOT trigger fallback."""
        cached = ocr_cache.get(image)
        if cached:
            return cached

        if self._primary == "paddleocr":
            result = self._paddle_recognize(image)
        else:
            result = self._tess_recognize(image)
            
        ocr_cache.set(image, result)
        return result

    def recognize_with_fallback(
        self, image: Image.Image, threshold: float = 0.60
    ) -> OcrResult:
        """
        Run primary OCR. If confidence < threshold, retry with fallback engine.
        Returns the higher-confidence result.
        """
        primary_result = self.recognize(image)
        if primary_result.confidence >= threshold:
            return primary_result

        logger.info(
            "Primary OCR confidence %.2f below %.2f — trying fallback engine",
            primary_result.confidence,
            threshold,
        )
        try:
            fallback_result = self._tess_recognize(image)
        except Exception as exc:
            logger.warning("Fallback OCR failed: %s", exc)
            return primary_result

        # Return whichever result has higher confidence and non-empty text
        if (
            fallback_result.confidence > primary_result.confidence
            and fallback_result.raw_text.strip()
        ):
            return fallback_result
        return primary_result

    # ------------------------------------------------------------------
    # Internal — PaddleOCR

    def _get_paddle(self):
        if self._paddle_ocr is None:
            from paddleocr import PaddleOCR  # type: ignore
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang=self._paddle_lang,
            )
        return self._paddle_ocr

    def _paddle_recognize(self, image: Image.Image) -> OcrResult:
        try:
            ocr = self._get_paddle()
            result = ocr.ocr(np.array(image))

            texts: list[str] = []
            confs: list[float] = []

            if result:
                # PaddleOCR typically returns a list of lists: [[[[box], [text, conf]], ...]]
                page_res = result[0] if isinstance(result, list) and result else []
                if isinstance(page_res, list):
                    for line in page_res:
                        if isinstance(line, list) and len(line) >= 2:
                             content = line[1]
                             if isinstance(content, (list, tuple)) and len(content) >= 2:
                                 text, conf = str(content[0]), float(content[1])
                                 texts.append(text)
                                 confs.append(conf)

            raw_text = " ".join(texts).strip()
            confidence = float(np.mean(confs)) if confs else 0.0

            return OcrResult(
                raw_text=raw_text,
                confidence=confidence,
                engine="paddleocr",
            )
        except Exception as exc:
            logger.error("PaddleOCR error: %s", exc)
            return OcrResult(raw_text="", confidence=0.0, engine="paddleocr")

    # ------------------------------------------------------------------
    # Internal — Tesseract

    def _tess_recognize(self, image: Image.Image) -> OcrResult:
        try:
            import pytesseract  # type: ignore

            raw_text = pytesseract.image_to_string(
                image, lang=self._tess_lang, config=self._tess_config
            ).strip()

            # Tesseract doesn't give per-character confidence via image_to_string;
            # use image_to_data for word-level confidences.
            data = pytesseract.image_to_data(
                image,
                lang=self._tess_lang,
                config=self._tess_config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [
                c / 100.0
                for c in data["conf"]
                if isinstance(c, (int, float)) and c >= 0
            ]
            confidence = float(np.mean(confs)) if confs else 0.0

            return OcrResult(
                raw_text=raw_text,
                confidence=confidence,
                engine="tesseract",
            )
        except Exception as exc:
            logger.error("Tesseract error: %s", exc)
            return OcrResult(raw_text="", confidence=0.0, engine="tesseract")
