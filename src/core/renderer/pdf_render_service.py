"""
PdfRenderService — renders PDF pages to PIL Images via PyMuPDF.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RenderedPage:
    page_number: int   # 1-indexed
    image: Image.Image
    width: int
    height: int
    dpi: int


class PdfRenderService:
    """Renders PDF pages to PIL Images at a fixed DPI."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi
        self._zoom = dpi / 72.0  # PyMuPDF native unit is 72 DPI

    # ------------------------------------------------------------------

    def render_page(self, pdf_path: str | Path, page_number: int) -> RenderedPage:
        """
        Render a single page (1-indexed) to a PIL Image.
        Raises ValueError on bad page number, RuntimeError on render failure.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        with fitz.open(str(path)) as doc:
            total = len(doc)
            if not (1 <= page_number <= total):
                raise ValueError(
                    f"Page {page_number} out of range (1–{total})"
                )
            page = doc[page_number - 1]
            return self._render_fitz_page(page, page_number)

    def render_all_pages(
        self, pdf_path: str | Path
    ) -> Generator[RenderedPage, None, None]:
        """Yield RenderedPage for every page in the PDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        with fitz.open(str(path)) as doc:
            total = len(doc)
            logger.info("Rendering %d pages from %s at %d DPI", total, path.name, self.dpi)
            for idx, page in enumerate(doc):
                try:
                    yield self._render_fitz_page(page, idx + 1)
                except Exception as exc:
                    logger.error(
                        "Failed to render page %d: %s", idx + 1, exc
                    )
                    raise RuntimeError(
                        f"Render failure on page {idx + 1}"
                    ) from exc

    def page_count(self, pdf_path: str | Path) -> int:
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)

    # ------------------------------------------------------------------
    # Internal

    def _render_fitz_page(self, page: fitz.Page, page_number: int) -> RenderedPage:
        mat = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return RenderedPage(
            page_number=page_number,
            image=img,
            width=pix.width,
            height=pix.height,
            dpi=self.dpi,
        )
