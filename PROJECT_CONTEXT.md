# Project: Japanese Export Certificate (EC) Extraction Pipeline

## Overview
The goal of this project is to build a production-grade, deterministic extraction pipeline for Japanese Export Certificate (EC) PDFs. The system uses a "Surgical ROI" approach: rendering specific regions of the PDF and running OCR only on those areas to ensure maximum accuracy and bypass document noise.

## Current Tech Stack
- **Languages:** Python 3.12+
- **PDF Engine:** `PyMuPDF` (fitz) for rendering and coordinate mapping.
- **OCR Engine:** `PaddleOCR` (PP-OCRv5) for text recognition.
- **Image Processing:** `OpenCV` and `Pillow` (PIL) for preprocessing and cropping.

## Journey & Key Milestones

### 1. Initial Setup
- Transitioned from full-page probabilistic OCR to a deterministic ROI-based strategy.
- Calibrated the system for A4 Landscape documents (842x595 PDF points).

### 2. The "Coordinate Confusion" Phase
- **Issue:** Using pixel coordinates from external tools (`roi_picker`) caused boxes to be drawn off-screen because the script expects native PDF points (1/72 inch).
- **Fix:** All coordinates in `ROIS` were converted back to native PDF points. The script handles the scaling to high-resolution pixels (300 DPI) internally.

### 3. PaddleOCR Stability War
This has been the main pain point. The environment (Linux on ARM/WSL/Standard) has specific incompatibilities with the latest Paddle features.
- **PIR Compiler Error:** `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` occurred when using the default detection models.
- **The "Surgical Fix":** We disabled the **Detection** stage (`det=False`) and the **Angle Classifier** (`cls=False`). Since we are already cropping the ROI, we only need the **Recognition** model.
- **Argument Errors:** Different versions of `PaddleOCR` on the system have inconsistent argument support. We've had to iteratively remove `show_log`, `det`, `rec`, and `use_gpu` from various calls to find the "safe" subset.

### 4. Visual Calibration
- The user provided manual annotations on a screenshot of `EC4.30.pdf`.
- **Chassis No:** Top-right area.
- **Expiry Date:** Middle-bottom (Validity Period).
- **Issue Date:** Bottom-left (Application Date).
- Coordinates have been refined to:
  - Chassis: `{"x1": 590, "y1": 95, "x2": 820, "y2": 145}`
  - Expiry: `{"x1": 210, "y1": 445, "x2": 465, "y2": 505}`
  - Issue: `{"x1": 40, "y1": 515, "x2": 340, "y2": 575}`

## Current Status (as of 2026-05-07)
- **Script:** `scripts/probe_rois.py` is the source of truth for verification.
- **Output:** Saves `probe_page1.png` and individual crops in `static/debug/`.
- **Last Error:** `ValueError: Unknown argument: use_gpu` during `PaddleOCR` initialization.

## Main Pain Points & Gotchas
1. **Scanned PDFs:** The target PDFs are scanned images. `page.get_text()` returns nothing; OCR is mandatory.
2. **Environment Sensitivity:** Paddle environment variables (FLAGS) are critical to prevent crashes.
3. **Coordinate Scaling:** Always work in 72 DPI "Points" for ROIs. Let the script scale them by `(Target_DPI / 72)`.
4. **Paddle versioning:** The installed version of `paddleocr` seems to be a wrapper around `paddlex` which has a different API signature than standard `paddleocr`.

## Future Work
- Hardening the `get_ocr()` initialization for the specific environment.
- Finalizing the parsing logic for dates and chassis numbers to handle OCR noise.
- Scaling the pipeline to handle multi-page documents and batches.
