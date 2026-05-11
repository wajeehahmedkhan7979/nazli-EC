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
- **Issue:** Using pixel coordinates from screenshots/external tools caused systemic drift.
- **Fix:** Switched to **Native PDF Points (1/72")**. By defining ROIs in the 842x595 point space, the script now "snaps" to the correct locations regardless of the render resolution (DPI).

### 3. PaddleOCR Stability War (Final Resolution)
- **Problem:** Frequent `TypeError` and `ValueError` crashes due to "unexpected keyword arguments" (`det`, `cls`, `show_log`) across different PaddleOCR wrapper versions.
- **The "Bulletproof" Solution:** Refactored `_run_ocr` to use the most basic call `ocr.ocr(arr)` with a defensive result parser. 
- **Environment Flags:** Fixed C++/OneDNN runtime crashes using:
  `FLAGS_enable_pir_api=0 FLAGS_use_onednn=0`

### 4. Visual Arrow Calibration (Final Mapping)
Based on direct visual feedback and "Visual Arrow" annotations, we identified that the target fields live in different "elevations" than initially thought:
- **Chassis No (Red):** Top-right row.
- **Expiry Date (Green):** Middle row (Export Scheduled Day).
- **Issue Date (Blue):** Bottom-left stamp, narrowly cropped to avoid QR codes.

#### Final Calibrated Coordinates (842x595 Points):
- **Chassis:** `{"x1": 420, "y1": 180, "x2": 660, "y2": 225}`
- **Expiry:** `{"x1": 60, "y1": 435, "x2": 450, "y2": 485}`
- **Issue:** `{"x1": 80, "y1": 530, "x2": 320, "y2": 580}`

## Current Status (as of 2026-05-08)
- **Stable:** `src/extractor.py` has a hardened `_run_ocr` and `_preprocess` pipeline.
- **Verified:** Calibration is confirmed via `scripts/probe_rois.py`.
- **DPI:** Standardized at `300 DPI` for OCR quality.

## Main Pain Points & Gotchas
1. **API Drift:** PaddleOCR's Python wrapper is extremely volatile. Never rely on non-essential arguments like `det=False` or `show_log=False` as they break across minor version updates.
2. **Scaling Logic:** Always render at `fitz.Matrix(DPI / 72, DPI / 72)` to ensure ROI points match pixel locations perfectly.
3. **QR Interference:** Keep the Issue Date ROI narrow; modern ECs have QR codes in the bottom-left corner that can confuse OCR if included in the crop.

## Future Work
- Integrate the verified ROIs into the main ERP processing loop.
- Implement post-processing regex to clean common OCR substitutions (e.g., '0' for 'O', '1' for 'I').
- Add human-in-the-loop triggers for low-confidence Chassis extractions.
