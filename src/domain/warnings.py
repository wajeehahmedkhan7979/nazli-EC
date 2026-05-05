"""
Standardised warning codes for the extraction pipeline.
Use these constants instead of free-text warning strings.
"""
from enum import Enum


class ExtractionWarning(str, Enum):
    DATE_INCONSISTENT = "DATE_INCONSISTENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    OCR_NOISE_DETECTED = "OCR_NOISE_DETECTED"
    ROI_EMPTY = "ROI_EMPTY"
    TEMPLATE_MISMATCH = "TEMPLATE_MISMATCH"
    FALLBACK_TRIGGERED = "FALLBACK_TRIGGERED"
    PARSE_FAILURE = "PARSE_FAILURE"
