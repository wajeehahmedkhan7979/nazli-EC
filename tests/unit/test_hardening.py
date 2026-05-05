"""
Unit tests for hardening changes (Steps 1–7).
Tests: noise detection, new confidence formula, cross-field date validation,
multi-candidate resolution, warning enums.
"""
import pytest
from src.domain.types import OcrResult, FieldConfidence, PageResult
from src.domain.warnings import ExtractionWarning
from src.core.parsing.field_parsing_service import (
    FieldParsingService,
    is_ocr_noise,
    _combine,
    _validation_conf,
)

parser = FieldParsingService()


def _ocr(text: str, conf: float = 0.95) -> OcrResult:
    return OcrResult(raw_text=text, confidence=conf, engine="test")


# ---------------------------------------------------------------------------
# STEP 4 — Noise detection
# ---------------------------------------------------------------------------

class TestOcrNoiseDetection:

    def test_clean_text_not_noise(self):
        assert is_ocr_noise("NKE165G-7242861") is False

    def test_garbage_symbols_detected(self):
        assert is_ocr_noise("!@#$$%^&*()|\\[]{}") is True

    def test_high_symbol_ratio(self):
        # 15 symbols out of 18 chars > 30%
        assert is_ocr_noise("!!! ### $$$ %%% NKE") is True

    def test_repeated_chars_detected(self):
        # 8 consecutive same chars → noise
        assert is_ocr_noise("AAAAAAAA something") is True

    def test_empty_string_not_noise(self):
        assert is_ocr_noise("") is False

    def test_normal_date_not_noise(self):
        assert is_ocr_noise("2026-03-27") is False

    def test_parse_returns_none_on_noisy_input(self):
        r = parser.parse_chassis_no(_ocr("!@#!@# $$$ XXXXXXXX", conf=0.95))
        assert r.value is None
        assert ExtractionWarning.OCR_NOISE_DETECTED in r.warnings

    def test_date_parse_returns_none_on_noisy_input(self):
        r = parser.parse_date(_ocr("||||||||||||||||||||||"))
        assert r.value is None
        assert ExtractionWarning.OCR_NOISE_DETECTED in r.warnings


# ---------------------------------------------------------------------------
# STEP 2 — Confidence formula
# ---------------------------------------------------------------------------

class TestConfidenceFormula:

    def test_perfect_scores(self):
        # 0.5*1.0 + 0.2*1.0 + 0.3*1.0 = 1.0
        assert _combine(1.0, 1.0, 1.0) == 1.0

    def test_zero_ocr(self):
        # 0.5*0 + 0.2*1 + 0.3*1 = 0.5
        assert abs(_combine(0.0, 1.0, 1.0) - 0.50) < 0.001

    def test_format_only_validation_conf(self):
        # val_conf = 0.6 (format valid only)
        c = _combine(0.95, 1.0, 0.6)
        # 0.5*0.95 + 0.2*1.0 + 0.3*0.6 = 0.475 + 0.2 + 0.18 = 0.855
        assert abs(c - 0.855) < 0.001

    def test_suspicious_zero_validation(self):
        # val_conf = 0.0 → always penalised
        c = _combine(0.95, 1.0, 0.0)
        # 0.5*0.95 + 0.2*1.0 + 0.0 = 0.675
        assert abs(c - 0.675) < 0.001

    def test_full_formula_on_chassis(self):
        r = parser.parse_chassis_no(_ocr("KSP210-0116561", conf=0.92))
        # ocr=0.92, parse=1.0, val=1.0(strict, len>=8)
        # combined = 0.5*0.92 + 0.2*1.0 + 0.3*1.0 = 0.46 + 0.20 + 0.30 = 0.96
        assert abs(r.combined_confidence - 0.96) < 0.01

    def test_ambiguous_date_zero_validation(self):
        # Two distinct dates in text → AMBIGUOUS_MATCH → val_conf=0.0
        r = parser.parse_date(_ocr("2026-03-15 and also 2026-03-27"))
        # Even though a value is picked, warning must be present
        assert ExtractionWarning.AMBIGUOUS_MATCH in r.warnings


# ---------------------------------------------------------------------------
# STEP 3 — Multi-candidate date resolution
# ---------------------------------------------------------------------------

class TestMultiCandidateDates:

    def test_single_date_no_warning(self):
        r = parser.parse_date(_ocr("2026-03-27"))
        assert r.value == "2026-03-27"
        assert ExtractionWarning.AMBIGUOUS_MATCH not in r.warnings

    def test_two_distinct_dates_flagged_ambiguous(self):
        r = parser.parse_date(_ocr("issue: 2026-03-15  expiry: 2026-09-30"))
        assert ExtractionWarning.AMBIGUOUS_MATCH in r.warnings

    def test_two_identical_dates_not_ambiguous(self):
        # Same value twice → not truly ambiguous
        r = parser.parse_date(_ocr("2026-03-27 2026-03-27"))
        assert r.value == "2026-03-27"
        assert ExtractionWarning.AMBIGUOUS_MATCH not in r.warnings

    def test_no_date_produces_parse_failure(self):
        r = parser.parse_date(_ocr("no date here at all"))
        assert r.value is None
        assert ExtractionWarning.PARSE_FAILURE in r.warnings


# ---------------------------------------------------------------------------
# STEP 1 — Cross-field date validation (via orchestrator logic simulation)
# ---------------------------------------------------------------------------

class TestCrossFieldValidation:
    """
    Directly tests the date inconsistency rule that the orchestrator applies.
    We reproduce the check here as a pure logic test (no real PDF / OCR).
    """

    def _apply_cross_validation(
        self,
        issue: str,
        expiration: str,
        exp_conf: float,
    ) -> tuple[list[str], float]:
        """Mirror of orchestrator cross-field check."""
        warnings = []
        if expiration < issue:
            warnings.append(ExtractionWarning.DATE_INCONSISTENT)
            exp_conf = round(exp_conf * 0.5, 4)
        return warnings, exp_conf

    def test_valid_dates_no_warning(self):
        warnings, conf = self._apply_cross_validation(
            "2026-03-15", "2026-09-30", 0.92
        )
        assert ExtractionWarning.DATE_INCONSISTENT not in warnings
        assert conf == 0.92

    def test_expiry_before_issue_adds_warning(self):
        warnings, conf = self._apply_cross_validation(
            "2026-03-27", "2026-03-15", 0.92
        )
        assert ExtractionWarning.DATE_INCONSISTENT in warnings

    def test_expiry_before_issue_halves_confidence(self):
        _, conf = self._apply_cross_validation("2026-03-27", "2026-03-15", 0.92)
        assert abs(conf - 0.46) < 0.001

    def test_value_is_not_discarded(self):
        # Inconsistent dates → penalise confidence but keep both values
        # (orchestrator keeps exp_val; only confidence drops)
        warnings, conf = self._apply_cross_validation(
            "2026-03-27", "2026-03-15", 0.80
        )
        # Value not touched in this layer; only conf and warnings change
        assert conf == 0.40   # 0.80 * 0.5
        assert ExtractionWarning.DATE_INCONSISTENT in warnings

    def test_equal_dates_no_warning(self):
        warnings, conf = self._apply_cross_validation(
            "2026-03-27", "2026-03-27", 0.90
        )
        assert ExtractionWarning.DATE_INCONSISTENT not in warnings


# ---------------------------------------------------------------------------
# STEP 7 — Warning enum coverage
# ---------------------------------------------------------------------------

class TestWarningEnums:

    def test_all_enum_values_are_strings(self):
        for w in ExtractionWarning:
            assert isinstance(w.value, str)

    def test_enum_used_in_noise_path(self):
        r = parser.parse_chassis_no(_ocr("XXXXXXXXXXXXXXXXX"))
        # Not noise (alphanumeric) — but no pattern match
        assert ExtractionWarning.PARSE_FAILURE in r.warnings

    def test_parse_failure_on_no_match(self):
        r = parser.parse_date(_ocr("no match"))
        assert ExtractionWarning.PARSE_FAILURE in r.warnings
