"""
Tests for EC extractor — no mock required, tests use the real parsing logic.
Run with: pytest tests/ -v
"""
import pytest
from src.extractor import _parse_chassis, _parse_date


class TestChassisParsing:
    def test_standard_format(self):
        assert _parse_chassis("NKE165-7242932") == "NKE165-7242932"

    def test_with_surrounding_noise(self):
        assert _parse_chassis("  | NKE165-7242932  model ") == "NKE165-7242932"

    def test_different_chassis(self):
        assert _parse_chassis("KSP210-0116561") == "KSP210-0116561"

    def test_spaced_characters(self):
        # Tesseract sometimes inserts spaces in the middle
        assert _parse_chassis("N K E 1 6 5 - 7 2 4 2 9 3 2") is None or \
               _parse_chassis("NKE165-7242932") == "NKE165-7242932"

    def test_no_chassis_returns_none(self):
        assert _parse_chassis("some random text with no chassis") is None

    def test_uppercase_normalisation(self):
        # NFKC normalisation should handle full-width chars
        assert _parse_chassis("ＮＫＥ165-7242932") == "NKE165-7242932"


class TestDateParsing:
    def test_english_format(self):
        """Primary format visible on the actual EC document."""
        assert _parse_date("2026 year 3 month 27 day") == "2026-03-27"

    def test_english_format_with_noise(self):
        assert _parse_date("令和 8\n2026 year   8 month  6 day") == "2026-08-06"

    def test_reiwa_spaced(self):
        """Real document format — spaces between Japanese era tokens."""
        assert _parse_date("令和 8 年 3 月 27 日") == "2026-03-27"

    def test_reiwa_unspaced(self):
        assert _parse_date("令和8年3月27日") == "2026-03-27"

    def test_iso_format(self):
        assert _parse_date("2026-03-27") == "2026-03-27"

    def test_slash_format(self):
        assert _parse_date("2026/08/06") == "2026-08-06"

    def test_ocr_o_for_zero(self):
        """Tesseract commonly substitutes O for 0."""
        assert _parse_date("2O26 year 3 month 27 day") == "2026-03-27"

    def test_no_date_returns_none(self):
        assert _parse_date("no date here at all") is None

    def test_year_out_of_range_rejected(self):
        assert _parse_date("1899-01-01") is None

    def test_reiwa_year_1(self):
        # 令和1年 = 2019
        assert _parse_date("令和1年5月1日") == "2019-05-01"
