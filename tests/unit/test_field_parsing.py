"""
Unit tests — date parser and chassis parser.
"""
import pytest

from src.domain.types import OcrResult
from src.core.parsing.field_parsing_service import FieldParsingService

parser = FieldParsingService()


def _ocr(text: str, conf: float = 0.95) -> OcrResult:
    return OcrResult(raw_text=text, confidence=conf, engine="test")


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

class TestDateParser:

    def test_iso_hyphen(self):
        r = parser.parse_date(_ocr("2026-03-27"), "issue_date")
        assert r.value == "2026-03-27"
        assert r.parse_confidence == 1.0

    def test_iso_slash(self):
        r = parser.parse_date(_ocr("2026/03/27"))
        assert r.value == "2026-03-27"

    def test_dmy_dot(self):
        r = parser.parse_date(_ocr("27.03.2026"))
        assert r.value == "2026-03-27"

    def test_reiwa_era(self):
        # 令和8年3月27日 → 2026-03-27
        r = parser.parse_date(_ocr("令和8年3月27日"))
        assert r.value == "2026-03-27"

    def test_heisei_era(self):
        # 平成31年4月30日 → 2019-04-30
        r = parser.parse_date(_ocr("平成31年4月30日"))
        assert r.value == "2019-04-30"

    def test_compact(self):
        r = parser.parse_date(_ocr("20260327"))
        assert r.value == "2026-03-27"

    def test_invalid_date_returns_none(self):
        r = parser.parse_date(_ocr("2026-13-99"))
        assert r.value is None
        assert r.parse_confidence == 0.0

    def test_out_of_range_year(self):
        r = parser.parse_date(_ocr("1800-01-01"))
        assert r.value is None

    def test_empty_text(self):
        r = parser.parse_date(_ocr(""))
        assert r.value is None

    def test_noisy_text_with_valid_date(self):
        r = parser.parse_date(_ocr("発行日: 2026-03-15 その他情報"))
        assert r.value == "2026-03-15"


# ---------------------------------------------------------------------------
# Chassis parser
# ---------------------------------------------------------------------------

class TestChassisParser:

    @pytest.mark.parametrize("text,expected", [
        ("NKE165G-7242861", "NKE165G-7242861"),
        ("KSP210-0116561", "KSP210-0116561"),
        ("  1946AA-NKE165G  ", "1946AA-NKE165G"),
        ("車台番号 ZWR80G-0483830 その他", "ZWR80G-0483830"),
        ("MH95S-291382", "MH95S-291382"),
        ("A202A-0075880", "A202A-0075880"),
    ])
    def test_valid_chassis(self, text, expected):
        r = parser.parse_chassis_no(_ocr(text))
        assert r.value == expected
        assert r.parse_confidence == 1.0

    def test_no_chassis_returns_none(self):
        r = parser.parse_chassis_no(_ocr("発行日 2026-03-27"))
        assert r.value is None
        assert r.parse_confidence == 0.0

    def test_ocr_garbage(self):
        r = parser.parse_chassis_no(_ocr("!@# $$% ^^^"))
        assert r.value is None

    def test_low_ocr_confidence_lowers_combined(self):
        r = parser.parse_chassis_no(_ocr("NKE165G-7242861", conf=0.50))
        # parse_conf=1.0, ocr_conf=0.50, val_conf=1.0 (strict)
        # combined = 0.5*0.50 + 0.2*1.0 + 0.3*1.0 = 0.25 + 0.20 + 0.30 = 0.75
        assert abs(r.combined_confidence - 0.75) < 0.01
        assert r.value == "NKE165G-7242861"

    def test_nfkc_normalisation(self):
        # Fullwidth hyphen-minus
        r = parser.parse_chassis_no(_ocr("NKE165G－7242861"))  # U+FF0D
        # After NFKC this becomes a regular hyphen
        assert r.value is not None
