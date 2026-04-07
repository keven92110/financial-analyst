# -*- coding: utf-8 -*-
"""Tests for rates_fetcher.py — rate data utilities and constants."""

import pandas as pd
import pytest

from rates_fetcher import (
    TREASURY_SERIES, FED_FUNDS_SERIES, FOMC_DOT_SERIES, _MONTH_CODES
)


class TestConstants:
    """Verify FRED series IDs and month codes are correct."""

    def test_treasury_has_11_maturities(self):
        assert len(TREASURY_SERIES) == 11

    def test_treasury_includes_key_maturities(self):
        for key in ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']:
            assert key in TREASURY_SERIES

    def test_10y_treasury_series_id(self):
        assert TREASURY_SERIES['10Y'] == 'DGS10'

    def test_fed_funds_has_3_series(self):
        assert len(FED_FUNDS_SERIES) == 3
        assert 'effective' in FED_FUNDS_SERIES
        assert 'target_upper' in FED_FUNDS_SERIES
        assert 'target_lower' in FED_FUNDS_SERIES

    def test_fomc_dot_series(self):
        assert 'median' in FOMC_DOT_SERIES
        assert 'median_longrun' in FOMC_DOT_SERIES

    def test_month_codes_complete(self):
        """All 12 months should have a ZQ futures code."""
        assert len(_MONTH_CODES) == 12
        for m in range(1, 13):
            assert m in _MONTH_CODES

    def test_month_codes_unique(self):
        """All month codes should be unique single letters."""
        codes = list(_MONTH_CODES.values())
        assert len(codes) == len(set(codes))
        assert all(len(c) == 1 and c.isalpha() for c in codes)


class TestLoadRateHistory:
    """Test load_rate_history with no history file."""

    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr('rates_fetcher.HISTORY_DIR', str(tmp_path))
        from rates_fetcher import load_rate_history
        result = load_rate_history()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
