# -*- coding: utf-8 -*-
"""Tests for iv_risk.py — implied volatility and ERP calculations.

These tests focus on the mathematical transformations rather than
yfinance API calls, using mocked market data.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestIVtoERPConversion:
    """Test the IV-to-ERP formula: ERP = 0.5 * (IV/100)^2."""

    def test_erp_formula_30pct_iv(self):
        """30% blended IV -> ERP = 0.5 * 0.30^2 = 0.045 (4.5%)."""
        iv_decimal = 30.0 / 100
        erp = 0.5 * iv_decimal ** 2
        assert pytest.approx(erp, abs=1e-6) == 0.045

    def test_erp_formula_20pct_iv(self):
        """20% IV -> ERP = 0.5 * 0.04 = 0.02 (2%)."""
        iv_decimal = 20.0 / 100
        erp = 0.5 * iv_decimal ** 2
        assert pytest.approx(erp, abs=1e-6) == 0.02

    def test_erp_formula_50pct_iv(self):
        """50% IV -> ERP = 0.5 * 0.25 = 0.125 (12.5%)."""
        iv_decimal = 50.0 / 100
        erp = 0.5 * iv_decimal ** 2
        assert pytest.approx(erp, abs=1e-6) == 0.125

    def test_erp_increases_with_iv(self):
        """ERP should increase quadratically with IV."""
        ivs = [10, 20, 30, 40, 50]
        erps = [0.5 * (iv / 100) ** 2 for iv in ivs]
        for i in range(len(erps) - 1):
            assert erps[i + 1] > erps[i]


class TestBlendedIV:
    """Test the 60/40 blending of beta_iv and realized_vol."""

    def test_blending_weights(self):
        """60% beta_iv + 40% realized_vol."""
        beta_iv = 35.0
        realized_vol = 25.0
        blended = 0.6 * beta_iv + 0.4 * realized_vol
        assert pytest.approx(blended) == 31.0

    def test_equal_inputs_equal_output(self):
        """When beta_iv == realized_vol, blended should equal both."""
        val = 30.0
        blended = 0.6 * val + 0.4 * val
        assert pytest.approx(blended) == val


class TestBetaCalculation:
    """Test rolling beta computation logic."""

    def test_beta_of_spy_is_one(self):
        """SPY regressed against itself should have beta ~1.0."""
        np.random.seed(42)
        n = 200
        spy_returns = pd.Series(np.random.normal(0, 0.01, n))
        # Stock = SPY (beta=1)
        stock_returns = spy_returns.copy()

        cov = stock_returns.rolling(120, min_periods=60).cov(spy_returns)
        var = spy_returns.rolling(120, min_periods=60).var()
        beta = (cov / var).clip(0.5, 5.0)

        # After warm-up, beta should be ~1.0
        valid = beta.dropna()
        assert len(valid) > 0
        assert pytest.approx(valid.iloc[-1], abs=0.01) == 1.0

    def test_beta_double_leverage(self):
        """Stock with 2x SPY returns should have beta ~2.0."""
        np.random.seed(42)
        n = 200
        spy_returns = pd.Series(np.random.normal(0, 0.01, n))
        stock_returns = spy_returns * 2.0

        cov = stock_returns.rolling(120, min_periods=60).cov(spy_returns)
        var = spy_returns.rolling(120, min_periods=60).var()
        beta = (cov / var).clip(0.5, 5.0)

        valid = beta.dropna()
        assert pytest.approx(valid.iloc[-1], abs=0.05) == 2.0

    def test_beta_clamping_lower(self):
        """Very low beta should be clamped to 0.5."""
        np.random.seed(42)
        n = 200
        spy_returns = pd.Series(np.random.normal(0, 0.01, n))
        # Near-zero correlation stock
        stock_returns = pd.Series(np.random.normal(0, 0.001, n))

        cov = stock_returns.rolling(120, min_periods=60).cov(spy_returns)
        var = spy_returns.rolling(120, min_periods=60).var()
        beta = (cov / var).clip(0.5, 5.0)

        valid = beta.dropna()
        assert valid.min() >= 0.5

    def test_beta_clamping_upper(self):
        """Very high beta should be clamped to 5.0."""
        np.random.seed(42)
        n = 200
        spy_returns = pd.Series(np.random.normal(0, 0.01, n))
        stock_returns = spy_returns * 10.0  # 10x leverage

        cov = stock_returns.rolling(120, min_periods=60).cov(spy_returns)
        var = spy_returns.rolling(120, min_periods=60).var()
        beta = (cov / var).clip(0.5, 5.0)

        valid = beta.dropna()
        assert valid.max() <= 5.0


class TestRealizedVolatility:
    """Test realized volatility calculation."""

    def test_realized_vol_annualization(self):
        """30-day rolling std * sqrt(252) should annualize correctly."""
        np.random.seed(42)
        daily_vol = 0.01  # 1% daily vol
        n = 100
        returns = pd.Series(np.random.normal(0, daily_vol, n))

        realized = returns.rolling(30, min_periods=20).std() * np.sqrt(252) * 100
        valid = realized.dropna()

        # Expected annual vol ~ 1% * sqrt(252) * 100 ~ 15.87%
        expected = daily_vol * np.sqrt(252) * 100
        assert pytest.approx(valid.mean(), rel=0.3) == expected


class TestComputeStockIV:
    """Integration test for compute_stock_iv with mocked yfinance data."""

    @patch('iv_risk.yf.Ticker')
    @patch('iv_risk._get_vix')
    @patch('iv_risk._get_spy_returns')
    def test_returns_expected_columns(self, mock_spy, mock_vix, mock_ticker):
        """compute_stock_iv should return DataFrame with expected columns."""
        n = 300
        dates = pd.bdate_range('2022-01-01', periods=n, tz='America/New_York')

        # Mock stock prices
        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, n)) + 5, index=dates)
        prices = prices.clip(lower=1.0)
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame({'Close': prices})
        mock_ticker.return_value = mock_stock

        # Mock VIX
        mock_vix.return_value = pd.Series(np.random.uniform(15, 30, n), index=dates)

        # Mock SPY returns
        mock_spy.return_value = pd.Series(np.random.normal(0, 0.01, n), index=dates)

        from iv_risk import compute_stock_iv
        result = compute_stock_iv('TEST', start_date='2022-01-01')

        expected_cols = ['vix', 'realized_vol', 'rolling_beta', 'beta_iv',
                         'blended_iv', 'implied_erp']
        for col in expected_cols:
            assert col in result.columns

    @patch('iv_risk.yf.Ticker')
    @patch('iv_risk._get_vix')
    @patch('iv_risk._get_spy_returns')
    def test_implied_erp_is_positive(self, mock_spy, mock_vix, mock_ticker):
        """Implied ERP should always be non-negative."""
        n = 300
        dates = pd.bdate_range('2022-01-01', periods=n, tz='America/New_York')

        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.02, n)) + 5, index=dates)
        prices = prices.clip(lower=1.0)
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame({'Close': prices})
        mock_ticker.return_value = mock_stock

        mock_vix.return_value = pd.Series(np.random.uniform(15, 30, n), index=dates)
        mock_spy.return_value = pd.Series(np.random.normal(0, 0.01, n), index=dates)

        from iv_risk import compute_stock_iv
        result = compute_stock_iv('TEST', start_date='2022-01-01')

        assert (result['implied_erp'].dropna() >= 0).all()
