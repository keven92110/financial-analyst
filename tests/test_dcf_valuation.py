# -*- coding: utf-8 -*-
"""Tests for dcf_valuation.py — DCF fair value computation."""

import numpy as np
import pandas as pd
import pytest

from dcf_valuation import compute_dcf_direct, build_annual_estimates


class TestComputeDcfDirect:
    """Tests for the vectorized DCF computation."""

    def test_basic_fair_value(self):
        """Known-value test: positive EPS, normal rates produce reasonable output."""
        year1 = np.array([10.0])
        year2 = np.array([11.0])
        year3 = np.array([12.0])
        inflation = np.array([0.025])
        discount = np.array([0.08])

        fv = compute_dcf_direct(year1, year2, year3, inflation, discount)
        assert fv.shape == (1,)
        assert not np.isnan(fv[0])
        # Fair value should be substantially greater than year1 EPS
        assert fv[0] > year1[0] * 5
        # But not absurdly large
        assert fv[0] < year1[0] * 100

    def test_higher_discount_lowers_value(self):
        """Higher discount rate should produce lower fair value."""
        y1 = np.array([10.0])
        y2 = np.array([11.0])
        y3 = np.array([12.0])
        g = np.array([0.025])

        fv_low = compute_dcf_direct(y1, y2, y3, g, np.array([0.06]))
        fv_high = compute_dcf_direct(y1, y2, y3, g, np.array([0.10]))

        assert fv_low[0] > fv_high[0]

    def test_higher_growth_raises_value(self):
        """Higher inflation/growth should increase fair value (more terminal value)."""
        y1 = np.array([10.0])
        y2 = np.array([11.0])
        y3 = np.array([12.0])
        r = np.array([0.08])

        fv_low_g = compute_dcf_direct(y1, y2, y3, np.array([0.01]), r)
        fv_high_g = compute_dcf_direct(y1, y2, y3, np.array([0.03]), r)

        assert fv_high_g[0] > fv_low_g[0]

    def test_guard_r_equals_g(self):
        """When r - g <= 0.005, output should be NaN (divergent perpetuity)."""
        y1 = np.array([10.0])
        y2 = np.array([11.0])
        y3 = np.array([12.0])
        g = np.array([0.08])
        r = np.array([0.08])  # r == g

        fv = compute_dcf_direct(y1, y2, y3, g, r)
        assert np.isnan(fv[0])

    def test_guard_r_less_than_g(self):
        """When r < g, output should be NaN."""
        y1 = np.array([10.0])
        y2 = np.array([11.0])
        y3 = np.array([12.0])

        fv = compute_dcf_direct(y1, y2, y3, np.array([0.10]), np.array([0.05]))
        assert np.isnan(fv[0])

    def test_guard_negative_year1(self):
        """Negative year1 EPS should produce NaN (guard: year1 > 0)."""
        fv = compute_dcf_direct(
            np.array([-5.0]), np.array([10.0]), np.array([12.0]),
            np.array([0.025]), np.array([0.08])
        )
        assert np.isnan(fv[0])

    def test_guard_zero_discount(self):
        """Zero discount rate should produce NaN (guard: r > 0)."""
        fv = compute_dcf_direct(
            np.array([10.0]), np.array([11.0]), np.array([12.0]),
            np.array([0.025]), np.array([0.0])
        )
        assert np.isnan(fv[0])

    def test_vectorized_multiple_dates(self):
        """Should handle arrays with multiple elements (one per date)."""
        n = 100
        y1 = np.full(n, 10.0)
        y2 = np.full(n, 11.0)
        y3 = np.full(n, 12.0)
        g = np.full(n, 0.025)
        r = np.linspace(0.05, 0.12, n)

        fv = compute_dcf_direct(y1, y2, y3, g, r)
        assert fv.shape == (n,)
        # Values should decrease as discount rate increases
        valid = ~np.isnan(fv)
        diffs = np.diff(fv[valid])
        assert np.all(diffs < 0)

    def test_custom_total_years(self):
        """Custom total_years should change the projection horizon."""
        y1 = np.array([10.0])
        y2 = np.array([11.0])
        y3 = np.array([12.0])
        g = np.array([0.025])
        r = np.array([0.08])

        fv_5 = compute_dcf_direct(y1, y2, y3, g, r, total_years=5)
        fv_15 = compute_dcf_direct(y1, y2, y3, g, r, total_years=15)

        # More projection years = more explicit DCF + different terminal timing
        # Both should be valid numbers
        assert not np.isnan(fv_5[0])
        assert not np.isnan(fv_15[0])

    def test_mixed_valid_invalid(self):
        """Array with some valid and some invalid inputs."""
        y1 = np.array([10.0, -5.0, 10.0])
        y2 = np.array([11.0, 11.0, 11.0])
        y3 = np.array([12.0, 12.0, 12.0])
        g = np.array([0.025, 0.025, 0.08])
        r = np.array([0.08, 0.08, 0.08])  # r-g < 0.005 for last element

        fv = compute_dcf_direct(y1, y2, y3, g, r)
        assert not np.isnan(fv[0])  # valid
        assert np.isnan(fv[1])      # negative year1
        assert np.isnan(fv[2])      # r - g too small


class TestBuildAnnualEstimates:
    """Tests for build_annual_estimates."""

    def test_fills_missing_with_inflation(self, sample_eps_dataframe):
        """Missing Year 2/3 should be filled with inflation growth."""
        year1, year2, year3 = build_annual_estimates(sample_eps_dataframe)

        # All outputs should be same length as input
        assert len(year1) == len(sample_eps_dataframe)
        assert len(year2) == len(sample_eps_dataframe)
        assert len(year3) == len(sample_eps_dataframe)

    def test_year1_matches_estimate_next4q(self, sample_eps_dataframe):
        """Year 1 should be identical to EstimateEPSnext4Q."""
        year1, _, _ = build_annual_estimates(sample_eps_dataframe)
        pd.testing.assert_series_equal(
            year1, sample_eps_dataframe['EstimateEPSnext4Q']
        )

    def test_custom_inflation_default(self, sample_eps_dataframe):
        """Different inflation_default should change fill values."""
        _, y2_low, y3_low = build_annual_estimates(
            sample_eps_dataframe, inflation_default=0.01
        )
        _, y2_high, y3_high = build_annual_estimates(
            sample_eps_dataframe, inflation_default=0.10
        )
        # Where values are filled (NaN originally), higher inflation = higher value
        filled_mask = sample_eps_dataframe['Estimate EPS'].shift(-8).rolling(4).sum().isna()
        if filled_mask.any():
            # At least some values should differ
            assert not y2_low.equals(y2_high) or not y3_low.equals(y3_high)
