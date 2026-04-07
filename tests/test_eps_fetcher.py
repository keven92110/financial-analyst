# -*- coding: utf-8 -*-
"""Tests for eps_fetcher.py — EPS data pipeline utilities."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from eps_fetcher import _quarter_name, _next_quarter, _split_factor, _fix_double_adjustment


class TestQuarterName:
    """Test report date -> quarter name mapping.

    Key rule: earnings are reported ~45 days after quarter end, so:
      Jan-Mar report -> previous year Q4
      Apr-Jun report -> current year Q1
      Jul-Sep report -> current year Q2
      Oct-Dec report -> current year Q3
    """

    def test_january_is_prev_year_q4(self):
        dt = datetime(2025, 1, 28)
        assert _quarter_name(dt) == "2024Q4"

    def test_february_is_prev_year_q4(self):
        dt = datetime(2025, 2, 15)
        assert _quarter_name(dt) == "2024Q4"

    def test_march_is_prev_year_q4(self):
        dt = datetime(2025, 3, 31)
        assert _quarter_name(dt) == "2024Q4"

    def test_april_is_q1(self):
        dt = datetime(2025, 4, 15)
        assert _quarter_name(dt) == "2025Q1"

    def test_june_is_q1(self):
        dt = datetime(2025, 6, 30)
        assert _quarter_name(dt) == "2025Q1"

    def test_july_is_q2(self):
        dt = datetime(2025, 7, 20)
        assert _quarter_name(dt) == "2025Q2"

    def test_september_is_q2(self):
        dt = datetime(2025, 9, 30)
        assert _quarter_name(dt) == "2025Q2"

    def test_october_is_q3(self):
        dt = datetime(2025, 10, 25)
        assert _quarter_name(dt) == "2025Q3"

    def test_december_is_q3(self):
        dt = datetime(2025, 12, 31)
        assert _quarter_name(dt) == "2025Q3"

    def test_year_boundary(self):
        """Jan 2026 report -> 2025 Q4."""
        dt = datetime(2026, 1, 5)
        assert _quarter_name(dt) == "2025Q4"


class TestNextQuarter:
    """Test quarter name advancement."""

    def test_q1_to_q2(self):
        assert _next_quarter("2025Q1") == "2025Q2"

    def test_q2_to_q3(self):
        assert _next_quarter("2025Q2") == "2025Q3"

    def test_q3_to_q4(self):
        assert _next_quarter("2025Q3") == "2025Q4"

    def test_q4_wraps_to_next_year_q1(self):
        assert _next_quarter("2025Q4") == "2026Q1"

    def test_chain_four_quarters(self):
        """Advancing 4 times should bring us to the same quarter next year."""
        q = "2024Q2"
        for _ in range(4):
            q = _next_quarter(q)
        assert q == "2025Q2"


class TestSplitFactor:
    """Test cumulative split ratio computation."""

    def test_no_splits(self):
        """Empty splits series should return factor 1.0."""
        splits = pd.Series(dtype=float)
        assert _split_factor(splits, datetime(2024, 6, 1)) == 1.0

    def test_single_split_before(self):
        """Date before a 4:1 split should accumulate factor 4."""
        splits = pd.Series(
            {pd.Timestamp('2024-06-10'): 4.0}
        )
        factor = _split_factor(splits, datetime(2024, 1, 1))
        assert factor == 4.0

    def test_single_split_after(self):
        """Date after a split should NOT accumulate the split."""
        splits = pd.Series(
            {pd.Timestamp('2024-06-10'): 4.0}
        )
        factor = _split_factor(splits, datetime(2024, 7, 1))
        assert factor == 1.0

    def test_multiple_splits(self):
        """Multiple splits should multiply together."""
        splits = pd.Series({
            pd.Timestamp('2022-01-15'): 2.0,
            pd.Timestamp('2023-06-10'): 3.0,
        })
        # Date before both splits: 2 * 3 = 6
        factor = _split_factor(splits, datetime(2021, 1, 1))
        assert factor == 6.0

    def test_between_splits(self):
        """Date between two splits should only accumulate the later one."""
        splits = pd.Series({
            pd.Timestamp('2022-01-15'): 2.0,
            pd.Timestamp('2023-06-10'): 3.0,
        })
        factor = _split_factor(splits, datetime(2022, 6, 1))
        assert factor == 3.0


class TestFixDoubleAdjustment:
    """Test double split-adjustment detection and correction."""

    def test_no_splits_no_change(self):
        """With no splits, rows should not be modified."""
        rows = [
            {'Date': '2024.01.15', 'EPS': 1.50, '_adj_eps': 1.50,
             '_orig_eps': 1.50, '_orig_est': 1.40},
            {'Date': '2024.04.15', 'EPS': 1.60, '_adj_eps': 1.60,
             '_orig_eps': 1.60, '_orig_est': 1.50},
        ]
        splits = pd.Series(dtype=float)
        _fix_double_adjustment(rows, splits)
        assert rows[0]['EPS'] == 1.50
        assert rows[1]['EPS'] == 1.60

    def test_detects_double_adjustment(self):
        """A value that is <20% of neighbors near a split date should be restored."""
        rows = [
            {'Date': '2024.01.15', 'EPS': 6.00, '_adj_eps': 6.00,
             '_orig_eps': 6.00, '_orig_est': None},
            # This one was double-adjusted: original=1.50, adj=0.375 (divided by 4 unnecessarily)
            {'Date': '2024.04.15', 'EPS': 0.375, '_adj_eps': 0.375,
             '_orig_eps': 1.50, '_orig_est': 1.40},
            {'Date': '2024.07.15', 'EPS': 6.50, '_adj_eps': 6.50,
             '_orig_eps': 6.50, '_orig_est': None},
        ]
        splits = pd.Series({pd.Timestamp('2024-06-10'): 4.0})
        _fix_double_adjustment(rows, splits)
        # The middle row should be restored to original Yahoo value
        assert rows[1]['EPS'] == 1.50
        assert rows[1]['Estimate EPS'] == 1.40

    def test_no_false_positive_far_from_split(self):
        """Values far from split dates should not be modified even if small."""
        rows = [
            {'Date': '2020.01.15', 'EPS': 5.00, '_adj_eps': 5.00,
             '_orig_eps': 5.00, '_orig_est': None},
            # Small value but >365 days from split
            {'Date': '2020.04.15', 'EPS': 0.50, '_adj_eps': 0.50,
             '_orig_eps': 0.50, '_orig_est': None},
            {'Date': '2020.07.15', 'EPS': 5.00, '_adj_eps': 5.00,
             '_orig_eps': 5.00, '_orig_est': None},
        ]
        splits = pd.Series({pd.Timestamp('2024-06-10'): 4.0})
        _fix_double_adjustment(rows, splits)
        # Should NOT change because split is years away
        assert rows[1]['EPS'] == 0.50

    def test_skips_rows_without_adj_eps(self):
        """Rows without _adj_eps should be skipped."""
        rows = [
            {'Date': '2024.01.15', 'EPS': 1.50, '_adj_eps': None,
             '_orig_eps': 1.50, '_orig_est': None},
        ]
        splits = pd.Series({pd.Timestamp('2024-06-10'): 4.0})
        _fix_double_adjustment(rows, splits)
        assert rows[0]['EPS'] == 1.50
