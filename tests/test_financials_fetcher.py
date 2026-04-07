# -*- coding: utf-8 -*-
"""Tests for financials_fetcher.py — financial statement utilities."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from financials_fetcher import STATEMENTS, fetch_financials, load_all_financials


class TestConstants:
    """Verify statement type definitions."""

    def test_has_6_statement_types(self):
        assert len(STATEMENTS) == 6

    def test_all_statement_types_present(self):
        expected = [
            'income_annual', 'income_quarterly',
            'balance_annual', 'balance_quarterly',
            'cashflow_annual', 'cashflow_quarterly',
        ]
        for key in expected:
            assert key in STATEMENTS

    def test_csv_names_are_unique(self):
        csv_names = [csv for _, csv in STATEMENTS.values()]
        assert len(csv_names) == len(set(csv_names))

    def test_csv_names_end_with_csv(self):
        for _, csv_name in STATEMENTS.values():
            assert csv_name.endswith('.csv')


class TestFetchFinancials:
    """Test fetch_financials with mocked yfinance."""

    @patch('financials_fetcher.yf.Ticker')
    def test_returns_all_statement_keys(self, mock_ticker):
        """Should return a dict with all 6 statement keys."""
        mock_stock = MagicMock()
        # Mock each attribute to return an empty DataFrame
        for key, (attr_name, _) in STATEMENTS.items():
            setattr(mock_stock, attr_name, pd.DataFrame())
        mock_ticker.return_value = mock_stock

        result = fetch_financials('TEST')
        assert len(result) == 6
        for key in STATEMENTS:
            assert key in result

    @patch('financials_fetcher.yf.Ticker')
    def test_handles_none_statements(self, mock_ticker):
        """Should return empty DataFrame when yfinance returns None."""
        mock_stock = MagicMock()
        for key, (attr_name, _) in STATEMENTS.items():
            setattr(mock_stock, attr_name, None)
        mock_ticker.return_value = mock_stock

        result = fetch_financials('TEST')
        for key in STATEMENTS:
            assert isinstance(result[key], pd.DataFrame)
            assert result[key].empty

    @patch('financials_fetcher.yf.Ticker')
    def test_converts_timestamp_columns(self, mock_ticker):
        """Timestamp columns should be converted to date strings."""
        mock_stock = MagicMock()
        dates = [pd.Timestamp('2024-03-31'), pd.Timestamp('2023-03-31')]
        df = pd.DataFrame(
            [[100e9, 90e9]],
            index=['Total Revenue'],
            columns=dates,
        )

        # Set income_stmt to our test DataFrame, others to empty
        mock_stock.income_stmt = df
        for key, (attr_name, _) in STATEMENTS.items():
            if attr_name != 'income_stmt':
                setattr(mock_stock, attr_name, pd.DataFrame())
        mock_ticker.return_value = mock_stock

        result = fetch_financials('TEST')
        # Columns should be string dates
        cols = result['income_annual'].columns.tolist()
        assert all(isinstance(c, str) for c in cols)
        assert '2024-03-31' in cols


class TestLoadAllFinancials:
    """Test caching behavior of load_all_financials."""

    @patch('financials_fetcher.fetch_financials')
    def test_uses_cache_when_fresh(self, mock_fetch, tmp_path, monkeypatch):
        """Should not call fetch_financials when cache is fresh."""
        import os
        monkeypatch.setattr('financials_fetcher.CACHE_DIR', str(tmp_path))

        # Create a fresh cache file
        ticker_dir = tmp_path / 'TEST'
        ticker_dir.mkdir()
        for key, (_, csv_name) in STATEMENTS.items():
            path = ticker_dir / csv_name
            pd.DataFrame({'A': [1]}).to_csv(path)

        result = load_all_financials(['TEST'], cache_hours=24)
        mock_fetch.assert_not_called()
        assert 'TEST' in result
