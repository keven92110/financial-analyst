# -*- coding: utf-8 -*-
"""Tests for server.py — caching logic and serialization helpers."""

import time
import pandas as pd
import numpy as np
import pytest


class TestCaching:
    """Test the in-memory TTL cache."""

    def test_set_and_get(self):
        from server import _get_cached, _set_cached, _cache, _cache_ts
        # Clean state
        _cache.clear()
        _cache_ts.clear()

        _set_cached('test_key', {'data': 42})
        result = _get_cached('test_key')
        assert result == {'data': 42}

    def test_cache_miss(self):
        from server import _get_cached, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        result = _get_cached('nonexistent')
        assert result is None

    def test_cache_expiry(self):
        from server import _get_cached, _set_cached, _cache, _cache_ts, CACHE_TTL
        _cache.clear()
        _cache_ts.clear()

        _set_cached('expire_test', 'value')
        # Manually expire by backdating timestamp
        _cache_ts['expire_test'] = time.time() - CACHE_TTL - 1
        result = _get_cached('expire_test')
        assert result is None

    def test_cache_not_expired(self):
        from server import _get_cached, _set_cached, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        _set_cached('fresh_test', 'value')
        result = _get_cached('fresh_test')
        assert result == 'value'


class TestTsToStr:
    """Test timestamp index to string conversion."""

    def test_datetime_index(self):
        from server import _ts_to_str
        idx = pd.DatetimeIndex(['2024-01-01', '2024-06-15'])
        result = _ts_to_str(idx)
        assert result == ['2024-01-01', '2024-06-15']

    def test_string_index(self):
        from server import _ts_to_str
        idx = ['2024-01-01', '2024-06-15']
        result = _ts_to_str(idx)
        assert result == ['2024-01-01', '2024-06-15']


class TestSeriesToDict:
    """Test pandas Series serialization."""

    def test_basic_series(self):
        from server import _series_to_dict
        s = pd.Series([1.0, 2.5, 3.7], index=pd.date_range('2024-01-01', periods=3))
        result = _series_to_dict(s)
        assert 'dates' in result
        assert 'values' in result
        assert len(result['dates']) == 3
        assert len(result['values']) == 3
        assert result['values'] == [1.0, 2.5, 3.7]

    def test_drops_nan(self):
        from server import _series_to_dict
        s = pd.Series([1.0, np.nan, 3.0], index=pd.date_range('2024-01-01', periods=3))
        result = _series_to_dict(s)
        # NaN rows should be dropped
        assert len(result['dates']) == 2
        assert len(result['values']) == 2

    def test_rounds_to_4_decimals(self):
        from server import _series_to_dict
        s = pd.Series([1.123456789], index=pd.date_range('2024-01-01', periods=1))
        result = _series_to_dict(s)
        assert result['values'][0] == 1.1235


class TestDfToDict:
    """Test DataFrame serialization."""

    def test_basic_dataframe(self):
        from server import _df_to_dict
        df = pd.DataFrame({
            'A': [1.0, 2.0],
            'B': [3.0, 4.0],
        }, index=['row1', 'row2'])
        result = _df_to_dict(df)
        assert result['index'] == ['row1', 'row2']
        assert result['columns'] == ['A', 'B']
        assert result['data']['A'] == [1.0, 2.0]
        assert result['data']['B'] == [3.0, 4.0]

    def test_handles_nan(self):
        from server import _df_to_dict
        df = pd.DataFrame({
            'A': [1.0, np.nan],
        }, index=['r1', 'r2'])
        result = _df_to_dict(df)
        assert result['data']['A'][0] == 1.0
        assert result['data']['A'][1] is None
