# -*- coding: utf-8 -*-
"""Shared fixtures for financial-analyst tests."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_eps_dataframe():
    """A minimal EPS DataFrame mimicking eps_fetcher output."""
    dates = pd.date_range('2023-01-15', periods=12, freq='QS')
    dates = dates.tz_localize('America/New_York')
    data = {
        'Name': [f"{2023 + i // 4}Q{i % 4 + 1}" for i in range(12)],
        'EPS': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, None, None, None, None],
        'Estimate EPS': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
    }
    df = pd.DataFrame(data, index=dates)
    df['EPSpast4Q'] = df['EPS'].shift(1).rolling(window=4).sum()
    df['EstimateEPSnext4Q'] = df['Estimate EPS'].shift(-4).rolling(window=4).sum()
    return df


@pytest.fixture
def sample_treasury_df():
    """Sample treasury yield DataFrame."""
    dates = pd.date_range('2024-01-01', periods=5, freq='B')
    return pd.DataFrame({
        '1M': [5.30, 5.31, 5.29, 5.28, 5.30],
        '3M': [5.25, 5.26, 5.24, 5.23, 5.25],
        '1Y': [4.80, 4.82, 4.78, 4.77, 4.80],
        '2Y': [4.35, 4.37, 4.33, 4.32, 4.35],
        '5Y': [4.10, 4.12, 4.08, 4.07, 4.10],
        '10Y': [4.20, 4.22, 4.18, 4.17, 4.20],
        '30Y': [4.40, 4.42, 4.38, 4.37, 4.40],
    }, index=dates)
