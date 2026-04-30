"""Statistical baseline predictors.

These mirror what the desktop app already does, but exposed as functions that
take a (test_row, train_panel) and return an array of historical fwd-return
samples — i.e. an empirical distribution prediction.

Walk-forward fairness: each predictor only sees `train_panel`, never test data.

Note on cat_c (K-means): the K-means model in the cached panel was fit on
*all* historical data including future periods. This causes a small look-ahead
leak (cluster centroids are slightly informed by future). The leak is small
because K-means clusters paths, not returns, but if rigor matters we should
refit K-means walk-forward at each cutoff. For Phase 1 we accept the leak;
NN models will be evaluated on the same panel for fair comparison.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# Trailing-window sizes (calendar days) used by *_trailing predictors
TRAIL_DAYS = {
    'all':  None,   # no trailing filter; use all of train_panel
    '5y':   365 * 5,
    '10y':  365 * 10,
    '2y':   365 * 2,
    '12m':  365,
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def _trailing_filter(train: pd.DataFrame, end_date: pd.Timestamp,
                     trail_days: Optional[int]) -> pd.DataFrame:
    if trail_days is None:
        return train
    cutoff = end_date - pd.Timedelta(days=trail_days)
    return train[train['end_date'] >= cutoff]


# ─── Predictors ───────────────────────────────────────────────────────────

def predict_unconditional(test_row, train: pd.DataFrame, horizon: str,
                          trail: str = 'all',
                          per_index: bool = False) -> np.ndarray:
    """No conditioning — just historical fwd returns over the trailing window.

    Sanity baseline. Anything more sophisticated must beat this.
    """
    end_date = test_row['end_date']
    sub = _trailing_filter(train, end_date, TRAIL_DAYS[trail])
    if per_index:
        sub = sub[sub['index'] == test_row['index']]
    return sub[horizon].dropna().values


def predict_method(test_row, train: pd.DataFrame, horizon: str,
                   method: str, trail: str = 'all',
                   per_index: bool = False) -> np.ndarray:
    """Match historical windows that share the same `method` category as the
    test row (e.g. cat_d == 'Oversold (RSI<30)')."""
    end_date = test_row['end_date']
    cat = test_row[method]
    if cat is None or (isinstance(cat, float) and np.isnan(cat)):
        return np.array([])
    sub = _trailing_filter(train, end_date, TRAIL_DAYS[trail])
    if per_index:
        sub = sub[sub['index'] == test_row['index']]
    sub = sub[sub[method] == cat]
    return sub[horizon].dropna().values


def predict_combined(test_row, train: pd.DataFrame, horizon: str,
                     trail: str = 'all',
                     per_index: bool = False) -> np.ndarray:
    """Pool windows that match ANY of B/C/D categories of the test row.
    (Union — bigger sample, more robust than intersection which collapses fast.)"""
    end_date = test_row['end_date']
    sub = _trailing_filter(train, end_date, TRAIL_DAYS[trail])
    if per_index:
        sub = sub[sub['index'] == test_row['index']]
    masks = []
    for m in ('cat_b', 'cat_c', 'cat_d'):
        cat = test_row[m]
        if cat is not None and not (isinstance(cat, float) and np.isnan(cat)):
            masks.append(sub[m] == cat)
    if not masks:
        return np.array([])
    union = masks[0]
    for m in masks[1:]:
        union = union | m
    return sub.loc[union, horizon].dropna().values


def predict_intersection(test_row, train: pd.DataFrame, horizon: str,
                         trail: str = 'all',
                         per_index: bool = False) -> np.ndarray:
    """Match windows where ALL of B/C/D categories match the test row.
    Smaller sample but tighter conditioning."""
    end_date = test_row['end_date']
    sub = _trailing_filter(train, end_date, TRAIL_DAYS[trail])
    if per_index:
        sub = sub[sub['index'] == test_row['index']]
    for m in ('cat_b', 'cat_c', 'cat_d'):
        cat = test_row[m]
        if cat is None or (isinstance(cat, float) and np.isnan(cat)):
            return np.array([])  # can't match; return empty
        sub = sub[sub[m] == cat]
    return sub[horizon].dropna().values


# ─── Predictor registry ──────────────────────────────────────────────────

PREDICTORS = {
    # name → (function, kwargs)
    'unconditional_all':       (predict_unconditional, {'trail': 'all'}),
    'unconditional_5y':        (predict_unconditional, {'trail': '5y'}),
    'B_5y':                    (predict_method, {'method': 'cat_b', 'trail': '5y'}),
    'C_5y':                    (predict_method, {'method': 'cat_c', 'trail': '5y'}),
    'D_5y':                    (predict_method, {'method': 'cat_d', 'trail': '5y'}),
    'B_all':                   (predict_method, {'method': 'cat_b', 'trail': 'all'}),
    'C_all':                   (predict_method, {'method': 'cat_c', 'trail': 'all'}),
    'D_all':                   (predict_method, {'method': 'cat_d', 'trail': 'all'}),
    'union_5y':                (predict_combined, {'trail': '5y'}),
    'intersection_5y':         (predict_intersection, {'trail': '5y'}),
    # Per-index variants — fairer for index-specific models like NN
    'D_5y_per_idx':            (predict_method, {'method': 'cat_d', 'trail': '5y', 'per_index': True}),
    'C_5y_per_idx':            (predict_method, {'method': 'cat_c', 'trail': '5y', 'per_index': True}),
    'union_5y_per_idx':        (predict_combined, {'trail': '5y', 'per_index': True}),
}


def run_predictor(name: str, test_row, train: pd.DataFrame,
                  horizon: str) -> np.ndarray:
    fn, kwargs = PREDICTORS[name]
    return fn(test_row, train, horizon, **kwargs)


__all__ = [
    'PREDICTORS', 'run_predictor', 'TRAIL_DAYS',
    'predict_unconditional', 'predict_method',
    'predict_combined', 'predict_intersection',
]
