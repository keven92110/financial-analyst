"""Walk-forward evaluation harness.

Splits the panel into expanding train sets vs out-of-sample test periods, so
predictors are never evaluated on data they 'saw'.

Default schedule:
- Test period 1: train=[..2010-01-01], test=[2010-01-01..2011-01-01]
- Test period 2: train=[..2011-01-01], test=[2011-01-01..2012-01-01]
- ...
- Test period N: train=[..2025-01-01], test=[2025-01-01..]

This gives ~16 yearly windows × ~5 indices × ~250 trading days = ~20k test samples.
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import pandas as pd

from metrics import metrics_from_samples


# ─── Walk-forward iteration ──────────────────────────────────────────────

def walk_forward_splits(panel: pd.DataFrame,
                        start_year: int = 2010,
                        end_year: int = 2027,
                        step_years: int = 1):
    """Yield (train_df, test_df, cutoff) tuples."""
    for year in range(start_year, end_year, step_years):
        cutoff = pd.Timestamp(f'{year}-01-01')
        next_cutoff = pd.Timestamp(f'{year + step_years}-01-01')
        train = panel[panel['end_date'] < cutoff]
        test = panel[(panel['end_date'] >= cutoff) &
                     (panel['end_date'] < next_cutoff)]
        if len(train) < 100 or len(test) < 1:
            continue
        yield train, test, cutoff


# ─── Evaluation ──────────────────────────────────────────────────────────

def evaluate_predictor(panel: pd.DataFrame, predictor_fn: Callable,
                       horizon: str = 'fwd_5d',
                       start_year: int = 2010, end_year: int = 2027,
                       min_samples: int = 20,
                       progress: bool = True) -> pd.DataFrame:
    """Run walk-forward eval. Returns one row per test prediction.

    Args:
        panel: full window panel
        predictor_fn: f(test_row, train_df, horizon) -> np.ndarray of samples
        horizon: 'fwd_5d' or 'fwd_20d'
        min_samples: skip predictions with fewer than this many historical samples
                     (statistics are unreliable below this; counts as 'no prediction')
    """
    rows = []
    splits = list(walk_forward_splits(panel, start_year, end_year))
    for i, (train, test, cutoff) in enumerate(splits):
        if progress:
            print(f'  [{i+1:>2d}/{len(splits)}] cutoff={cutoff.date()}  '
                  f'train={len(train):>6,}  test={len(test):>5,}', flush=True)
        # Iterate test rows
        for _, row in test.iterrows():
            y_true = row.get(horizon)
            if y_true is None or (isinstance(y_true, float) and np.isnan(y_true)):
                continue
            samples = predictor_fn(row, train, horizon)
            if samples is None or len(samples) < min_samples:
                # Record as "no prediction" — important to track coverage
                rows.append({
                    'cutoff':    cutoff,
                    'end_date':  row['end_date'],
                    'index':     row['index'],
                    'horizon':   horizon,
                    'y_true':    float(y_true),
                    'n_samples': len(samples) if samples is not None else 0,
                    'covered':   False,
                    'mu': np.nan, 'sigma': np.nan, 'p_up': np.nan,
                    'mse': np.nan, 'mae': np.nan,
                    'nll': np.nan, 'crps': np.nan,
                    'hit': np.nan, 'hit_prob': np.nan,
                })
                continue
            m = metrics_from_samples(float(y_true), samples)
            rows.append({
                'cutoff':    cutoff,
                'end_date':  row['end_date'],
                'index':     row['index'],
                'horizon':   horizon,
                'y_true':    float(y_true),
                'n_samples': int(len(samples)),
                'covered':   True,
                **m,
            })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict:
    """Return mean metrics over rows where prediction was made (covered=True)."""
    sub = df[df['covered']]
    if len(sub) == 0:
        return {'n_predictions': 0, 'coverage': 0.0}
    return {
        'n_predictions': int(len(sub)),
        'coverage':      float(sub['covered'].mean() if len(df) else 0.0),
        'mse':           float(sub['mse'].mean()),
        'rmse':          float(np.sqrt(sub['mse'].mean())),
        'mae':           float(sub['mae'].mean()),
        'nll':           float(sub['nll'].mean()),
        'crps':          float(sub['crps'].mean()),
        'hit_rate':      float(sub['hit'].mean() * 100),
        'hit_prob_rate': float(sub['hit_prob'].mean() * 100),
        'avg_n_samples': float(sub['n_samples'].mean()),
    }


__all__ = ['walk_forward_splits', 'evaluate_predictor', 'aggregate']
