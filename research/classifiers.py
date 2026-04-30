"""Classification methods for 30-day windows.

Each classifier takes a window (DataFrame slice) plus optional pre-context
and returns a category label (string). Categories are intentionally
human-readable so plots are interpretable.

Conventions:
- `window` = DataFrame with one column 'Adj Close' over WINDOW_DAYS rows
- `prior` = DataFrame with longer history (for method E, 200-day MA)
- All return `None` if data is insufficient.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOW_DAYS = 30


# ─── Helpers ──────────────────────────────────────────────────────────────

def _log_returns(closes: pd.Series) -> pd.Series:
    return np.log(closes / closes.shift(1)).dropna()


def _window_return(closes: pd.Series) -> float:
    """Total log return over the window."""
    return float(np.log(closes.iloc[-1] / closes.iloc[0]))


def _window_vol(closes: pd.Series) -> float:
    """Daily return std (annualized)."""
    rets = _log_returns(closes)
    if len(rets) < 5:
        return np.nan
    return float(rets.std() * np.sqrt(252))


# ─── Method A: Return × Volatility (3×3 grid → 9 cells) ──────────────────

def classify_a(closes: pd.Series, ret_thresholds: tuple, vol_thresholds: tuple) -> str:
    """3x3 grid: return tercile × volatility tercile.

    ret_thresholds = (low_cut, high_cut) — terciles from baseline distribution
    vol_thresholds = (low_cut, high_cut)
    """
    r = _window_return(closes)
    v = _window_vol(closes)
    if np.isnan(v):
        return None

    if   r < ret_thresholds[0]: r_lbl = 'Down'
    elif r > ret_thresholds[1]: r_lbl = 'Up'
    else:                       r_lbl = 'Flat'

    if   v < vol_thresholds[0]: v_lbl = 'LowVol'
    elif v > vol_thresholds[1]: v_lbl = 'HighVol'
    else:                       v_lbl = 'MidVol'

    return f'{r_lbl}/{v_lbl}'


# ─── Method B: Shape patterns ─────────────────────────────────────────────

def classify_b(closes: pd.Series) -> str:
    """Shape: V / inverted-V / TrendUp / TrendDown / Sideways.

    Heuristic:
    - Split window into halves, compute each half's return.
    - Magnitude threshold: |half_ret| > 1% qualifies as 'moved'.
    - V: H1 down + H2 up (with magnitude); InvV: H1 up + H2 down.
    - TrendUp: both halves up. TrendDown: both halves down.
    - Else: Sideways.
    """
    n = len(closes)
    mid = n // 2
    h1 = float(np.log(closes.iloc[mid - 1] / closes.iloc[0]))
    h2 = float(np.log(closes.iloc[-1] / closes.iloc[mid - 1]))
    total = h1 + h2

    THRESH = 0.01  # 1% per half to count as "moved"

    if abs(h1) < THRESH and abs(h2) < THRESH:
        return 'Sideways'
    if h1 < -THRESH and h2 > THRESH:
        return 'V-bottom'
    if h1 > THRESH and h2 < -THRESH:
        return 'InvV-top'
    if h1 > 0 and h2 > 0 and total > THRESH:
        return 'TrendUp'
    if h1 < 0 and h2 < 0 and total < -THRESH:
        return 'TrendDown'
    # Mixed but not extreme enough
    return 'Sideways'


# ─── Method C: K-means on normalized path ─────────────────────────────────

def extract_path_features(closes: pd.Series) -> np.ndarray:
    """Normalize 30-day path to start=1, return array of 30 values."""
    return (closes / closes.iloc[0]).values


# K-means is fit once on all windows; classifier just assigns label.
# This is handled in analysis.py (need global fitting). Here we just
# expose the feature extractor.


# ─── Method D: RSI / streak ────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> float:
    """Wilder's RSI at the last bar."""
    deltas = closes.diff().dropna()
    if len(deltas) < period:
        return np.nan
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.iloc[-period:].mean()
    avg_loss = losses.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def classify_d(closes: pd.Series) -> str:
    """RSI bucket at end of window: Oversold / Neutral / Overbought."""
    rsi = _rsi(closes, period=14)
    if np.isnan(rsi):
        return None
    if rsi < 30: return 'Oversold (RSI<30)'
    if rsi > 70: return 'Overbought (RSI>70)'
    if rsi < 50: return 'Neutral-Low'
    return 'Neutral-High'


# ─── Method E: Distance from 200-day MA ────────────────────────────────────

def classify_e(closes_with_history: pd.Series) -> str:
    """Distance from 200MA at the end of the window.

    closes_with_history must include >=200 days BEFORE the window's last day.
    Categories: Below5+ / Near (±5%) / Above5-15 / Above15+
    """
    if len(closes_with_history) < 200:
        return None
    ma200 = closes_with_history.rolling(200).mean().iloc[-1]
    last = closes_with_history.iloc[-1]
    if np.isnan(ma200):
        return None
    pct = (last / ma200 - 1) * 100
    if pct < -5:  return '< MA200 -5%'
    if pct < 5:   return 'Near MA200 (±5%)'
    if pct < 15:  return 'Above MA200 +5~15%'
    return 'Above MA200 +15%'


# ─── Tercile threshold computation (for method A) ─────────────────────────

def compute_terciles(values: np.ndarray) -> tuple:
    """Return (33%, 67%) percentiles, ignoring NaNs."""
    arr = values[~np.isnan(values)]
    return (float(np.percentile(arr, 33.33)), float(np.percentile(arr, 66.67)))


__all__ = [
    'WINDOW_DAYS',
    '_window_return', '_window_vol',
    'classify_a', 'classify_b', 'classify_d', 'classify_e',
    'extract_path_features', 'compute_terciles',
]
