"""Compare our model's empirical σ to market-implied vol (CBOE indices).

Conversion notes
----------------
Our σ values come from log-return std over N trading days (decimal).
Market IV is annualized in percent.

To compare on the same axis (annualized %):
    σ_annualized_pct = σ_Nd_decimal × sqrt(252 / N) × 100

Variance Risk Premium (VRP):
    VRP = IV - σ_emp     (annualized %, both)
    VRP > 0  → market expects MORE vol than history of matching cases
              → potential "sell vol" edge IF our model is right
    VRP < 0  → market under-estimates  → "buy vol"
              (rarer; usually means crisis approaching)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))   # research/

from iv.iv_fetcher import fetch_current_iv, fetch_history, VOL_INDEX_30D


# ─── Annualization ────────────────────────────────────────────────────────

def annualize_sigma(sigma_decimal: float, days: int) -> float:
    """σ over `days` trading days (decimal log return std) → annualized %."""
    if sigma_decimal is None or np.isnan(sigma_decimal):
        return float('nan')
    return float(sigma_decimal * np.sqrt(252 / days) * 100)


def deannualize_iv(iv_pct: float, days: int) -> float:
    """Annualized IV % → expected std (decimal) over `days` trading days."""
    if iv_pct is None or np.isnan(iv_pct):
        return float('nan')
    return float(iv_pct / 100 * np.sqrt(days / 252))


# ─── Today snapshot ───────────────────────────────────────────────────────

def compare_today(today_data: dict) -> pd.DataFrame:
    """Compare today's empirical σ (from D=RSI method) to market IV.

    today_data: dict from predict.get_today_signals().
    Returns DataFrame with columns:
        index, method_cat (cat_d), n_5d, sigma_emp_5d_pct, iv_9d_pct, vrp_5d_pct,
        n_20d, sigma_emp_20d_pct, iv_30d_pct, vrp_20d_pct
    Uses cat_d (RSI bucket) historical full-period match for σ_emp.
    """
    rows = []
    for idx_name in ['SPX', 'NDX', 'DJI', 'RUT', 'SOX', 'TWII']:
        entry = today_data.get('indices', {}).get(idx_name)
        if entry is None: continue
        cls = entry.get('classification') or {}
        preds = entry.get('predictions', {})

        cat_d = cls.get('cat_d')
        f5 = preds.get('cat_d', {}).get('fwd_5d', {}) if cat_d else {}
        f20 = preds.get('cat_d', {}).get('fwd_20d', {}) if cat_d else {}
        # σ_emp comes from std of historical fwd returns (in decimal)
        # std_pct from _summarize is already in % (i.e. y_pct = y * 100)
        sig_emp_5d_dec  = (f5.get('std_pct')  or 0) / 100 if f5 else None
        sig_emp_20d_dec = (f20.get('std_pct') or 0) / 100 if f20 else None

        sig_emp_5d_ann  = annualize_sigma(sig_emp_5d_dec, 5)  if sig_emp_5d_dec  else float('nan')
        sig_emp_20d_ann = annualize_sigma(sig_emp_20d_dec, 20) if sig_emp_20d_dec else float('nan')

        iv9  = fetch_current_iv(idx_name, '9d')
        iv30 = fetch_current_iv(idx_name, '30d')
        iv9_pct  = iv9['iv_pct']  if iv9['available']  else float('nan')
        iv30_pct = iv30['iv_pct'] if iv30['available'] else float('nan')

        vrp_5d  = (iv9_pct  - sig_emp_5d_ann)  if (not np.isnan(iv9_pct)  and not np.isnan(sig_emp_5d_ann))  else float('nan')
        vrp_20d = (iv30_pct - sig_emp_20d_ann) if (not np.isnan(iv30_pct) and not np.isnan(sig_emp_20d_ann)) else float('nan')

        rows.append({
            'index':              idx_name,
            'cat_d':              cat_d or '—',
            'n_5d':               f5.get('n', 0)  if f5 else 0,
            'sigma_emp_5d_pct':   sig_emp_5d_ann,
            'iv_9d_pct':          iv9_pct,
            'vrp_5d_pct':         vrp_5d,
            'n_20d':              f20.get('n', 0) if f20 else 0,
            'sigma_emp_20d_pct':  sig_emp_20d_ann,
            'iv_30d_pct':         iv30_pct,
            'vrp_20d_pct':        vrp_20d,
        })
    return pd.DataFrame(rows)


# ─── Historical alignment (for backtest) ─────────────────────────────────

def align_panel_with_vix(panel: pd.DataFrame, idx_name: str = 'SPX',
                         vix_symbol: str = '^VIX') -> pd.DataFrame:
    """For each window in panel matching `idx_name`, attach the VIX value at end_date.

    Returns the subset with both VIX and fwd_20d available.
    """
    sub = panel[panel['index'] == idx_name].copy()
    vix = fetch_history(vix_symbol)
    sub['end_date'] = pd.to_datetime(sub['end_date'])
    vix.index = pd.to_datetime(vix.index)
    sub = sub.merge(vix.rename('vix'), left_on='end_date', right_index=True,
                    how='left')
    return sub


__all__ = [
    'annualize_sigma', 'deannualize_iv',
    'compare_today', 'align_panel_with_vix',
]
