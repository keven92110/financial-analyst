"""Predict / lookup module for the desktop app.

- Load saved classifier bundle (K-means + tercile cuts)
- Classify today's 30-day windows for each of the 5 indices
- Look up historical fwd_5d / fwd_20d distributions for each (index, method, category, time-range) combination
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_all, build_returns_panel, INDICES
from classifiers import _window_return, _window_vol, _rsi, WINDOW_DAYS

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models'
OUTPUT_DIR = ROOT / 'output'

METHOD_LABELS = {'cat_b': 'B · Shape', 'cat_c': 'C · K-means', 'cat_d': 'D · RSI'}
ALL_METHODS = ['cat_b', 'cat_c', 'cat_d']
INDEX_NAMES = list(INDICES.keys())  # all underlyings (US 5 + TWII + ...)
US_INDEX_NAMES = ['SPX', 'DJI', 'NDX', 'RUT', 'SOX']   # ALL_5IDX aggregates these
ALL_INDEX = 'ALL_5IDX'   # label kept; aggregates US_INDEX_NAMES only


# ─── Bundle loading ──────────────────────────────────────────────────────

_bundle_cache = None
_panel_cache = None


def load_bundle() -> dict:
    """Load saved classifier bundle (K-means model + tercile cuts)."""
    global _bundle_cache
    if _bundle_cache is None:
        path = MODELS_DIR / 'classifier_bundle.pkl'
        with open(path, 'rb') as f:
            _bundle_cache = pickle.load(f)
    return _bundle_cache


def load_panel() -> pd.DataFrame:
    """Load the historical window panel CSV (cached)."""
    global _panel_cache
    if _panel_cache is None:
        df = pd.read_csv(OUTPUT_DIR / 'window_panel.csv', parse_dates=['end_date'])
        _panel_cache = df
    return _panel_cache


def reset_caches():
    global _bundle_cache, _panel_cache
    _bundle_cache = None
    _panel_cache = None


# ─── Classification of a single fresh window ──────────────────────────────

def classify_window(closes: pd.Series, bundle: dict) -> dict:
    """Classify a 30-day price window using B/C/D methods.

    closes: pd.Series of length WINDOW_DAYS, indexed by date.
    Returns: {'cat_b': str, 'cat_c': str, 'cat_d': str, 'window_ret': float, ...}
    """
    if len(closes) < WINDOW_DAYS:
        return None

    closes = closes.iloc[-WINDOW_DAYS:].copy()
    wret = _window_return(closes)
    wvol = _window_vol(closes)

    # Method B: shape
    mid = WINDOW_DAYS // 2
    h1 = float(np.log(closes.iloc[mid - 1] / closes.iloc[0]))
    h2 = float(np.log(closes.iloc[-1] / closes.iloc[mid - 1]))
    total = h1 + h2
    T = 0.01
    if abs(h1) < T and abs(h2) < T:
        cat_b = 'Sideways'
    elif h1 < -T and h2 > T:
        cat_b = 'V-bottom'
    elif h1 > T and h2 < -T:
        cat_b = 'InvV-top'
    elif h1 > 0 and h2 > 0 and total > T:
        cat_b = 'TrendUp'
    elif h1 < 0 and h2 < 0 and total < -T:
        cat_b = 'TrendDown'
    else:
        cat_b = 'Sideways'

    # Method C: K-means
    cat_c = None
    km = bundle.get('kmeans')
    rename = bundle.get('kmeans_rename')
    if km is not None and rename is not None:
        path = (closes / closes.iloc[0]).values.reshape(1, -1)
        raw_label = int(km.predict(path)[0])
        cat_c = rename.get(raw_label)

    # Method D: RSI
    rsi = _rsi(closes, period=14)
    if np.isnan(rsi):
        cat_d = None
    elif rsi < 30:
        cat_d = 'Oversold (RSI<30)'
    elif rsi > 70:
        cat_d = 'Overbought (RSI>70)'
    elif rsi < 50:
        cat_d = 'Neutral-Low'
    else:
        cat_d = 'Neutral-High'

    return {
        'cat_b': cat_b, 'cat_c': cat_c, 'cat_d': cat_d,
        'window_ret_pct': wret * 100,
        'window_vol_pct': wvol * 100,
        'rsi14': rsi,
        'path': (closes / closes.iloc[0]).values,
        'dates': closes.index.tolist(),
        'first_close': float(closes.iloc[0]),
        'last_close': float(closes.iloc[-1]),
    }


# ─── Today's signals ──────────────────────────────────────────────────────

def get_today_signals(refresh: bool = False) -> dict:
    """Compute today's classification + historical-distribution prediction
    for each of the 5 indices (and ALL_5IDX as combined).

    Returns:
    {
        'as_of_date': pd.Timestamp,
        'indices': {
            'SPX': {
                'classification': {...},  # from classify_window
                'predictions': {  # from historical lookup
                    'cat_b': {'fwd_5d': {...}, 'fwd_20d': {...}},
                    ...
                }
            },
            ...,
            'ALL_5IDX': {...},  # uses each index's own window
        }
    }
    """
    bundle = load_bundle()
    panel = load_panel()

    # Fetch fresh data
    raw = load_all(refresh=refresh)
    panel_close = build_returns_panel(raw)

    # Find latest common date with non-null
    last_dates = {idx: panel_close[idx].dropna().index.max()
                  for idx in INDEX_NAMES}
    as_of = max(last_dates.values())

    out = {'as_of_date': as_of, 'last_date_per_index': last_dates, 'indices': {}}

    # Per-index classification
    for idx_name in INDEX_NAMES:
        series = panel_close[idx_name].dropna()
        if len(series) < WINDOW_DAYS:
            continue
        win = series.iloc[-WINDOW_DAYS:]
        cls = classify_window(win, bundle)
        if cls is None:
            continue
        preds = lookup_predictions(panel, idx_name, cls)
        out['indices'][idx_name] = {'classification': cls, 'predictions': preds}

    # ALL_5IDX: aggregate predictions across the 5 indices using each one's category
    all_preds = aggregate_all5idx_predictions(panel, out['indices'])
    out['indices'][ALL_INDEX] = {
        'classification': None,  # there isn't a single "today's" classification for the combined view
        'predictions': all_preds,
    }
    return out


def lookup_predictions(panel: pd.DataFrame, index_name: str,
                       classification: dict) -> dict:
    """Given today's classification, look up the historical distribution.

    Uses **full history of the index** for the matching category.
    Returns: {method: {fwd_5d: {n, mean, median, hit, std, vals},
                       fwd_20d: {...}}}
    """
    sub = panel[panel['index'] == index_name]
    out = {}
    for method in ALL_METHODS:
        cat = classification.get(method)
        if cat is None:
            out[method] = None; continue
        match = sub[sub[method] == cat]
        # Drop NaNs in fwd cols
        fwd5 = match['fwd_5d'].dropna().values
        fwd20 = match['fwd_20d'].dropna().values
        out[method] = {
            'category': cat,
            'fwd_5d':  _summarize(fwd5),
            'fwd_20d': _summarize(fwd20),
        }
    return out


def aggregate_all5idx_predictions(panel: pd.DataFrame, per_idx: dict) -> dict:
    """For ALL_5IDX, pool the historical match-windows across the 5 US indices,
    each using their own today-classification.

    TWII (and any other non-US underlying) is excluded from the pool so the
    'ALL_5IDX' label remains semantically a US-equity-aggregate.
    """
    out = {}
    for method in ALL_METHODS:
        all_5d = []
        all_20d = []
        cat_label_parts = []
        for idx_name in US_INDEX_NAMES:
            entry = per_idx.get(idx_name)
            if entry is None: continue
            cat = entry['classification'].get(method)
            if cat is None: continue
            cat_label_parts.append(f'{idx_name}={cat}')
            sub = panel[(panel['index'] == idx_name) & (panel[method] == cat)]
            all_5d.append(sub['fwd_5d'].dropna().values)
            all_20d.append(sub['fwd_20d'].dropna().values)
        if not all_5d:
            out[method] = None; continue
        f5 = np.concatenate(all_5d) if all_5d else np.array([])
        f20 = np.concatenate(all_20d) if all_20d else np.array([])
        out[method] = {
            'category': ' + '.join(cat_label_parts),
            'fwd_5d':  _summarize(f5),
            'fwd_20d': _summarize(f20),
        }
    return out


def _summarize(vals: np.ndarray) -> dict:
    if vals is None or len(vals) == 0:
        return {'n': 0, 'mean_pct': np.nan, 'median_pct': np.nan,
                'std_pct': np.nan, 'hit_rate': np.nan, 'vals': vals}
    return {
        'n': int(len(vals)),
        'mean_pct': float(vals.mean() * 100),
        'median_pct': float(np.median(vals) * 100),
        'std_pct': float(vals.std() * 100),
        'hit_rate': float((vals > 0).mean() * 100),
        'p10_pct': float(np.percentile(vals, 10) * 100),
        'p90_pct': float(np.percentile(vals, 90) * 100),
        'vals': vals,
    }


# ─── Backtest queries ─────────────────────────────────────────────────────

BACKTEST_RANGES = [
    ('6M',  180),
    ('12M', 365),
    ('2Y',  365 * 2),
    ('5Y',  365 * 5),
    ('10Y', 365 * 10),
    ('20Y', 365 * 20),
    ('Full', None),
]


def get_categories_for_method(panel: pd.DataFrame, method: str,
                              index_name: Optional[str] = None) -> list:
    """Return sorted list of category labels for a method (per-index or global)."""
    sub = panel if index_name is None or index_name == ALL_INDEX \
        else panel[panel['index'] == index_name]
    cats = sub[method].dropna().unique().tolist()
    # Sort sensibly
    if method == 'cat_c':
        cats.sort(key=lambda x: int(x[1:]))   # C1 < C2 < ...
    else:
        cats.sort()
    return cats


def get_backtest_distributions(method: str, category: str,
                               index_name: str, horizon: str) -> dict:
    """Return dict {range_label: {n, mean_pct, ..., vals}}.

    horizon = 'fwd_5d' or 'fwd_20d'.
    index_name = ALL_5IDX or specific.
    """
    panel = load_panel()
    sub = panel if index_name == ALL_INDEX \
        else panel[panel['index'] == index_name]
    sub = sub[sub[method] == category]

    if len(sub) == 0:
        return {label: _summarize(np.array([])) for label, _ in BACKTEST_RANGES}

    # Latest available end_date
    last = sub['end_date'].max()

    out = {}
    for label, days in BACKTEST_RANGES:
        if days is None:
            slice_ = sub
        else:
            cutoff = last - pd.Timedelta(days=days)
            slice_ = sub[sub['end_date'] >= cutoff]
        vals = slice_[horizon].dropna().values
        out[label] = _summarize(vals)
    return out


def get_today_distributions_by_range(method: str, index_name: str,
                                     today_data: dict) -> tuple:
    """For today's classification of `index_name` under `method`,
    return per-range historical distributions for BOTH fwd_5d and fwd_20d.

    For ALL_5IDX, pools windows across all 5 indices, each using its own
    today-classification for the chosen method.

    Returns:
        (
            distributions: { range_label: { 'fwd_5d': summarize_dict,
                                            'fwd_20d': summarize_dict } },
            category_label: str   # human-readable; for ALL_5IDX it's compound
        )
    """
    panel = load_panel()
    empty = {label: {'fwd_5d': _summarize(np.array([])),
                     'fwd_20d': _summarize(np.array([]))}
             for label, _ in BACKTEST_RANGES}

    if index_name == ALL_INDEX:
        per_idx_cats = {}
        for nm in US_INDEX_NAMES:
            entry = today_data.get('indices', {}).get(nm)
            if entry is None:
                continue
            cls = entry.get('classification') or {}
            cat = cls.get(method)
            if cat is None:
                continue
            per_idx_cats[nm] = cat
        if not per_idx_cats:
            return empty, None
        # Pool: each index contributes its own (idx, cat)-matched windows
        parts = []
        for nm, cat in per_idx_cats.items():
            sub = panel[(panel['index'] == nm) & (panel[method] == cat)]
            parts.append(sub)
        big = pd.concat(parts)
        if len(big) == 0:
            return empty, None
        last = big['end_date'].max()
        out = {}
        for label, days in BACKTEST_RANGES:
            slc = big if days is None else \
                big[big['end_date'] >= (last - pd.Timedelta(days=days))]
            out[label] = {
                'fwd_5d':  _summarize(slc['fwd_5d'].dropna().values),
                'fwd_20d': _summarize(slc['fwd_20d'].dropna().values),
            }
        cat_label = ' + '.join(f'{nm}={c}' for nm, c in per_idx_cats.items())
        return out, cat_label

    # Individual index
    entry = today_data.get('indices', {}).get(index_name)
    if entry is None:
        return empty, None
    cls = entry.get('classification') or {}
    cat = cls.get(method)
    if cat is None:
        return empty, None
    sub = panel[(panel['index'] == index_name) & (panel[method] == cat)]
    if len(sub) == 0:
        return empty, cat
    last = sub['end_date'].max()
    out = {}
    for label, days in BACKTEST_RANGES:
        slc = sub if days is None else \
            sub[sub['end_date'] >= (last - pd.Timedelta(days=days))]
        out[label] = {
            'fwd_5d':  _summarize(slc['fwd_5d'].dropna().values),
            'fwd_20d': _summarize(slc['fwd_20d'].dropna().values),
        }
    return out, cat


__all__ = [
    'METHOD_LABELS', 'ALL_METHODS',
    'INDEX_NAMES', 'US_INDEX_NAMES', 'ALL_INDEX',
    'BACKTEST_RANGES',
    'load_bundle', 'load_panel', 'reset_caches',
    'classify_window', 'get_today_signals',
    'get_categories_for_method', 'get_backtest_distributions',
    'get_today_distributions_by_range',
]


# ─── Quick CLI test ──────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Today Signals ===')
    sig = get_today_signals()
    print(f'as_of_date: {sig["as_of_date"]}')
    for idx in INDEX_NAMES + [ALL_INDEX]:
        entry = sig['indices'].get(idx)
        if entry is None: continue
        cls = entry['classification']
        if cls is not None:
            print(f'\n{idx}  (window {cls["dates"][0].date()} → {cls["dates"][-1].date()},  '
                  f'30d ret={cls["window_ret_pct"]:+.2f}%)')
            print(f"  B={cls['cat_b']}, C={cls['cat_c']}, "
                  f"D={cls['cat_d']} (RSI={cls['rsi14']:.1f})")
        else:
            print(f'\n{idx}  (combined view)')
        for method in ALL_METHODS:
            p = entry['predictions'].get(method)
            if p is None: continue
            f5 = p['fwd_5d']; f20 = p['fwd_20d']
            cat = p['category']
            print(f"  {method}={cat}")
            if f5['n']:
                print(f"     fwd_5d  N={f5['n']:>5,}  μ={f5['mean_pct']:+.2f}%  hit={f5['hit_rate']:.1f}%")
            if f20['n']:
                print(f"     fwd_20d N={f20['n']:>5,}  μ={f20['mean_pct']:+.2f}%  hit={f20['hit_rate']:.1f}%")

    print('\n\n=== Backtest sample: D · Oversold (RSI<30) · ALL_5IDX · fwd_20d ===')
    bt = get_backtest_distributions('cat_d', 'Oversold (RSI<30)', ALL_INDEX, 'fwd_20d')
    for label, stats in bt.items():
        flag = '⚠️' if 0 < stats['n'] < 30 else '  '
        print(f'  {flag} {label:5s}  N={stats["n"]:>5,}  μ={stats["mean_pct"]:+.2f}%  '
              f'hit={stats["hit_rate"]:.1f}%')
