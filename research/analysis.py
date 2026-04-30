"""Build window panel, classify, compute forward returns, detect outliers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from classifiers import (
    WINDOW_DAYS, _window_return, _window_vol,
    classify_a, classify_b, classify_d, classify_e,
    extract_path_features, compute_terciles,
)


HORIZONS = [1, 5, 20]   # forward-return horizons in trading days
KMEANS_K = 6            # number of clusters for method C


# ─── Build window panel ────────────────────────────────────────────────────

def build_window_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Iterate over every (index, end-date) pair, build feature row.

    Returns DataFrame with columns:
        index, end_date, window_ret, window_vol, fwd_1d, fwd_5d, fwd_20d,
        path_norm_1..30 (for K-means), rsi14, dist_ma200_pct, half1_ret, half2_ret
    """
    rows = []

    for idx_name in panel.columns:
        series = panel[idx_name].dropna()
        # need 200 prior + 30 window + max(HORIZONS) forward
        min_idx = 200
        max_idx = len(series) - max(HORIZONS) - 1

        for i in range(min_idx, max_idx + 1):
            end_pos = i + WINDOW_DAYS - 1
            if end_pos > len(series) - max(HORIZONS) - 1:
                break
            window = series.iloc[i:i + WINDOW_DAYS]
            if len(window) < WINDOW_DAYS:
                continue
            history_for_ma = series.iloc[i + WINDOW_DAYS - 200 : i + WINDOW_DAYS]
            # Need 200 days ending at window-end. If i < 200 we already skipped.

            wret = _window_return(window)
            wvol = _window_vol(window)
            # forward returns
            end_idx = i + WINDOW_DAYS - 1
            last_close = series.iloc[end_idx]
            fwd = {}
            for h in HORIZONS:
                fwd_close = series.iloc[end_idx + h]
                fwd[f'fwd_{h}d'] = float(np.log(fwd_close / last_close))

            # Path features (normalized to start=1)
            path = (window / window.iloc[0]).values

            # RSI 14
            from classifiers import _rsi
            rsi = _rsi(window, period=14)

            # MA200 distance
            ma200 = history_for_ma.mean() if len(history_for_ma) == 200 else np.nan
            dist_ma = (last_close / ma200 - 1) * 100 if ma200 and not np.isnan(ma200) else np.nan

            # Halves (for shape)
            mid = WINDOW_DAYS // 2
            h1 = float(np.log(window.iloc[mid - 1] / window.iloc[0]))
            h2 = float(np.log(window.iloc[-1] / window.iloc[mid - 1]))

            row = {
                'index': idx_name,
                'end_date': window.index[-1],
                'window_ret': wret,
                'window_vol': wvol,
                'rsi14': rsi,
                'dist_ma200_pct': dist_ma,
                'half1_ret': h1,
                'half2_ret': h2,
                **fwd,
            }
            for k in range(WINDOW_DAYS):
                row[f'p{k}'] = path[k]
            rows.append(row)

    return pd.DataFrame(rows)


# ─── Apply classifications ─────────────────────────────────────────────────

def apply_classifiers(panel_df: pd.DataFrame, baseline_mask: pd.Series = None) -> pd.DataFrame:
    """Add columns: cat_a, cat_b, cat_c, cat_d, cat_e to panel_df.

    baseline_mask: which rows define the "baseline distribution" for
                   tercile cuts and K-means fitting. Default = all rows.
    """
    df = panel_df.copy()
    if baseline_mask is None:
        baseline_mask = pd.Series(True, index=df.index)

    base = df[baseline_mask]

    # ── Method A terciles
    ret_cuts = compute_terciles(base['window_ret'].values)
    vol_cuts = compute_terciles(base['window_vol'].values)
    df['cat_a'] = df.apply(
        lambda r: _label_a(r['window_ret'], r['window_vol'], ret_cuts, vol_cuts),
        axis=1
    )

    # ── Method B shape (no fitting needed)
    df['cat_b'] = df.apply(_label_b, axis=1)

    # ── Method C K-means on normalized path
    path_cols = [f'p{k}' for k in range(WINDOW_DAYS)]
    X = df[path_cols].values
    X_base = base[path_cols].values
    rename = None
    km = None
    if len(X_base) >= KMEANS_K:
        km = KMeans(n_clusters=KMEANS_K, random_state=42, n_init=10)
        km.fit(X_base)
        labels = km.predict(X)
        cluster_returns = []
        for c in range(KMEANS_K):
            mask = labels == c
            if mask.sum() == 0:
                cluster_returns.append((c, 0))
            else:
                cluster_returns.append((c, df.loc[mask, 'window_ret'].mean()))
        cluster_returns.sort(key=lambda x: x[1])
        rename = {old: f'C{new+1}' for new, (old, _) in enumerate(cluster_returns)}
        df['cat_c'] = pd.Series(labels, index=df.index).map(rename)
    else:
        df['cat_c'] = None

    # ── Method D RSI
    df['cat_d'] = df['rsi14'].apply(_label_d)

    # ── Method E MA200
    df['cat_e'] = df['dist_ma200_pct'].apply(_label_e)

    return df, {
        'ret_cuts': ret_cuts,
        'vol_cuts': vol_cuts,
        'kmeans': km,
        'kmeans_rename': rename,
    }


def _label_a(r, v, ret_cuts, vol_cuts):
    if np.isnan(v) or np.isnan(r):
        return None
    if   r < ret_cuts[0]: r_l = 'Down'
    elif r > ret_cuts[1]: r_l = 'Up'
    else:                 r_l = 'Flat'
    if   v < vol_cuts[0]: v_l = 'LowVol'
    elif v > vol_cuts[1]: v_l = 'HighVol'
    else:                 v_l = 'MidVol'
    return f'{r_l}/{v_l}'


def _label_b(row):
    h1 = row['half1_ret']
    h2 = row['half2_ret']
    total = h1 + h2
    T = 0.01
    if abs(h1) < T and abs(h2) < T:
        return 'Sideways'
    if h1 < -T and h2 > T: return 'V-bottom'
    if h1 > T and h2 < -T: return 'InvV-top'
    if h1 > 0 and h2 > 0 and total > T: return 'TrendUp'
    if h1 < 0 and h2 < 0 and total < -T: return 'TrendDown'
    return 'Sideways'


def _label_d(rsi):
    if np.isnan(rsi): return None
    if rsi < 30: return 'Oversold (RSI<30)'
    if rsi > 70: return 'Overbought (RSI>70)'
    if rsi < 50: return 'Neutral-Low'
    return 'Neutral-High'


def _label_e(dist):
    if np.isnan(dist): return None
    if dist < -5:  return '< MA200 -5%'
    if dist < 5:   return 'Near MA200 (±5%)'
    if dist < 15:  return 'Above MA200 +5~15%'
    return 'Above MA200 +15%'


# ─── Statistics per category ──────────────────────────────────────────────

def category_stats(df: pd.DataFrame, cat_col: str, fwd_col: str,
                   baseline_mean: float = None) -> pd.DataFrame:
    """For each category, compute:
        n, mean, median, std, hit_rate (% positive),
        t_stat (vs baseline_mean), p_value
    """
    if baseline_mean is None:
        baseline_mean = df[fwd_col].dropna().mean()

    out = []
    for cat, sub in df.groupby(cat_col, dropna=True):
        vals = sub[fwd_col].dropna().values
        if len(vals) < 10:
            continue
        mean = vals.mean()
        median = np.median(vals)
        std = vals.std()
        hit = (vals > 0).mean() * 100
        # t-test: is this category's mean different from baseline_mean?
        if std > 0:
            t_stat, p_val = stats.ttest_1samp(vals, baseline_mean)
        else:
            t_stat, p_val = np.nan, np.nan
        out.append({
            'category': cat,
            'n': len(vals),
            'mean_pct': mean * 100,        # convert log return to %
            'median_pct': median * 100,
            'std_pct': std * 100,
            'hit_rate': hit,
            't_stat': t_stat,
            'p_value': p_val,
            'mean_diff_pct': (mean - baseline_mean) * 100,
        })
    return pd.DataFrame(out).sort_values('mean_pct', ascending=False)


def flag_outliers(stats_df: pd.DataFrame,
                  p_thresh: float = 0.01,
                  hit_diff: float = 5.0,
                  baseline_hit: float = 50.0) -> pd.DataFrame:
    """Flag categories with statistically significant deviation."""
    df = stats_df.copy()
    df['flag_pvalue'] = df['p_value'] < p_thresh
    df['flag_hit'] = (df['hit_rate'] - baseline_hit).abs() > hit_diff
    df['flag'] = df['flag_pvalue'] | df['flag_hit']
    return df


# ─── Time range filtering ─────────────────────────────────────────────────

def filter_range(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Filter by end_date (the window's last day)."""
    sub = df
    if start is not None:
        sub = sub[sub['end_date'] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub['end_date'] <= pd.Timestamp(end)]
    return sub


def trailing_range(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Last N calendar days from the latest end_date."""
    last = df['end_date'].max()
    cutoff = last - pd.Timedelta(days=days)
    return df[df['end_date'] >= cutoff]


__all__ = [
    'HORIZONS', 'KMEANS_K',
    'build_window_panel', 'apply_classifiers',
    'category_stats', 'flag_outliers',
    'filter_range', 'trailing_range',
]
