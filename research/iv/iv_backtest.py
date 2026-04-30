"""Historical backtest: does our σ_emp identify periods where market IV is mispriced?

Methodology
-----------
For each historical window (SPX, since VIX history covers it):

1. Compute σ_emp = std of historical fwd_20d log-returns for matching cat_d
   (RSI bucket), using a 5-year trailing window of data BEFORE the test date.
   → annualized %
2. Get VIX (= market IV) at end_date.
3. VRP_implied = VIX - σ_emp_annualized
4. Compute realized vol over fwd 20 days (from raw daily returns).
5. RV_diff = VIX - RV_realized   (the actual "premium" you would have collected
   if you sold variance at VIX and paid out at RV)

Outputs
-------
- Time series of VIX, σ_emp, RV
- Scatter: σ_emp vs VIX
- Strategy P&L: when VRP_implied > threshold, "sell vol" → measure realized P&L
  as proportional to (VIX^2 - RV^2) (variance swap analog) over 20d periods
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))   # research/

from data_loader import load_all
from iv.iv_fetcher import fetch_history
from iv.iv_compare import annualize_sigma


PANEL_PATH = Path(__file__).parent.parent / 'output' / 'window_panel.csv'
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ─── Compute σ_emp via walk-forward (no leakage) ─────────────────────────

def compute_sigma_emp(panel: pd.DataFrame, idx_name: str = 'SPX',
                      method: str = 'cat_d',
                      trail_days: int = 365 * 5,
                      min_samples: int = 30) -> pd.DataFrame:
    """For each row in panel(idx_name), compute σ_emp using prior-only data
    matching the row's `method` category over a trailing window."""
    sub = panel[panel['index'] == idx_name].copy().sort_values('end_date')
    sub['end_date'] = pd.to_datetime(sub['end_date'])

    sigma_5d = np.full(len(sub), np.nan)
    sigma_20d = np.full(len(sub), np.nan)
    n_5d = np.zeros(len(sub), dtype=int)
    n_20d = np.zeros(len(sub), dtype=int)

    # Pre-filter: same method, same index
    end_dates = sub['end_date'].values
    cats     = sub[method].values
    f5_arr   = sub['fwd_5d'].values
    f20_arr  = sub['fwd_20d'].values

    for i in range(len(sub)):
        if pd.isna(cats[i]):
            continue
        ed = end_dates[i]
        cutoff = ed - pd.Timedelta(days=trail_days)
        # match same category, end_date < ed AND end_date >= cutoff
        mask = ((end_dates < ed) & (end_dates >= cutoff) & (cats == cats[i]))
        if not mask.any():
            continue
        f5 = f5_arr[mask]
        f20 = f20_arr[mask]
        f5 = f5[~np.isnan(f5)]
        f20 = f20[~np.isnan(f20)]
        if len(f5) >= min_samples:
            sigma_5d[i] = np.std(f5)
        if len(f20) >= min_samples:
            sigma_20d[i] = np.std(f20)
        n_5d[i] = len(f5)
        n_20d[i] = len(f20)

    sub['sigma_5d_dec']   = sigma_5d
    sub['sigma_20d_dec']  = sigma_20d
    sub['sigma_5d_ann']   = sub['sigma_5d_dec']  * np.sqrt(252 / 5)  * 100
    sub['sigma_20d_ann']  = sub['sigma_20d_dec'] * np.sqrt(252 / 20) * 100
    sub['n_5d_match']     = n_5d
    sub['n_20d_match']    = n_20d
    return sub


# ─── Realized vol (forward 20d) ──────────────────────────────────────────

def compute_realized_vol(closes: pd.Series, end_dates: pd.DatetimeIndex,
                         days: int = 20) -> pd.Series:
    """For each end_date, compute realized vol of next `days` log returns
    (annualized %)."""
    log_ret = np.log(closes / closes.shift(1))
    rv = pd.Series(np.nan, index=end_dates)
    closes_idx = closes.index
    for ed in end_dates:
        if ed not in closes_idx:
            continue
        pos = closes_idx.get_loc(ed)
        future = log_ret.iloc[pos + 1: pos + 1 + days]
        if len(future) < days // 2:
            continue
        rv.loc[ed] = float(future.std() * np.sqrt(252) * 100)
    return rv


# ─── Main backtest ───────────────────────────────────────────────────────

def run():
    print('Loading panel...')
    panel = pd.read_csv(PANEL_PATH, parse_dates=['end_date'])
    print(f'  {len(panel):,} rows')

    print('Computing walk-forward σ_emp (SPX, cat_d, trailing 5Y)...')
    sub = compute_sigma_emp(panel, 'SPX', method='cat_d',
                            trail_days=365 * 5, min_samples=30)
    print(f'  rows with σ_5d:  {sub["sigma_5d_ann"].notna().sum():,}')
    print(f'  rows with σ_20d: {sub["sigma_20d_ann"].notna().sum():,}')

    print('Loading VIX...')
    vix = fetch_history('^VIX')
    sub = sub.merge(vix.rename('vix'), left_on='end_date', right_index=True,
                    how='left')
    print(f'  rows with vix:   {sub["vix"].notna().sum():,}')

    print('Computing realized 20-day vol from SPX prices...')
    raw = load_all()
    spx_close = raw['SPX']['Adj Close']
    sub['rv_20d'] = compute_realized_vol(spx_close, sub['end_date'], 20).values

    # ── Compute spreads
    sub['vrp_implied'] = sub['vix'] - sub['sigma_20d_ann']  # market - emp model
    sub['vrp_realized'] = sub['vix'] - sub['rv_20d']        # market - realized

    # ── Save
    cols = ['end_date', 'cat_d', 'sigma_5d_ann', 'sigma_20d_ann',
            'n_5d_match', 'n_20d_match', 'vix', 'rv_20d',
            'vrp_implied', 'vrp_realized']
    sub[cols].to_csv(RESULTS_DIR / 'iv_backtest_spx.csv', index=False)

    # ── Stats
    valid = sub.dropna(subset=['vix', 'sigma_20d_ann', 'rv_20d'])
    print(f'\nSamples with all metrics: {len(valid):,}')
    print(f'  σ_emp_20d_ann:  mean={valid["sigma_20d_ann"].mean():.2f}%, '
          f'std={valid["sigma_20d_ann"].std():.2f}%')
    print(f'  VIX:            mean={valid["vix"].mean():.2f}%, '
          f'std={valid["vix"].std():.2f}%')
    print(f'  RV (20d):       mean={valid["rv_20d"].mean():.2f}%, '
          f'std={valid["rv_20d"].std():.2f}%')
    print(f'  VRP (VIX - σ_emp):     mean={valid["vrp_implied"].mean():+.2f}%, '
          f'positive {(valid["vrp_implied"] > 0).mean()*100:.1f}% of days')
    print(f'  VRP (VIX - RV):        mean={valid["vrp_realized"].mean():+.2f}%, '
          f'positive {(valid["vrp_realized"] > 0).mean()*100:.1f}% of days')
    corr = valid[['sigma_20d_ann', 'vix', 'rv_20d']].corr()
    print('\nCorrelations:')
    print(corr.to_string(float_format=lambda x: f'{x:.3f}'))

    # ── Strategy: when VRP_implied > threshold, "sell vol"
    # P&L proxy: variance swap → P&L ≈ VIX^2 - RV^2  (in annualized var units)
    print('\n══ Hypothetical "sell vol" strategy ══')
    print('Trigger: VRP_implied > threshold → sell 20d variance at VIX, pay RV.')
    print('P&L proxy (annualized vol^2):  VIX^2 - RV^2  (each row = 1 trade)\n')

    valid['pnl_var'] = (valid['vix'] ** 2 - valid['rv_20d'] ** 2) / 1e4  # in (decimal vol)^2
    for thresh in [-100, 0, 2, 5, 8]:
        sig = valid[valid['vrp_implied'] > thresh]
        if len(sig) == 0: continue
        win_rate = (sig['pnl_var'] > 0).mean() * 100
        avg_pnl = sig['pnl_var'].mean() * 1e4   # back to %^2 for readability
        worst = sig['pnl_var'].min() * 1e4
        sharpe = sig['pnl_var'].mean() / sig['pnl_var'].std() if sig['pnl_var'].std() > 0 else 0
        print(f'  threshold VRP > {thresh:>4d}:  N={len(sig):>5,}  '
              f'win {win_rate:5.1f}%  avg_pnl={avg_pnl:+7.2f}%^2  '
              f'worst={worst:+8.2f}%^2  Sharpe={sharpe:+.2f}')

    # ── Plots
    plot_timeseries(valid, RESULTS_DIR / 'iv_timeseries.png')
    plot_scatter(valid, RESULTS_DIR / 'iv_scatter.png')
    plot_strategy(valid, RESULTS_DIR / 'iv_strategy.png')
    print(f'\nOutputs in {RESULTS_DIR}')


def plot_timeseries(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['end_date'], df['vix'], color='#d62728', lw=0.9,
            label='VIX (market IV, 30d)', alpha=0.85)
    ax.plot(df['end_date'], df['sigma_20d_ann'], color='#1f77b4', lw=0.9,
            label='σ_emp (our model, 20d annualized)', alpha=0.85)
    ax.plot(df['end_date'], df['rv_20d'], color='#2ca02c', lw=0.5,
            label='RV (realized 20d)', alpha=0.4)
    ax.set_ylabel('Annualized vol (%)')
    ax.set_title('SPX volatility: market IV (VIX) vs our σ_emp vs realized',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # σ_emp vs VIX
    ax = axes[0]
    ax.scatter(df['sigma_20d_ann'], df['vix'], s=4, alpha=0.18, color='#4575b4')
    lim = max(df['sigma_20d_ann'].max(), df['vix'].max()) * 1.05
    ax.plot([0, lim], [0, lim], color='gray', ls='--', lw=0.8, label='y=x')
    ax.set_xlabel('σ_emp (our model, annualized %)')
    ax.set_ylabel('VIX (market IV, %)')
    corr = df[['sigma_20d_ann', 'vix']].corr().iloc[0, 1]
    ax.set_title(f'σ_emp vs VIX  (correlation = {corr:.3f})')
    ax.legend()
    ax.grid(alpha=0.25)
    # VIX vs RV
    ax = axes[1]
    ax.scatter(df['rv_20d'], df['vix'], s=4, alpha=0.18, color='#d62728')
    lim2 = max(df['rv_20d'].max(), df['vix'].max()) * 1.05
    ax.plot([0, lim2], [0, lim2], color='gray', ls='--', lw=0.8, label='y=x')
    ax.set_xlabel('RV (realized 20d, %)')
    ax.set_ylabel('VIX (market IV, %)')
    corr2 = df[['rv_20d', 'vix']].corr().iloc[0, 1]
    ax.set_title(f'VIX vs realized  (correlation = {corr2:.3f})')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_strategy(df: pd.DataFrame, save_path: Path):
    """Cumulative P&L of 'sell vol when VRP_implied > 0'."""
    df = df.sort_values('end_date').reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Top: cumulative P&L for several thresholds
    ax = axes[0]
    for thresh, color in [(-100, '#888'), (0, '#1f77b4'), (5, '#2ca02c'),
                          (8, '#d62728')]:
        mask = df['vrp_implied'] > thresh
        pnl = np.where(mask, df['vrp_implied'] - (df['vix'] - df['rv_20d']),
                       0)
        # Above is positioning logic; for now use simpler "VIX - RV" P&L when triggered
        pnl_simple = np.where(mask, df['vix'] - df['rv_20d'], 0.0)
        cum = pnl_simple.cumsum() / 20  # divide ~by 20 trades/year worth
        if mask.sum() > 0:
            label = f'VRP>{thresh}  ({mask.sum():,} trades, win {(pnl_simple[mask]>0).mean()*100:.0f}%)'
            ax.plot(df['end_date'], cum, color=color, lw=1.3, alpha=0.9,
                    label=label)
    ax.set_title("Cumulative 'sell vol when VRP_implied > X' P&L  "
                 "(P&L = VIX - RV per triggered day, in vol-points)",
                 fontsize=11)
    ax.set_ylabel('Cumulative VIX-RV (vol-points)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)

    # Bottom: histogram of VIX-RV (the variance risk premium PER day)
    ax = axes[1]
    ax.hist(df['vrp_realized'].dropna(), bins=80, color='#4575b4',
            edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', lw=0.8)
    mean_vrp = df['vrp_realized'].mean()
    ax.axvline(mean_vrp, color='#d62728', lw=1.5,
               label=f'mean = {mean_vrp:+.2f}')
    ax.set_xlabel('VIX - RV (annualized %)')
    ax.set_ylabel('# days')
    pos = (df['vrp_realized'].dropna() > 0).mean() * 100
    ax.set_title(f'Daily variance risk premium distribution  '
                 f'({pos:.1f}% of days have VIX>RV → "sell vol" wins)')
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import sys, io
    # Force UTF-8 stdout to avoid cp950 issues on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    run()
