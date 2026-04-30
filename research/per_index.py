"""Per-index analysis: does any single index give cleaner signal than pooled?"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))
from analysis import category_stats, flag_outliers

OUTPUT_DIR = Path(__file__).parent / 'output'
PANEL = OUTPUT_DIR / 'window_panel.csv'

INDICES = ['SPX', 'DJI', 'NDX', 'RUT', 'SOX']
METHODS = ['cat_b', 'cat_c', 'cat_d']
HORIZONS = ['fwd_5d', 'fwd_20d']

BIG_RANGES = {
    '1994-2000':  ('1994-01-01', '2000-12-31'),
    '2000-2009':  ('2001-01-01', '2009-12-31'),
    '2010-2019':  ('2010-01-01', '2019-12-31'),
    '2020-2024':  ('2020-01-01', '2024-12-31'),
    'Last_5Y':    None,   # trailing
    'Last_10Y':   None,
    'Full':       None,
}


def filter_range(df, label):
    spec = BIG_RANGES[label]
    if spec is None:
        if label.startswith('Last_'):
            n = int(label.split('_')[1].replace('Y','')) * 365
            cutoff = pd.Timestamp(df['end_date'].max()) - pd.Timedelta(days=n)
            return df[df['end_date'] >= cutoff]
        return df  # Full
    a, b = spec
    return df[(df['end_date'] >= a) & (df['end_date'] <= b)]


def per_index_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (index, method, horizon, range), compute avg|t| and %sig."""
    out = []
    for idx in ['ALL_5IDX'] + INDICES:
        sub_full = panel if idx == 'ALL_5IDX' else panel[panel['index'] == idx]
        for rng_label in BIG_RANGES:
            sub = filter_range(sub_full, rng_label)
            if len(sub) == 0:
                continue
            for h_col in HORIZONS:
                base_mean = sub[h_col].dropna().mean()
                base_hit = (sub[h_col].dropna() > 0).mean() * 100
                for m in METHODS:
                    sdf = category_stats(sub, m, h_col, baseline_mean=base_mean)
                    if sdf.empty: continue
                    sdf = flag_outliers(sdf, p_thresh=0.01, hit_diff=5.0,
                                        baseline_hit=base_hit)
                    out.append({
                        'index':    idx,
                        'range':    rng_label,
                        'method':   m,
                        'horizon':  h_col,
                        'n_total':  len(sub),
                        'n_cats':   len(sdf),
                        'avg_abs_t': sdf['t_stat'].abs().mean(),
                        'max_abs_t': sdf['t_stat'].abs().max(),
                        'pct_sig':  (sdf['p_value'] < 0.01).mean() * 100,
                        'spread_mu': sdf['mean_pct'].max() - sdf['mean_pct'].min(),
                        'spread_hit': sdf['hit_rate'].max() - sdf['hit_rate'].min(),
                    })
    return pd.DataFrame(out)


def plot_per_index_ranking(summary: pd.DataFrame, save_path: Path):
    """Two-panel chart: avg|t| per (index × method) for 5d and 20d.

    Aggregated across all big ranges (mean of avg_abs_t).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    palette = {'cat_b':'#4575b4', 'cat_c':'#fdae61', 'cat_d':'#d73027'}

    for ax, h in zip(axes, HORIZONS):
        sub = summary[summary['horizon'] == h]
        # Average across ranges
        agg = sub.groupby(['index', 'method'])['avg_abs_t'].mean().unstack('method')
        order = ['ALL_5IDX'] + INDICES
        agg = agg.reindex(order)

        x = np.arange(len(agg))
        width = 0.26
        for i, m in enumerate(['cat_b', 'cat_c', 'cat_d']):
            vals = agg[m].values
            ax.bar(x + (i-1)*width, vals, width=width,
                   label=m.replace('cat_','').upper(),
                   color=palette[m], edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, fontsize=10)
        ax.set_ylabel('avg |t-stat|  (mean across ranges)')
        ax.set_title(f'horizon = {h}', fontsize=12)
        ax.legend(title='method', fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        ax.axvline(0.5, color='gray', lw=0.5, ls='--', alpha=0.5)  # ALL_5IDX divider

    fig.suptitle('Per-index discriminative power · across 7 big-sample ranges',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_pct_sig(summary: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    palette = {'cat_b':'#4575b4', 'cat_c':'#fdae61', 'cat_d':'#d73027'}
    for ax, h in zip(axes, HORIZONS):
        sub = summary[summary['horizon'] == h]
        agg = sub.groupby(['index', 'method'])['pct_sig'].mean().unstack('method')
        order = ['ALL_5IDX'] + INDICES
        agg = agg.reindex(order)
        x = np.arange(len(agg))
        width = 0.26
        for i, m in enumerate(['cat_b', 'cat_c', 'cat_d']):
            ax.bar(x + (i-1)*width, agg[m].values, width=width,
                   label=m.replace('cat_','').upper(),
                   color=palette[m], edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(agg.index, fontsize=10)
        ax.set_ylabel('% categories with p<0.01')
        ax.set_title(f'horizon = {h}', fontsize=12)
        ax.legend(title='method', fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Per-index % significant · across 7 big-sample ranges',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print('Loading panel...')
    panel = pd.read_csv(PANEL, parse_dates=['end_date'])
    print(f'  {len(panel):,} windows')
    print(f'  per-index counts: {panel["index"].value_counts().to_dict()}')

    print('\nComputing per-index summary...')
    summary = per_index_summary(panel)
    summary.to_csv(OUTPUT_DIR / 'per_index_summary.csv', index=False)

    # ── Rank table
    print('\n' + '='*88)
    print('PER-INDEX × METHOD ranking  (avg|t| averaged across 7 big-sample ranges)')
    print('='*88)
    for h in HORIZONS:
        print(f'\n--- horizon = {h} ---')
        sub = summary[summary['horizon'] == h]
        agg = (sub.groupby(['index', 'method'])
                  .agg(avg_abs_t=('avg_abs_t', 'mean'),
                       pct_sig=('pct_sig', 'mean'),
                       spread_mu=('spread_mu', 'mean'))
                  .round(3))
        print(agg.to_string())

    # ── Best per-method index
    print('\n' + '='*88)
    print('BEST INDEX PER METHOD (highest avg|t|)')
    print('='*88)
    for h in HORIZONS:
        print(f'\nhorizon = {h}')
        sub = summary[summary['horizon'] == h]
        for m in METHODS:
            mm = sub[sub['method'] == m]
            agg = mm.groupby('index')['avg_abs_t'].mean().sort_values(ascending=False)
            print(f'  {m}:')
            for idx, val in agg.head(7).items():
                marker = '★' if idx != 'ALL_5IDX' and val > agg.get('ALL_5IDX', 0) else ' '
                print(f'    {marker} {idx:7s}  avg|t| = {val:.3f}')

    plot_per_index_ranking(summary, OUTPUT_DIR / '_per_index_ranking.png')
    plot_pct_sig(summary, OUTPUT_DIR / '_per_index_pctsig.png')

    print('\nWrote:')
    print(f'  {OUTPUT_DIR / "per_index_summary.csv"}')
    print(f'  {OUTPUT_DIR / "_per_index_ranking.png"}')
    print(f'  {OUTPUT_DIR / "_per_index_pctsig.png"}')
