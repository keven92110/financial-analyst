"""Plot distribution histograms per category, per method, per time-range."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


METHOD_LABELS = {
    'cat_a': 'A · Return × Volatility',
    'cat_b': 'B · Shape Pattern',
    'cat_c': 'C · K-means (path)',
    'cat_d': 'D · RSI bucket',
    'cat_e': 'E · Distance from MA200',
}


def plot_method_distributions(
    df: pd.DataFrame, cat_col: str, fwd_col: str,
    range_label: str, baseline_mean: float = None,
    flagged: set = None, save_path: Path = None,
) -> None:
    """One figure: subplot per category showing histogram of fwd returns.

    flagged = set of category names to highlight in red.
    """
    flagged = flagged or set()
    cats = sorted(df[cat_col].dropna().unique())
    n = len(cats)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                             squeeze=False)
    fig.suptitle(
        f'{METHOD_LABELS.get(cat_col, cat_col)}  ·  {range_label}  ·  fwd={fwd_col}',
        fontsize=12, y=0.995,
    )

    # Compute global x-range across all categories for visual comparability
    all_vals = (df[fwd_col].dropna() * 100).values
    if len(all_vals) > 0:
        xmin = np.percentile(all_vals, 1)
        xmax = np.percentile(all_vals, 99)
    else:
        xmin, xmax = -5, 5

    for i, cat in enumerate(cats):
        ax = axes[i // cols, i % cols]
        vals = (df[df[cat_col] == cat][fwd_col].dropna() * 100).values
        if len(vals) == 0:
            ax.axis('off'); continue
        flagged_here = cat in flagged
        color = '#d73027' if flagged_here else '#4575b4'
        ax.hist(vals, bins=30, range=(xmin, xmax), color=color,
                edgecolor='white', linewidth=0.4, alpha=0.85)
        m = vals.mean()
        med = np.median(vals)
        hit = (vals > 0).mean() * 100
        ax.axvline(0, color='gray', lw=0.7, ls=':')
        ax.axvline(m, color='black', lw=1.2)
        if baseline_mean is not None:
            ax.axvline(baseline_mean * 100, color='#888', lw=0.8, ls='--')
        title = f'{cat}\nN={len(vals):,}  μ={m:+.2f}%  med={med:+.2f}%  hit={hit:.1f}%'
        if flagged_here:
            title = '⚑ ' + title
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('return %', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Hide unused
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_summary_grid(
    summary_df: pd.DataFrame, save_path: Path = None,
) -> None:
    """Master heatmap-style summary, split by horizon, top-K flagged per group."""
    if summary_df.empty:
        return
    flagged = summary_df[summary_df['flag']].copy()
    if flagged.empty:
        print('  (no flagged categories)')
        return

    # Show only the analysis-focus horizons (skip 1d on master plot)
    target_horizons = [h for h in ['5d', '20d'] if h in flagged['horizon'].unique()]
    if not target_horizons:
        target_horizons = sorted(flagged['horizon'].unique(),
                                 key=lambda x: int(x.replace('d', '')))
    horizons = target_horizons
    fig, axes = plt.subplots(len(horizons), 1,
                             figsize=(13, 6.5 * len(horizons)), squeeze=False)
    axes = axes[:, 0]

    for hi, h in enumerate(horizons):
        ax = axes[hi]
        sub = flagged[flagged['horizon'] == h].copy()
        # Top-K by absolute mean diff
        sub['abs_diff'] = sub['mean_diff_pct'].abs()
        sub = sub.sort_values('abs_diff', ascending=False).head(25)
        sub = sub.sort_values('mean_pct')

        y = np.arange(len(sub))
        colors = ['#d73027' if v < 0 else '#1a9850' for v in sub['mean_pct']]
        ax.barh(y, sub['mean_pct'], color=colors, edgecolor='white')
        labels = [
            f"[{r['range']}]  {r['method'].replace('cat_','').upper()} · {r['category']}  "
            f"(N={r['n']}, hit={r['hit_rate']:.0f}%, p={r['p_value']:.1e})"
            for _, r in sub.iterrows()
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(f'Mean fwd_{h} return (%)', fontsize=10)
        ax.set_title(f'Top 25 flagged · horizon = {h}', fontsize=12)
        ax.axvline(0, color='gray', lw=0.7)
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_method_ranking(summary_df: pd.DataFrame, save_path: Path = None) -> None:
    """Bar chart: avg|t| per method per horizon, big-sample ranges only."""
    if summary_df.empty:
        return
    big = ['1994-2000','2000-2009','2010-2019','2020-2024',
           'Last_3Y','Last_5Y','Last_10Y','Last_12M','All_1994-present']
    sub = summary_df[summary_df['range'].isin(big)].copy()
    sub['abs_t'] = sub['t_stat'].abs()

    pivot = (sub.groupby(['method', 'horizon'])['abs_t'].mean()
                .unstack('horizon'))
    horizons = sorted(pivot.columns, key=lambda x: int(x.replace('d', '')))
    pivot = pivot[horizons]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / len(horizons)
    palette = ['#4575b4', '#fdae61', '#d73027', '#7b3294']
    for i, h in enumerate(horizons):
        ax.bar(x + i * width - 0.4 + width / 2,
               pivot[h].values, width=width,
               label=f'{h}', color=palette[i % len(palette)],
               edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('cat_', '').upper() for m in pivot.index],
                       fontsize=10)
    ax.set_ylabel('avg |t-stat|  (higher = stronger discrimination)')
    ax.set_title('Method ranking · across big-sample ranges', fontsize=11)
    ax.legend(title='horizon', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
