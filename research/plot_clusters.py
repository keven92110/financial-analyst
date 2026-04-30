"""Visualize K-means cluster shapes — what does each cluster look like?

For each cluster (C1..C6), plot:
- All member paths (semi-transparent)
- Mean path (bold)
- 25/75 percentile band (shaded)
- Stats: N, mean window return, mean fwd_5d, mean fwd_20d, hit rates
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path(__file__).parent / 'output'
PANEL = OUTPUT_DIR / 'window_panel.csv'

WINDOW_DAYS = 30
PATH_COLS = [f'p{i}' for i in range(WINDOW_DAYS)]


def plot_clusters(df: pd.DataFrame, save_path: Path):
    clusters = sorted(df['cat_c'].dropna().unique(),
                      key=lambda x: int(x[1:]))  # C1..C6
    n = len(clusters)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 5.0 * rows),
                             squeeze=False)
    fig.suptitle('K-means Clusters · normalized 30-day path  (start = 1.00)',
                 fontsize=13, y=0.995)

    x_axis = np.arange(WINDOW_DAYS)

    # Compute global y-range so all panels are on same scale
    global_min = df[PATH_COLS].min().min()
    global_max = df[PATH_COLS].max().max()
    # Cap extreme outliers for visualization
    lo = max(global_min, 0.55)
    hi = min(global_max, 1.45)

    for i, cl in enumerate(clusters):
        ax = axes[i // cols, i % cols]
        sub = df[df['cat_c'] == cl]
        n_paths = len(sub)
        if n_paths == 0:
            ax.axis('off'); continue

        # Sample up to 200 paths to render (avoid black mess)
        sample = sub.sample(min(200, n_paths), random_state=0)
        for _, row in sample.iterrows():
            path = row[PATH_COLS].values.astype(float)
            ax.plot(x_axis, path, color='#888', alpha=0.07, lw=0.5)

        # Mean path
        mean_path = sub[PATH_COLS].mean().values
        ax.plot(x_axis, mean_path, color='#c0392b', lw=2.4, label='mean path')

        # Percentile band
        p25 = sub[PATH_COLS].quantile(0.25).values
        p75 = sub[PATH_COLS].quantile(0.75).values
        ax.fill_between(x_axis, p25, p75, color='#c0392b', alpha=0.18,
                        label='25–75% band')

        ax.axhline(1.0, color='gray', lw=0.6, ls=':')

        # Stats
        wret = sub['window_ret'].mean() * 100
        fwd5 = sub['fwd_5d'].mean() * 100
        fwd20 = sub['fwd_20d'].mean() * 100
        hit5 = (sub['fwd_5d'] > 0).mean() * 100
        hit20 = (sub['fwd_20d'] > 0).mean() * 100
        end_ret = (mean_path[-1] - 1) * 100

        title = (
            f'{cl}   N = {n_paths:,}   30d-path-end {end_ret:+.1f}%\n'
            f'window μ={wret:+.1f}%   '
            f'fwd_5d μ={fwd5:+.2f}% (hit {hit5:.0f}%)   '
            f'fwd_20d μ={fwd20:+.2f}% (hit {hit20:.0f}%)'
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('day in window', fontsize=10)
        ax.set_ylabel('relative price', fontsize=10)
        ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_overlay(df: pd.DataFrame, save_path: Path):
    """All 6 cluster mean paths on one chart, color-coded by mean window return."""
    clusters = sorted(df['cat_c'].dropna().unique(),
                      key=lambda x: int(x[1:]))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap('RdYlGn')
    n = len(clusters)
    for i, cl in enumerate(clusters):
        sub = df[df['cat_c'] == cl]
        mean_path = sub[PATH_COLS].mean().values
        wret = sub['window_ret'].mean() * 100
        color = cmap(i / max(1, n - 1))
        ax.plot(np.arange(WINDOW_DAYS), mean_path, color=color, lw=2.4,
                label=f'{cl}  (N={len(sub):,}, win={wret:+.1f}%)')
    ax.axhline(1.0, color='gray', lw=0.6, ls=':')
    ax.set_xlabel('day in window')
    ax.set_ylabel('relative price (start = 1.00)')
    ax.set_title('K-means · mean path per cluster (sorted C1=lowest, C6=highest)',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    df = pd.read_csv(PANEL)
    print(f'Loaded panel: {len(df):,} rows')
    print(f'Clusters: {sorted(df["cat_c"].dropna().unique())}')
    print(f'Per-cluster counts:\n{df["cat_c"].value_counts().sort_index()}')

    plot_clusters(df, OUTPUT_DIR / '_kmeans_clusters.png')
    plot_cluster_overlay(df, OUTPUT_DIR / '_kmeans_overlay.png')
    print('\nWrote:')
    print(f'  {OUTPUT_DIR / "_kmeans_clusters.png"}')
    print(f'  {OUTPUT_DIR / "_kmeans_overlay.png"}')
