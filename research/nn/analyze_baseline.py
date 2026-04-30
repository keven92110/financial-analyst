"""Deeper analysis of baseline results — by year, by index, by horizon.

Reads `results/baseline_predictions.csv` (raw per-prediction rows) and
produces:
  - `baseline_by_year.csv` — per (predictor, horizon, year) metrics
  - `baseline_by_index.csv` — per (predictor, horizon, index) metrics
  - `baseline_by_year.png` — line chart: NLL/CRPS by year
  - `baseline_by_index.png` — bar chart: per-index winners
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / 'results'
PRED_PATH = RESULTS_DIR / 'baseline_predictions.csv'


def aggregate_by(df: pd.DataFrame, by: list) -> pd.DataFrame:
    sub = df[df['covered']].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    agg = (sub.groupby(by)
              .agg(n=('y_true', 'count'),
                   mse=('mse', 'mean'),
                   nll=('nll', 'mean'),
                   crps=('crps', 'mean'),
                   hit=('hit', 'mean'),
                   hit_prob=('hit_prob', 'mean'),
                   )
              .reset_index())
    agg['hit'] = agg['hit'] * 100
    agg['hit_prob'] = agg['hit_prob'] * 100
    agg['rmse'] = np.sqrt(agg['mse'])
    return agg


def plot_by_year(by_year: pd.DataFrame, save_path: Path):
    horizons = sorted(by_year['horizon'].unique())
    metrics = [('crps', 'CRPS (lower better)'),
               ('hit', 'Hit rate %  (higher better)')]
    fig, axes = plt.subplots(len(metrics), len(horizons),
                             figsize=(8 * len(horizons), 4 * len(metrics)),
                             squeeze=False)
    palette = plt.cm.tab10.colors
    for ci, h in enumerate(horizons):
        sub = by_year[by_year['horizon'] == h]
        predictors = sub['predictor'].unique()
        for ri, (col, label) in enumerate(metrics):
            ax = axes[ri, ci]
            for i, p in enumerate(predictors):
                ps = sub[sub['predictor'] == p].sort_values('year')
                ax.plot(ps['year'], ps[col], marker='o',
                        label=p, color=palette[i % len(palette)],
                        lw=1.4, markersize=4)
            ax.set_xlabel('year')
            ax.set_ylabel(label)
            ax.set_title(f'{label}  ·  horizon={h}', fontsize=10)
            ax.grid(alpha=0.25)
            if ri == 0 and ci == len(horizons) - 1:
                ax.legend(loc='upper left', fontsize=7,
                          bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_by_index(by_index: pd.DataFrame, save_path: Path):
    horizons = sorted(by_index['horizon'].unique())
    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(7 * len(horizons), 5), squeeze=False)
    for ci, h in enumerate(horizons):
        sub = by_index[by_index['horizon'] == h]
        # For each predictor, compute mean CRPS across indices
        pivot = sub.pivot_table(index='predictor', columns='index',
                                values='crps', aggfunc='mean')
        pivot = pivot.sort_values(by=pivot.columns[0])
        ax = axes[0, ci]
        x = np.arange(len(pivot))
        width = 0.8 / len(pivot.columns)
        for i, idx in enumerate(pivot.columns):
            ax.bar(x + (i - len(pivot.columns)/2 + 0.5) * width,
                   pivot[idx].values, width=width, label=idx,
                   edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('CRPS (lower better)')
        ax.set_title(f'CRPS by predictor × index  ·  horizon={h}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    print('Loading...', PRED_PATH)
    df = pd.read_csv(PRED_PATH, parse_dates=['cutoff', 'end_date'])
    df['year'] = df['end_date'].dt.year
    print(f'  {len(df):,} rows  ({df["predictor"].nunique()} predictors)')

    by_year = aggregate_by(df, ['predictor', 'horizon', 'year'])
    by_year.to_csv(RESULTS_DIR / 'baseline_by_year.csv', index=False)
    by_index = aggregate_by(df, ['predictor', 'horizon', 'index'])
    by_index.to_csv(RESULTS_DIR / 'baseline_by_index.csv', index=False)

    plot_by_year(by_year, RESULTS_DIR / 'baseline_by_year.png')
    plot_by_index(by_index, RESULTS_DIR / 'baseline_by_index.png')

    # ── Print headline numbers
    summary = aggregate_by(df, ['predictor', 'horizon'])
    print('\n══ Overall summary ══')
    print(summary.sort_values(['horizon', 'crps']).to_string(
        index=False, float_format=lambda x: f'{x:.4f}'))

    print('\n══ Best predictor per horizon (by CRPS) ══')
    for h, sub in summary.groupby('horizon'):
        best = sub.sort_values('crps').iloc[0]
        unc = sub[sub['predictor'].str.startswith('unconditional')].sort_values('crps').iloc[0]
        improvement = (unc['crps'] - best['crps']) / unc['crps'] * 100
        print(f'  {h}: {best["predictor"]:25s} CRPS={best["crps"]:.4f}  '
              f'(vs unconditional {unc["crps"]:.4f}, '
              f'improvement {improvement:+.1f}%)')


if __name__ == '__main__':
    main()
