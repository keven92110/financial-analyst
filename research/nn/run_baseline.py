"""Run all baseline predictors via walk-forward evaluation.

Usage:
    python run_baseline.py             # run all predictors, both horizons
    python run_baseline.py --quick     # just unconditional + D_5y for sanity check

Outputs:
    research/nn/results/baseline_predictions.csv   per-prediction metrics
    research/nn/results/baseline_summary.csv       aggregated metrics
    research/nn/results/baseline_summary.png       comparison bar chart
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))   # nn/
sys.path.insert(0, str(Path(__file__).parent.parent))   # research/

from baseline import PREDICTORS, run_predictor
from walk_forward import evaluate_predictor, aggregate

PANEL_PATH = Path(__file__).parent.parent / 'output' / 'window_panel.csv'
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def main(quick: bool = False, start_year: int = 2010):
    print('Loading panel...')
    panel = pd.read_csv(PANEL_PATH, parse_dates=['end_date'])
    print(f'  {len(panel):,} windows  ({panel["end_date"].min().date()} → '
          f'{panel["end_date"].max().date()})')

    horizons = ['fwd_5d', 'fwd_20d']
    if quick:
        predictors = ['unconditional_5y', 'D_5y', 'union_5y']
    else:
        predictors = list(PREDICTORS.keys())

    all_predictions = []
    summary_rows = []

    for horizon in horizons:
        for name in predictors:
            t0 = time.time()
            print(f'\n══ {name}  ·  horizon={horizon} ══')
            def fn(row, train, h, _name=name):
                return run_predictor(_name, row, train, h)
            df = evaluate_predictor(panel, fn, horizon=horizon,
                                    start_year=start_year, progress=True)
            df['predictor'] = name
            df['horizon'] = horizon
            all_predictions.append(df)

            agg = aggregate(df)
            agg['predictor'] = name
            agg['horizon'] = horizon
            summary_rows.append(agg)

            elapsed = time.time() - t0
            print(f'  → {agg.get("n_predictions", 0):>5,} preds   '
                  f'NLL={agg.get("nll", float("nan")):.3f}   '
                  f'CRPS={agg.get("crps", float("nan")):.3f}   '
                  f'RMSE={agg.get("rmse", float("nan")):.3f}   '
                  f'hit={agg.get("hit_rate", float("nan")):.1f}%   '
                  f'({elapsed:.1f}s)')

    # ── Save raw + summary
    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df.to_csv(RESULTS_DIR / 'baseline_predictions.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    cols = ['predictor', 'horizon', 'n_predictions', 'coverage', 'avg_n_samples',
            'nll', 'crps', 'rmse', 'mae', 'hit_rate', 'hit_prob_rate']
    summary = summary[[c for c in cols if c in summary.columns]]
    summary.to_csv(RESULTS_DIR / 'baseline_summary.csv', index=False)

    # ── Print summary tables
    for h in horizons:
        print(f'\n{"="*85}')
        print(f'BASELINE SUMMARY · horizon = {h}')
        print('='*85)
        sub = summary[summary['horizon'] == h].sort_values('crps')
        print(sub.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # ── Plot
    plot_summary(summary, RESULTS_DIR / 'baseline_summary.png')
    print(f'\n→ predictions:  {RESULTS_DIR / "baseline_predictions.csv"}')
    print(f'→ summary:      {RESULTS_DIR / "baseline_summary.csv"}')
    print(f'→ chart:        {RESULTS_DIR / "baseline_summary.png"}')
    return summary, pred_df


def plot_summary(summary: pd.DataFrame, save_path: Path):
    horizons = sorted(summary['horizon'].unique())
    metrics_to_plot = [('crps', 'CRPS  (lower better)'),
                       ('nll',  'NLL   (lower better)'),
                       ('rmse', 'RMSE  (lower better)'),
                       ('hit_rate', 'Hit rate %  (higher better)')]
    fig, axes = plt.subplots(len(metrics_to_plot), len(horizons),
                             figsize=(7 * len(horizons), 3.4 * len(metrics_to_plot)),
                             squeeze=False)
    for ci, h in enumerate(horizons):
        sub = summary[summary['horizon'] == h]
        for ri, (col, label) in enumerate(metrics_to_plot):
            ax = axes[ri, ci]
            # Sort by metric (better first)
            higher_better = (col == 'hit_rate')
            ordered = sub.sort_values(col, ascending=not higher_better)
            colors = ['#1a9850' if i == 0 else '#888' for i in range(len(ordered))]
            ax.barh(range(len(ordered)), ordered[col].values, color=colors,
                    edgecolor='white')
            ax.set_yticks(range(len(ordered)))
            ax.set_yticklabels(ordered['predictor'].values, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f'{label}  ·  {h}', fontsize=10)
            ax.grid(axis='x', alpha=0.25)
            # Annotate values
            for i, v in enumerate(ordered[col].values):
                ax.text(v, i, f' {v:.3f}', va='center', fontsize=7)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='Run only a subset of predictors for fast sanity check')
    p.add_argument('--start-year', type=int, default=2010)
    args = p.parse_args()
    main(quick=args.quick, start_year=args.start_year)
