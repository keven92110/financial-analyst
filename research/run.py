"""Main orchestrator: load data → classify → stats → plots → summary."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all, build_returns_panel
from analysis import (
    HORIZONS, build_window_panel, apply_classifiers,
    category_stats, flag_outliers, filter_range, trailing_range,
)
from plotting import (OUTPUT_DIR, plot_method_distributions,
                      plot_summary_grid, plot_method_ranking)


# ─── Time range definitions ───────────────────────────────────────────────

# Trailing (nested) — main view, computed against latest end_date.
TRAILING_RANGES = [
    ('Last_3M',  90),     # ⚠️ small sample
    ('Last_6M',  180),    # ⚠️ small sample
    ('Last_12M', 365),
    ('Last_3Y',  365 * 3),
    ('Last_5Y',  365 * 5),
    ('Last_10Y', 365 * 10),
]

# Disjoint regimes — secondary
REGIME_RANGES = [
    ('1994-2000', '1994-01-01', '2000-12-31'),
    ('2000-2009', '2001-01-01', '2009-12-31'),
    ('2010-2019', '2010-01-01', '2019-12-31'),
    ('2020-2024', '2020-01-01', '2024-12-31'),
    ('2025-present', '2025-01-01', None),
]

METHODS = ['cat_b', 'cat_c', 'cat_d']                 # B/C/D only
PLOT_HORIZONS = [5, 20]                               # focus
SMALL_SAMPLE_RANGES = {'Last_3M', 'Last_6M', '2025-present'}
SMALL_SAMPLE_METHODS = {'cat_b', 'cat_d'}             # B & D have fewer categories


# ─── Pipeline ─────────────────────────────────────────────────────────────

def run():
    print('=' * 70)
    print('30-Day Window Classification Research')
    print('=' * 70)

    print('\n[1/5] Loading indices...')
    data = load_all()
    panel = build_returns_panel(data)

    print('\n[2/5] Building window panel (30-day windows × 5 indices)...')
    win_df = build_window_panel(panel)
    print(f'  → {len(win_df):,} windows total')
    print(f'  → date range: {win_df["end_date"].min().date()} to {win_df["end_date"].max().date()}')

    print('\n[3/5] Applying classifiers (A=ret×vol, B=shape, C=kmeans, D=rsi, E=ma200)...')
    win_df, thresholds = apply_classifiers(win_df)
    print(f"  Method A return terciles: {thresholds['ret_cuts'][0]:.4f} / {thresholds['ret_cuts'][1]:.4f}")
    print(f"  Method A vol terciles:    {thresholds['vol_cuts'][0]:.4f} / {thresholds['vol_cuts'][1]:.4f}")

    # Save full panel
    win_df.to_csv(OUTPUT_DIR / 'window_panel.csv', index=False)

    # Save classifier model bundle for predict.py / app.py
    import pickle
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / 'classifier_bundle.pkl', 'wb') as f:
        pickle.dump(thresholds, f)
    print(f"  → saved model bundle to {models_dir / 'classifier_bundle.pkl'}")

    # ── Run analysis per (range × method × horizon)
    summary_rows = []
    all_ranges = []

    # Build trailing ranges (against latest end_date)
    for label, days in TRAILING_RANGES:
        sub = trailing_range(win_df, days)
        all_ranges.append((label, sub, days))

    # Disjoint regimes
    for label, start, end in REGIME_RANGES:
        sub = filter_range(win_df, start, end)
        all_ranges.append((label, sub, None))

    # Full baseline
    all_ranges.append(('All_1994-present', win_df, None))

    print('\n[4/5] Running stats + flags + plots per range...')
    for range_label, sub_df, _ in all_ranges:
        if len(sub_df) == 0:
            print(f'  {range_label:20s}: empty, skipped')
            continue
        # For tiny ranges, re-fit kmeans / terciles is silly; reuse global labels.
        # We already applied classifiers globally, so just slice.
        small = range_label in SMALL_SAMPLE_RANGES

        for method in METHODS:
            if small and method not in SMALL_SAMPLE_METHODS:
                continue  # skip too-fine methods on tiny samples
            for h in HORIZONS:
                fwd_col = f'fwd_{h}d'
                # Baseline = full sample's distribution for this fwd horizon
                base_mean = win_df[fwd_col].dropna().mean()
                base_hit = (win_df[fwd_col].dropna() > 0).mean() * 100

                stats_df = category_stats(sub_df, method, fwd_col,
                                          baseline_mean=base_mean)
                if stats_df.empty:
                    continue
                stats_df = flag_outliers(stats_df, p_thresh=0.01,
                                         hit_diff=5.0,
                                         baseline_hit=base_hit)
                # Add metadata
                stats_df.insert(0, 'horizon', f'{h}d')
                stats_df.insert(0, 'method', method)
                stats_df.insert(0, 'range', range_label)
                summary_rows.append(stats_df)

                # Plot only horizons we care about
                if h in PLOT_HORIZONS:
                    flagged_set = set(stats_df[stats_df['flag']]['category'])
                    plot_method_distributions(
                        sub_df, method, fwd_col, range_label,
                        baseline_mean=base_mean,
                        flagged=flagged_set,
                        save_path=OUTPUT_DIR / f'{range_label}__{method}__h{h}.png',
                    )

        # Print short summary for this range
        n = len(sub_df)
        date_min = sub_df['end_date'].min().date()
        date_max = sub_df['end_date'].max().date()
        print(f'  {range_label:20s}: N={n:,}  ({date_min} → {date_max})')

    print('\n[5/5] Aggregating summary...')
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False)
    flagged = summary[summary['flag']]
    flagged.to_csv(OUTPUT_DIR / 'flagged.csv', index=False)
    print(f'  Total stat rows: {len(summary):,}')
    print(f'  Flagged rows:    {len(flagged):,}')

    # Master plots
    plot_summary_grid(summary, save_path=OUTPUT_DIR / '_flagged_summary.png')
    plot_method_ranking(summary, save_path=OUTPUT_DIR / '_method_ranking.png')

    # Print top flagged cases per horizon (B/C/D methods, p<0.01)
    for h in ['5d', '20d']:
        print('\n' + '=' * 70)
        print(f'TOP FLAGGED CASES · horizon={h} · sorted by |mean_diff_pct|')
        print('=' * 70)
        sub = flagged[(flagged['horizon'] == h) & (flagged['p_value'] < 0.01)].copy()
        sub['abs_diff'] = sub['mean_diff_pct'].abs()
        sub = sub.sort_values('abs_diff', ascending=False).head(15)
        for _, r in sub.iterrows():
            print(f"  [{r['range']:18s}] {r['method']} · {r['category']:25s} "
                  f"N={r['n']:>5,}  μ={r['mean_pct']:+.3f}%  "
                  f"hit={r['hit_rate']:5.1f}%  p={r['p_value']:.2e}")

    print('\nOutputs written to:', OUTPUT_DIR)
    return win_df, summary


if __name__ == '__main__':
    run()
