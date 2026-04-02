# -*- coding: utf-8 -*-
"""
Federal Funds Rate — History, Market Expectations & FOMC Projections

Plots:
  1. Historical fed funds target range (upper/lower as river band)
  2. Historical effective fed funds rate (solid line)
  3. Fed futures market-implied rate path (dashed line, future)
  4. FOMC dot plot median + range (dots with error bars, future)
  5. FOMC long-run neutral rate (horizontal dashed line)

Usage:
    python rates_plot.py
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import date, timedelta
from rates_fetcher import fetch_all_rates

# ── Fetch data ──────────────────────────────────────────────
rates = fetch_all_rates()

fed_funds = rates['fed_funds']
fomc_dots = rates['fomc_dots']
fed_futures = rates['fed_futures']

today = pd.Timestamp(date.today())

# ── Prepare historical data ─────────────────────────────────
# Target range river band
ff = fed_funds.copy()
ff.index = pd.to_datetime(ff.index)

# Forward-fill target range (announced at meetings, constant between)
ff['target_upper'] = ff['target_upper'].ffill()
ff['target_lower'] = ff['target_lower'].ffill()

# ── Prepare FOMC dot plot data ──────────────────────────────
# FRED stores year-end projections at Jan 1 of that year
# e.g. 2026-01-01 median=3.4 → projected rate at end of 2026
# Plot these at year-end dates
dots = fomc_dots.copy()
dots.index = pd.to_datetime(dots.index)

# Extract year-end projections (rows with median values)
year_dots = dots[dots['median'].notna()].copy()
# Shift to year-end for plotting
year_dots.index = year_dots.index.map(lambda d: pd.Timestamp(f"{d.year}-12-31"))
# Only keep future projections
year_dots = year_dots[year_dots.index > today]

# Extract long-run neutral rate (latest value)
longrun = dots['median_longrun'].dropna()
longrun_rate = longrun.iloc[-1] if not longrun.empty else None

# ── Prepare fed futures data ────────────────────────────────
futures = fed_futures.copy()
futures['date'] = pd.to_datetime(futures['month'] + '-15')  # mid-month

# ── Determine plot range ────────────────────────────────────
plot_start = pd.Timestamp('2020-01-01')

# End = max of futures or dots
end_candidates = [today + timedelta(days=30)]
if not futures.empty:
    end_candidates.append(futures['date'].max())
if not year_dots.empty:
    end_candidates.append(year_dots.index.max())
plot_end = max(end_candidates) + timedelta(days=60)

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# 1) Target range river band (historical)
hist_range = ff[(ff.index >= plot_start) & (ff.index <= today)]
ax.fill_between(
    hist_range.index,
    hist_range['target_lower'],
    hist_range['target_upper'],
    color='#3498db', alpha=0.25, label='Fed funds target range',
    step='post'
)

# 2) Effective fed funds rate
eff = ff['effective'].dropna()
eff = eff[(eff.index >= plot_start) & (eff.index <= today)]
ax.plot(eff.index, eff, color='#2c3e50', linewidth=1.0, label='Effective fed funds rate')

# 3) Today's divider line
ax.axvline(x=today, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.text(today, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.1, '  Today',
        fontsize=8, color='gray', va='bottom')

# 4) Fed futures implied path (market expectation)
if not futures.empty:
    # Connect from today's effective rate to futures
    current_rate = eff.iloc[-1] if not eff.empty else None
    if current_rate is not None:
        fut_dates = [today] + futures['date'].tolist()
        fut_rates = [current_rate] + futures['implied_rate'].tolist()
    else:
        fut_dates = futures['date'].tolist()
        fut_rates = futures['implied_rate'].tolist()

    ax.plot(fut_dates, fut_rates, color='#e74c3c', linewidth=2.0,
            linestyle='--', marker='o', markersize=3,
            label='Market-implied path (fed futures)', zorder=5)

# 5) FOMC dot plot — median + range
if not year_dots.empty:
    ax.errorbar(
        year_dots.index, year_dots['median'],
        yerr=[
            year_dots['median'] - year_dots['range_low'],
            year_dots['range_high'] - year_dots['median']
        ],
        fmt='D', color='#27ae60', markersize=8, capsize=6, capthick=2,
        elinewidth=2, label='FOMC dot plot (median ± range)', zorder=6
    )

# 6) Long-run neutral rate
if longrun_rate is not None:
    ax.axhline(y=longrun_rate, color='#8e44ad', linestyle=':',
               linewidth=1.5, alpha=0.8)
    ax.text(plot_end - timedelta(days=30), longrun_rate + 0.08,
            f'Long-run neutral: {longrun_rate:.1f}%',
            fontsize=9, color='#8e44ad', ha='right')

# ── Formatting ──────────────────────────────────────────────
ax.set_xlim(plot_start, plot_end)
ax.set_ylim(bottom=0)
ax.set_xlabel('Date')
ax.set_ylabel('Interest Rate (%)')
ax.set_title('Federal Funds Rate: History, Market Expectations & FOMC Projections')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()
