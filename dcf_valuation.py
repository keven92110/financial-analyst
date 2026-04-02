# -*- coding: utf-8 -*-
"""
DCF Valuation River Plot — Equity Risk Premium Sweep

At each historical date, computes DCF fair value using:
  - Year 1~3: direct analyst EPS estimates (quarterly sums)
  - Year 4~10: grow from Year 3 EPS at inflation rate
  - Terminal: Gordon Growth Model (g = breakeven inflation from TIPS spread)
  - Discount rate = 10Y Treasury yield + equity risk premium (swept)

No growth-rate extrapolation — uses actual forward estimates directly.

Usage:
    python dcf_valuation.py
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

from eps_fetcher import load_all_eps
from rates_fetcher import fetch_all_rates, _get_fred
from iv_risk import compute_stock_iv

TICKERS = ['MSFT', 'AAPL', 'NVDA', 'AMD', 'GOOG', 'META',
           'TSM', 'TSLA', 'PLTR', 'APP', 'MCD', 'COST']

# Equity risk premium levels to sweep (low → high = top → bottom on chart)
ERP_LEVELS = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

PLOT_START = '2022-01-01'
DCF_TOTAL_YEARS = 10   # explicit projection years before terminal value


# ─── Breakeven inflation (terminal growth proxy) ─────────────────────

def fetch_breakeven_inflation(start_date='2020-01-01'):
    """Fetch 10Y breakeven inflation rate from FRED (T10YIE).

    Market-implied inflation = 10Y nominal yield - 10Y TIPS yield.
    Used as terminal growth rate in DCF.
    """
    try:
        fred = _get_fred()
        data = fred.get_series('T10YIE', observation_start=start_date)
        data.name = 'breakeven_10y'
        return data
    except Exception as e:
        print(f"  Warning: Could not fetch breakeven inflation: {e}")
        print(f"  Using default 2.5% terminal growth")
        return pd.Series(dtype=float, name='breakeven_10y')


# ─── Vectorized DCF computation ──────────────────────────────────────

def compute_dcf_direct(year1, year2, year3, inflation, discount_rate,
                       total_years=DCF_TOTAL_YEARS):
    """Vectorized DCF using direct annual EPS estimates.

    All inputs are numpy arrays of same length (one value per date).

    Year 1~3: direct analyst estimates (used as-is)
    Year 4~total_years: grow from Year 3 at inflation rate
    Terminal: Gordon Growth Model at inflation rate after total_years

    Returns: numpy array of fair values
    """
    r = discount_rate
    g = inflation

    # Year 1~3: direct estimates, discounted
    fv = year1 / (1 + r) + year2 / (1 + r)**2 + year3 / (1 + r)**3

    # Year 4 ~ total_years: grow from Year 3 at inflation
    eps = year3.copy().astype(np.float64)
    for y in range(4, total_years + 1):
        eps = eps * (1 + g)
        fv += eps / (1 + r) ** y

    # Terminal value (Gordon Growth Model) after total_years
    terminal_eps = eps * (1 + g)
    denom = r - g

    # Guard: need r > g for convergent perpetuity
    valid = (denom > 0.005) & (year1 > 0) & (r > 0)
    tv = np.where(valid, terminal_eps / denom, 0.0)
    fv += np.where(valid, tv / (1 + r) ** total_years, 0.0)
    fv = np.where(valid, fv, np.nan)

    return fv


# ─── Build annual EPS estimates from quarterly data ───────────────────

def build_annual_estimates(eps_data, inflation_default=0.025):
    """Extract Year 1, 2, 3 annual EPS from quarterly estimates.

    Year 1 = EstimateEPSnext4Q (quarters +1~+4)
    Year 2 = quarters +5~+8
    Year 3 = quarters +9~+12

    Where Year 2/3 unavailable, fill with (previous year) * (1 + inflation).
    """
    est = eps_data['Estimate EPS']

    # Year 1: already computed
    year1 = eps_data['EstimateEPSnext4Q']

    # Year 2: sum of quarters +5 to +8
    year2 = est.shift(-8).rolling(4).sum()

    # Year 3: sum of quarters +9 to +12
    year3 = est.shift(-12).rolling(4).sum()

    # Fill missing with inflation growth
    g = inflation_default
    year2 = year2.fillna(year1 * (1 + g))
    year3 = year3.fillna(year2 * (1 + g))

    return year1, year2, year3


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')

    # ── Load data ──
    print("Loading EPS data...")
    FR = load_all_eps(TICKERS)

    print("Loading rates data...")
    rates = fetch_all_rates()
    treasury = rates['treasury']

    print("Loading breakeven inflation...")
    breakeven_raw = fetch_breakeven_inflation()

    # 10Y Treasury yield (daily, tz-naive)
    t10y_raw = treasury['10Y'].dropna()

    for ticker in TICKERS:
        eps_data = FR[ticker]

        # ── Build Year 1, 2, 3 estimates ──
        year1_q, year2_q, year3_q = build_annual_estimates(eps_data)

        # ── Date range ──
        last_valid = year1_q.dropna().index.max()
        if pd.isna(last_valid):
            print(f"  {ticker}: no forward EPS data, skipping")
            continue

        full_range = pd.date_range(
            start=PLOT_START,
            end=last_valid.tz_localize(None),
        ).tz_localize('America/New_York')

        # ── Reindex to daily (linear interpolation for smooth bands) ──
        annual_df = pd.DataFrame({
            'year1': year1_q,
            'year2': year2_q,
            'year3': year3_q,
        }, index=eps_data.index)
        daily = annual_df.reindex(full_range).interpolate(method='time')
        daily = daily.bfill()  # fill leading NaN before first data point

        # ── Align rates ──
        naive_range = full_range.tz_localize(None)

        # 10Y Treasury → risk-free rate
        risk_free = (t10y_raw.reindex(naive_range).ffill().bfill() / 100).values

        # Breakeven inflation → terminal growth
        if not breakeven_raw.empty:
            inflation = (breakeven_raw.reindex(naive_range).ffill().bfill() / 100).values
        else:
            inflation = np.full(len(full_range), 0.025)

        # ── Extract arrays ──
        y1 = daily['year1'].values
        y2 = daily['year2'].values
        y3 = daily['year3'].values

        # ── Compute DCF for each ERP level ──
        dcf_results = {}
        for erp in ERP_LEVELS:
            r = risk_free + erp
            fv = compute_dcf_direct(y1, y2, y3, inflation, r)
            dcf_results[erp] = fv

        # ── IV-implied ERP → DCF fair value line ──
        print(f"  Computing IV for {ticker}...")
        iv_data = compute_stock_iv(ticker, start_date=PLOT_START)
        iv_erp = iv_data['implied_erp'].reindex(full_range).ffill().bfill()
        iv_erp_arr = iv_erp.values
        r_iv = risk_free + iv_erp_arr
        dcf_iv = compute_dcf_direct(y1, y2, y3, inflation, r_iv)

        # ── Stock price ──
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        closing_prices = hist['Close'].reindex(full_range)

        # ── Implied ERP from price ──
        price_plot = closing_prices.dropna()
        current_price = price_plot.iloc[-1] if not price_plot.empty else np.nan
        today_idx = full_range.get_indexer(
            [pd.Timestamp(today, tz='America/New_York')], method='nearest')[0]

        implied_erp = np.nan
        if not np.isnan(current_price) and y1[today_idx] > 0:
            lo, hi = 0.001, 0.50
            for _ in range(50):
                mid = (lo + hi) / 2
                r_test = np.array([risk_free[today_idx] + mid])
                fv_test = compute_dcf_direct(
                    np.array([y1[today_idx]]),
                    np.array([y2[today_idx]]),
                    np.array([y3[today_idx]]),
                    np.array([inflation[today_idx]]),
                    r_test
                )[0]
                if np.isnan(fv_test) or fv_test < current_price:
                    hi = mid
                else:
                    lo = mid
            implied_erp = (lo + hi) / 2

        # ── Print summary ──
        iv_erp_now = iv_erp_arr[today_idx]
        print(f"\n{ticker}: price=${current_price:.1f}  |  "
              f"Y1={y1[today_idx]:.2f}  Y2={y2[today_idx]:.2f}  "
              f"Y3={y3[today_idx]:.2f}  |  "
              f"rf={risk_free[today_idx]*100:.1f}%  |  "
              f"inflation={inflation[today_idx]*100:.1f}%")
        print(f"  Price-implied ERP: {implied_erp*100:.1f}%  |  "
              f"IV-implied ERP: {iv_erp_now*100:.1f}%  |  "
              f"IV-DCF fair value: ${dcf_iv[today_idx]:.0f}")
        erp_sorted = sorted(ERP_LEVELS)
        for erp in erp_sorted:
            print(f"  ERP={erp*100:.0f}% → ${dcf_results[erp][today_idx]:.0f}")

        # ── Plot ──
        fig, ax = plt.subplots(figsize=(14, 7))

        # DCF river bands
        n_bands = len(erp_sorted) - 1
        for i in range(n_bands):
            erp_low = erp_sorted[i]
            erp_high = erp_sorted[i + 1]
            upper = dcf_results[erp_low]
            lower = dcf_results[erp_high]

            color = plt.cm.RdYlGn_r(i / n_bands)

            upper_val = upper[today_idx]
            lower_val = lower[today_idx]
            label = (f'ERP {erp_low*100:.0f}%-{erp_high*100:.0f}%'
                     f', value: ${lower_val:.0f}-${upper_val:.0f}')

            ax.fill_between(full_range, upper, lower,
                            color=color, alpha=0.5, label=label)

        # IV-implied DCF fair value line (purple, dashed)
        dcf_iv_plot = pd.Series(dcf_iv, index=full_range)
        # Only plot where we have valid data
        dcf_iv_valid = dcf_iv_plot.dropna()
        ax.plot(dcf_iv_valid.index, dcf_iv_valid.values,
                color='purple', linewidth=2, linestyle='--',
                label=f'IV-implied fair value (ERP={iv_erp_now*100:.1f}%)')

        # Stock price (only to today)
        ax.plot(price_plot.index, price_plot.values, color='black',
                linewidth=1.5, label='Stock Price')

        # Today's divider
        ax.axvline(x=today_tz, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.7)

        # Implied ERP annotation
        if not np.isnan(implied_erp) and not np.isnan(current_price):
            ax.annotate(
                f'Price-implied ERP\n{implied_erp*100:.1f}%',
                xy=(today_tz, current_price),
                xytext=(30, 30), textcoords='offset points',
                fontsize=9, color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            )

        # ── Right Y-axis: EPS (actual + estimate) ──
        ax2 = ax.twinx()

        # Actual EPS (past, quarterly → annualized as trailing 4Q)
        actual_eps = eps_data['EPSpast4Q'].dropna()
        actual_eps = actual_eps[actual_eps.index >= full_range[0]]
        ax2.plot(actual_eps.index, actual_eps.values,
                 color='#8B4513', linewidth=2, linestyle='-',
                 marker='s', markersize=4, label='Actual EPS (trailing 4Q)')

        # Forward EPS estimates (after today)
        fwd_eps = eps_data['EstimateEPSnext4Q'].dropna()
        # Use all forward estimates for context, but split at today
        est_past = fwd_eps[fwd_eps.index <= today_tz]
        est_future = fwd_eps[fwd_eps.index > today_tz]

        # Past forward estimates (what analysts predicted at each historical date)
        if not est_past.empty:
            ax2.plot(est_past.index, est_past.values,
                     color='#4169E1', linewidth=1.5, linestyle='--', alpha=0.6,
                     label='Estimate EPS (fwd 4Q, past)')

        # Future forward estimates
        if not est_future.empty:
            ax2.plot(est_future.index, est_future.values,
                     color='#4169E1', linewidth=2, linestyle='--',
                     marker='D', markersize=4, label='Estimate EPS (fwd 4Q, future)')

        ax2.set_ylabel('EPS ($)', color='#8B4513')
        ax2.tick_params(axis='y', labelcolor='#8B4513')

        # ── Formatting ──
        ax.set_xlim(full_range[0], full_range[-1])

        # Y-axis: auto-scale to show stock price nicely
        highest_erp_vals = dcf_results[erp_sorted[-1]]
        y_top = np.nanpercentile(highest_erp_vals, 95) * 2.5
        if not np.isnan(current_price):
            y_top = max(y_top, current_price * 2.0)
        ax.set_ylim(bottom=0, top=y_top)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        erp_text = f'Price ERP: {implied_erp*100:.1f}%' if not np.isnan(implied_erp) else ''
        iv_text = f'IV ERP: {iv_erp_now*100:.1f}%'
        ax.set_title(
            f'{ticker} DCF Fair Value (Risk Premium Sweep)\n'
            f'Year 1-3: analyst estimates → Year 4+: inflation growth | {erp_text} | {iv_text}'
        )

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
