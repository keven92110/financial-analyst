# -*- coding: utf-8 -*-
"""
Auto-fetch EPS data (actual + forward estimates) from yfinance.
Replaces manual stock_report.xlsx maintenance.

Data sources (in priority order):
  1. earnings_history  — last 4Q actual EPS (always split-adjusted,
     consistent with forward estimate definitions)
  2. earnings_dates    — older history with report dates & estimates
     (split-adjusted manually, with double-adjustment detection)
  3. earnings_estimate — forward 0q/+1q consensus, 0y/+1y annual
  4. Extrapolation     — +2q~+11q via YoY annual growth on seasonal pattern

Usage:
    from eps_fetcher import load_all_eps
    FR = load_all_eps(['AAPL', 'NVDA', ...])
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import date, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'eps_cache')
HISTORY_DIR = os.path.join(SCRIPT_DIR, 'eps_history')


def _quarter_name(report_date):
    """Derive quarter name from earnings report date.

    Jan-Mar report → previous year Q4
    Apr-Jun report → current year Q1
    Jul-Sep report → current year Q2
    Oct-Dec report → current year Q3
    """
    month = report_date.month
    year = report_date.year
    if month <= 3:
        return f"{year - 1}Q4"
    elif month <= 6:
        return f"{year}Q1"
    elif month <= 9:
        return f"{year}Q2"
    else:
        return f"{year}Q3"


def _next_quarter(name):
    """'2025Q3' -> '2025Q4', '2025Q4' -> '2026Q1'."""
    year, q = int(name[:4]), int(name[-1])
    if q == 4:
        return f"{year + 1}Q1"
    return f"{year}Q{q + 1}"


def _split_factor(splits, dt):
    """Cumulative split ratio from dt to present."""
    if splits.empty:
        return 1.0
    factor = 1.0
    dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    for split_date, ratio in splits.items():
        sd = split_date.replace(tzinfo=None) if split_date.tzinfo else split_date
        if dt_naive < sd:
            factor *= ratio
    return factor


def _fix_double_adjustment(rows, splits):
    """Fix values that were already split-adjusted by Yahoo but we divided again.

    Detection: if a value is <20% of both neighbors AND near a split date,
    restore the original (pre-adjustment) Yahoo values.
    """
    if splits.empty:
        return

    for i, r in enumerate(rows):
        if r.get('_adj_eps') is None:
            continue

        dt = pd.to_datetime(r['Date'], format='%Y.%m.%d')

        # Only check near split dates
        near_split = False
        for split_date, ratio in splits.items():
            sd = split_date.replace(tzinfo=None) if split_date.tzinfo else split_date
            if abs((dt - sd).days) < 365 and ratio > 1:
                near_split = True
                break
        if not near_split:
            continue

        # Compare adjusted EPS with neighbors
        prev_eps = next((rows[j].get('_adj_eps') or rows[j]['EPS']
                         for j in range(i - 1, -1, -1)
                         if rows[j]['EPS'] is not None), None)
        next_eps = next((rows[j].get('_adj_eps') or rows[j]['EPS']
                         for j in range(i + 1, len(rows))
                         if rows[j]['EPS'] is not None), None)

        adj = r['_adj_eps']
        if prev_eps and next_eps and prev_eps > 0 and next_eps > 0:
            if adj < prev_eps * 0.2 and adj < next_eps * 0.2:
                # Was double-adjusted — use original Yahoo values (already post-split)
                r['EPS'] = round(r['_orig_eps'], 2)
                if r['_orig_est'] is not None:
                    r['Estimate EPS'] = round(r['_orig_est'], 2)
                else:
                    r['Estimate EPS'] = None


def fetch_eps_data(ticker_symbol, num_history=30):
    """
    Fetch historical and forward EPS data for a single ticker.

    Returns DataFrame with columns: Date, Name, Estimate EPS, EPS
    """
    stock = yf.Ticker(ticker_symbol)

    # --- 1. Older history from earnings_dates ---
    try:
        ed = stock.get_earnings_dates(limit=num_history)
        ed = ed[(ed['Event Type'] == 'Earnings') & ed['Reported EPS'].notna()].copy()
        ed = ed.sort_index()
        ed = ed[~ed.index.duplicated(keep='last')]
    except Exception:
        ed = pd.DataFrame()

    splits = stock.splits

    seen_names = {}
    for report_date, row in ed.iterrows():
        name = _quarter_name(report_date)
        factor = _split_factor(splits, report_date)

        est_raw = row['EPS Estimate'] if pd.notna(row['EPS Estimate']) else None
        act_raw = row['Reported EPS']

        # Store both adjusted and original values for double-adjustment detection
        entry = {
            'Date': report_date.strftime('%Y.%m.%d'),
            'Name': name,
            'Estimate EPS': round(est_raw / factor, 2) if est_raw is not None else None,
            'EPS': round(act_raw / factor, 2),
            '_orig_est': est_raw,
            '_orig_eps': act_raw,
            '_adj_eps': act_raw / factor,  # unrounded for comparison
        }
        seen_names[name] = entry

    # Sort and fix split double-adjustment
    rows = sorted(seen_names.values(), key=lambda x: x['Date'])
    _fix_double_adjustment(rows, splits)
    for r in rows:
        seen_names[r['Name']] = r

    # --- 2. Override/supplement with earnings_history (last 4Q, correct values) ---
    try:
        eh = stock.earnings_history
    except Exception:
        eh = None

    if eh is not None and not eh.empty:
        for fiscal_end, eh_row in eh.iterrows():
            fiscal_end_ts = pd.to_datetime(fiscal_end)
            est_report = fiscal_end_ts + pd.Timedelta(days=45)
            name = _quarter_name(est_report)

            actual = eh_row['epsActual']
            estimate = eh_row['epsEstimate']

            if name in seen_names:
                # Keep original report date, override EPS values
                if pd.notna(actual):
                    seen_names[name]['EPS'] = round(actual, 2)
                if pd.notna(estimate):
                    seen_names[name]['Estimate EPS'] = round(estimate, 2)
            else:
                # Recent quarter not in earnings_dates
                seen_names[name] = {
                    'Date': est_report.strftime('%Y.%m.%d'),
                    'Name': name,
                    'Estimate EPS': round(estimate, 2) if pd.notna(estimate) else None,
                    'EPS': round(actual, 2) if pd.notna(actual) else None,
                }

    # Clean up internal fields and sort
    rows = sorted(seen_names.values(), key=lambda x: x['Date'])
    for r in rows:
        r.pop('_orig_est', None)
        r.pop('_orig_eps', None)
        r.pop('_adj_eps', None)

    if not rows:
        return pd.DataFrame(columns=['Date', 'Name', 'Estimate EPS', 'EPS'])

    # --- 3. Forward estimates ---
    try:
        ee = stock.earnings_estimate
    except Exception:
        ee = None

    if ee is None or ee.empty:
        return pd.DataFrame(rows)

    q0 = ee.loc['0q', 'avg'] if '0q' in ee.index else None
    q1 = ee.loc['+1q', 'avg'] if '+1q' in ee.index else None
    y0 = ee.loc['0y', 'avg'] if '0y' in ee.index else None
    y1 = ee.loc['+1y', 'avg'] if '+1y' in ee.index else None

    # Annual growth for extrapolation (more stable than quarterly)
    annual_growth = (y1 / y0 - 1) if y0 and y1 and y0 > 0 else 0.10

    # EPS lookup (historical actual)
    hist_eps = {r['Name']: r['EPS'] for r in rows if r['EPS'] is not None}

    # Generate 12 forward quarter names (3 years)
    last_name = rows[-1]['Name']
    last_date = pd.to_datetime(rows[-1]['Date'], format='%Y.%m.%d')
    forward_names = []
    q = last_name
    for _ in range(12):
        q = _next_quarter(q)
        forward_names.append(q)

    # Compute forward estimates
    forward_est = {}

    # 0q and +1q directly from analyst consensus
    if q0 is not None:
        forward_est[forward_names[0]] = round(q0, 2)
    if q1 is not None:
        forward_est[forward_names[1]] = round(q1, 2)

    # +2q ~ +11q: apply annual YoY growth to year-ago quarter (seasonal pattern)
    for i in range(2, 12):
        qname = forward_names[i]
        year_ago = f"{int(qname[:4]) - 1}{qname[4:]}"

        yago_val = hist_eps.get(year_ago) or forward_est.get(year_ago)
        if yago_val and yago_val > 0:
            forward_est[qname] = round(yago_val * (1 + annual_growth), 2)
        else:
            # Fallback: extrapolate from last computed estimate
            prev_vals = [v for v in forward_est.values() if v is not None]
            if prev_vals:
                q_growth = (1 + annual_growth) ** 0.25 - 1
                forward_est[qname] = round(prev_vals[-1] * (1 + q_growth), 2)

    # Append forward rows
    for i, qname in enumerate(forward_names):
        if qname in forward_est:
            est_date = last_date + pd.DateOffset(months=3 * (i + 1))
            rows.append({
                'Date': est_date.strftime('%Y.%m.%d'),
                'Name': qname,
                'Estimate EPS': forward_est[qname],
                'EPS': None,
            })

    return pd.DataFrame(rows)


def fetch_annual_estimates(ticker_symbol):
    """Fetch annual EPS estimates for DCF use.

    Returns dict with keys: current_year, next_year, year3, ltg, annual_growth
    """
    stock = yf.Ticker(ticker_symbol)
    try:
        ee = stock.earnings_estimate
        ge = stock.growth_estimates
    except Exception:
        return {}

    result = {}
    if ee is not None and not ee.empty:
        y0 = ee.loc['0y', 'avg'] if '0y' in ee.index else None
        y1 = ee.loc['+1y', 'avg'] if '+1y' in ee.index else None
        result['current_year'] = y0
        result['next_year'] = y1
        if y0 and y1 and y0 > 0:
            growth = y1 / y0 - 1
            result['year3'] = round(y1 * (1 + growth), 2)
            result['annual_growth'] = round(growth, 4)

    if ge is not None and not ge.empty:
        result['ltg'] = ge.loc['LTG', 'stockTrend'] if 'LTG' in ge.index else None

    return result


# ---------------------------------------------------------------------------
#  Estimate revision tracking
# ---------------------------------------------------------------------------

def _save_snapshot(ticker, df):
    """Save today's forward estimates as a historical snapshot."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, f'{ticker}.csv')

    forward = df[df['EPS'].isna() & df['Estimate EPS'].notna()]
    if forward.empty:
        return

    today_str = date.today().strftime('%Y-%m-%d')
    snapshot = {'fetch_date': today_str}
    for _, row in forward.iterrows():
        snapshot[row['Name']] = row['Estimate EPS']

    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        hist = hist[hist['fetch_date'] != today_str]  # replace if re-fetched today
        hist = pd.concat([hist, pd.DataFrame([snapshot])], ignore_index=True)
    else:
        hist = pd.DataFrame([snapshot])

    hist = hist.sort_values('fetch_date').reset_index(drop=True)
    hist.to_csv(history_path, index=False)


def _bootstrap_eps_trend(ticker, df):
    """Use eps_trend to seed ~90 days of initial estimate history for 0q/+1q."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, f'{ticker}.csv')

    # Skip if already have multiple data points
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        if len(existing) > 1:
            return

    stock = yf.Ticker(ticker)
    try:
        trend = stock.eps_trend
    except Exception:
        return
    if trend is None or trend.empty:
        return

    # Get forward quarter names from the DataFrame
    forward = df[df['EPS'].isna() & df['Estimate EPS'].notna()]
    if len(forward) < 2:
        return
    q0_name = forward.iloc[0]['Name']
    q1_name = forward.iloc[1]['Name']

    today = date.today()
    snapshots = []
    for col, days_ago in [('90daysAgo', 90), ('60daysAgo', 60),
                          ('30daysAgo', 30), ('7daysAgo', 7)]:
        snap_date = (today - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        row = {'fetch_date': snap_date}
        if '0q' in trend.index and pd.notna(trend.loc['0q', col]):
            row[q0_name] = round(float(trend.loc['0q', col]), 2)
        if '+1q' in trend.index and pd.notna(trend.loc['+1q', col]):
            row[q1_name] = round(float(trend.loc['+1q', col]), 2)
        if len(row) > 1:
            snapshots.append(row)

    if not snapshots:
        return

    new_hist = pd.DataFrame(snapshots)
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        existing_dates = set(existing['fetch_date'].values)
        new_hist = new_hist[~new_hist['fetch_date'].isin(existing_dates)]
        if not new_hist.empty:
            combined = pd.concat([existing, new_hist], ignore_index=True)
            combined.sort_values('fetch_date').reset_index(drop=True).to_csv(history_path, index=False)
    else:
        new_hist.to_csv(history_path, index=False)


def load_estimate_history(ticker):
    """Load estimate revision history for a ticker.

    Returns DataFrame with fetch_date index, columns = quarter names.
    """
    history_path = os.path.join(HISTORY_DIR, f'{ticker}.csv')
    if not os.path.exists(history_path):
        return pd.DataFrame()
    hist = pd.read_csv(history_path)
    hist['fetch_date'] = pd.to_datetime(hist['fetch_date'])
    hist = hist.set_index('fetch_date').sort_index()
    return hist


def load_all_eps(tickers, cache_hours=12, force_refresh=False):
    """Load EPS data for all tickers. Uses CSV cache if fresh.

    Args:
        tickers: list of ticker symbols
        cache_hours: cache TTL in hours (default 12)
        force_refresh: if True, ignore cache and re-fetch all

    Returns dict: {ticker: DataFrame} with same format as stock_report.xlsx.
    DataFrames have Date index (tz-aware), plus computed columns:
      EPSpast4Q, EstimateEPSnext4Q
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    result = {}

    for ticker in tickers:
        cache_path = os.path.join(CACHE_DIR, f'{ticker}.csv')

        # Check cache freshness
        use_cache = False
        if not force_refresh and os.path.exists(cache_path):
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours < cache_hours:
                use_cache = True

            # Data-staleness check: if the most recent reported (non-estimate)
            # EPS row is > ~95 days old, a new quarterly print is probably out.
            if use_cache:
                try:
                    df_check = pd.read_csv(cache_path)
                    df_check['Date'] = pd.to_datetime(df_check['Date'],
                                                     format='%Y.%m.%d',
                                                     errors='coerce')
                    today = pd.Timestamp.today()
                    reported = df_check[(df_check['EPS'].notna()) &
                                        (df_check['Date'] <= today)]
                    if not reported.empty:
                        latest = reported['Date'].max()
                        days_since = (today - latest).days
                        if days_since > 95:
                            print(f"  {ticker}: EPS cache stale "
                                  f"(latest reported {latest.date()}, "
                                  f"{days_since}d ago) — refetching")
                            use_cache = False
                except Exception:
                    pass

        if use_cache:
            df = pd.read_csv(cache_path)
        else:
            print(f"Fetching EPS for {ticker}...")
            df = fetch_eps_data(ticker)
            df.to_csv(cache_path, index=False)
            # Record estimate snapshot for revision tracking
            _save_snapshot(ticker, df)
            _bootstrap_eps_trend(ticker, df)
            time.sleep(0.5)

        # Process into same format as original Excel loading
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d')
        df['Date'] = df['Date'].dt.tz_localize('America/New_York')
        df.set_index('Date', inplace=True, drop=True)
        df['EPSpast4Q'] = df['EPS'].shift(1).rolling(window=4).sum()
        df['EstimateEPSnext4Q'] = df['Estimate EPS'].shift(-4).rolling(window=4).sum()

        result[ticker] = df

    return result


if __name__ == '__main__':
    tickers = ['AAPL', 'NVDA', 'AMD', 'GOOG', 'META', 'TSM', 'TSLA', 'PLTR', 'APP', 'MCD', 'COST']
    for t in tickers:
        print(f"\n{'='*50}")
        print(f"  {t}")
        print(f"{'='*50}")
        df = fetch_eps_data(t)
        print(df.to_string(index=False))

        cache_path = os.path.join(CACHE_DIR, f'{t}.csv')
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)

        annual = fetch_annual_estimates(t)
        print(f"\nAnnual estimates: {annual}")
        time.sleep(0.5)
