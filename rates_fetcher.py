# -*- coding: utf-8 -*-
"""
Fetch interest rate data for DCF analysis:
  - Federal funds rate (effective + target range)
  - US Treasury yields (1M ~ 30Y, 11 maturities)
  - FOMC dot plot projections (median, range)
  - Market-implied future fed funds rate (from ZQ futures)
  - Automatic snapshot tracking (saves when data changes)

Setup:
  1. pip install fredapi
  2. 免費申請 FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
  3. 在專案目錄建立 .env 檔，寫入: FRED_API_KEY=your_key_here

Usage:
    from rates_fetcher import fetch_all_rates, load_rate_history
    rates = fetch_all_rates()
    rates['treasury']     # 美債殖利率
    rates['fed_funds']    # 聯邦基金利率
    rates['fomc_dots']    # FOMC 點陣圖
    rates['fed_futures']  # 市場隱含未來利率
"""

import os
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'rates_cache')
HISTORY_DIR = os.path.join(SCRIPT_DIR, 'rates_history')

# FRED series IDs
TREASURY_SERIES = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
    '1Y': 'DGS1',   '2Y': 'DGS2',   '3Y': 'DGS3',
    '5Y': 'DGS5',   '7Y': 'DGS7',   '10Y': 'DGS10',
    '20Y': 'DGS20', '30Y': 'DGS30',
}

FED_FUNDS_SERIES = {
    'effective': 'DFF',
    'target_upper': 'DFEDTARU',
    'target_lower': 'DFEDTARL',
}

FOMC_DOT_SERIES = {
    'median': 'FEDTARMD',
    'median_longrun': 'FEDTARMDLR',
    'range_high': 'FEDTARRH',
    'range_low': 'FEDTARRL',
}

# yfinance ZQ futures month codes
_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}


# ---------------------------------------------------------------------------
#  FRED client
# ---------------------------------------------------------------------------

def _get_fred():
    """Get FRED API client. Reads key from env or .env file."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("pip install fredapi")

    api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        env_path = os.path.join(SCRIPT_DIR, '.env')
        if os.path.exists(env_path):
            with open(env_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('FRED_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break

    if not api_key:
        raise ValueError(
            "需要 FRED API key（免費）\n"
            "  申請: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  設定: 在專案目錄建立 .env 檔，寫入 FRED_API_KEY=your_key"
        )

    return Fred(api_key=api_key)


def _fetch_fred_series(series_dict, start_date='2020-01-01'):
    """Fetch multiple FRED series and combine into a DataFrame."""
    fred = _get_fred()
    frames = {}
    for label, series_id in series_dict.items():
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            frames[label] = data
        except Exception as e:
            print(f"  Warning: {series_id} ({label}): {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index.name = 'Date'
    return df


# ---------------------------------------------------------------------------
#  Data fetching
# ---------------------------------------------------------------------------

def fetch_treasury_yields(start_date='2020-01-01'):
    """Fetch daily US Treasury yields (1M ~ 30Y) from FRED."""
    return _fetch_fred_series(TREASURY_SERIES, start_date)


def fetch_fed_funds_rate(start_date='2020-01-01'):
    """Fetch daily federal funds rate (effective + target range) from FRED."""
    return _fetch_fred_series(FED_FUNDS_SERIES, start_date)


def fetch_fomc_dots():
    """Fetch FOMC dot plot projections from FRED."""
    return _fetch_fred_series(FOMC_DOT_SERIES, start_date='2012-01-01')


def fetch_fed_futures(months_ahead=12):
    """Fetch market-implied future fed funds rate from yfinance ZQ futures.

    Implied rate = 100 - futures settlement price.
    No API key needed.
    """
    import yfinance as yf

    today = date.today()
    results = []

    for i in range(months_ahead):
        target = today.replace(day=1) + timedelta(days=32 * i)
        target = target.replace(day=1)  # first of month
        code = _MONTH_CODES[target.month]
        yr = target.year % 100
        ticker = f'ZQ{code}{yr:02d}.CBT'

        try:
            zq = yf.Ticker(ticker)
            hist = zq.history(period='5d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                results.append({
                    'contract': ticker,
                    'month': target.strftime('%Y-%m'),
                    'price': round(float(price), 4),
                    'implied_rate': round(100 - float(price), 4),
                })
        except Exception:
            pass
        time.sleep(0.2)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
#  Caching & loading
# ---------------------------------------------------------------------------

def fetch_all_rates(start_date='2020-01-01', cache_hours=12, force_refresh=False):
    """Fetch all rate data with caching.

    Returns dict:
      'treasury'    — DataFrame: daily yields (1M~30Y)
      'fed_funds'   — DataFrame: daily fed funds rate
      'fomc_dots'   — DataFrame: FOMC dot plot projections
      'fed_futures' — DataFrame: market-implied future rates
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    result = {}

    fetchers = {
        'treasury':    lambda: fetch_treasury_yields(start_date),
        'fed_funds':   lambda: fetch_fed_funds_rate(start_date),
        'fomc_dots':   lambda: fetch_fomc_dots(),
        'fed_futures': lambda: fetch_fed_futures(),
    }

    for name, fetcher in fetchers.items():
        cache_path = os.path.join(CACHE_DIR, f'{name}.csv')

        use_cache = False
        if not force_refresh and os.path.exists(cache_path):
            age = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age < cache_hours:
                use_cache = True

        if use_cache:
            df = pd.read_csv(cache_path, index_col=0)
            df.index = pd.to_datetime(df.index, format='mixed')
        else:
            print(f"Fetching {name}...")
            try:
                df = fetcher()
                if not df.empty:
                    df.to_csv(cache_path)
            except Exception as e:
                print(f"  Error fetching {name}: {e}")
                # Try loading stale cache
                if os.path.exists(cache_path):
                    df = pd.read_csv(cache_path, index_col=0)
                    df.index = pd.to_datetime(df.index, format='mixed')
                    print(f"  Using stale cache for {name}")
                else:
                    df = pd.DataFrame()

        result[name] = df

    # Save snapshot for revision tracking
    _save_rate_snapshot(result)

    return result


# ---------------------------------------------------------------------------
#  Snapshot tracking (records changes over time)
# ---------------------------------------------------------------------------

def _save_rate_snapshot(data):
    """Save a snapshot of current rates. Skips if unchanged from last snapshot."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(HISTORY_DIR, 'rates.csv')

    today_str = date.today().strftime('%Y-%m-%d')
    snapshot = {'fetch_date': today_str}

    # Latest treasury yields
    if 'treasury' in data and not data['treasury'].empty:
        latest = data['treasury'].dropna(how='all').iloc[-1]
        for col in latest.index:
            if pd.notna(latest[col]):
                snapshot[f'yield_{col}'] = round(float(latest[col]), 3)

    # Latest fed funds rate
    if 'fed_funds' in data and not data['fed_funds'].empty:
        ff = data['fed_funds'].dropna(how='all').iloc[-1]
        for col in ['effective', 'target_upper', 'target_lower']:
            if col in ff and pd.notna(ff[col]):
                snapshot[f'ff_{col}'] = round(float(ff[col]), 2)

    # Market-implied future rates
    if 'fed_futures' in data and not data['fed_futures'].empty:
        for _, row in data['fed_futures'].iterrows():
            snapshot[f"futures_{row['month']}"] = row['implied_rate']

    if len(snapshot) <= 1:
        return  # Nothing to save

    # Load existing history and check for changes
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)

        # Replace today's entry if exists
        hist = hist[hist['fetch_date'] != today_str]

        # Check if values changed from last entry
        if len(hist) > 0:
            last = hist.iloc[-1]
            changed = False
            for k, v in snapshot.items():
                if k == 'fetch_date':
                    continue
                old_val = last.get(k)
                if pd.isna(old_val) or abs(float(old_val) - float(v)) > 0.005:
                    changed = True
                    break
            if not changed:
                return

        hist = pd.concat([hist, pd.DataFrame([snapshot])], ignore_index=True)
    else:
        hist = pd.DataFrame([snapshot])

    hist.sort_values('fetch_date').reset_index(drop=True).to_csv(history_path, index=False)


def load_rate_history():
    """Load rate snapshot history.

    Returns DataFrame with fetch_date index.
    Columns: yield_1M..yield_30Y, ff_effective, ff_target_*,
             futures_YYYY-MM (market-implied rates)
    """
    history_path = os.path.join(HISTORY_DIR, 'rates.csv')
    if not os.path.exists(history_path):
        return pd.DataFrame()
    hist = pd.read_csv(history_path)
    hist['fetch_date'] = pd.to_datetime(hist['fetch_date'])
    return hist.set_index('fetch_date').sort_index()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  Interest Rate Data Fetcher")
    print("=" * 60)

    rates = fetch_all_rates()

    if not rates['treasury'].empty:
        print(f"\nTreasury yields: {len(rates['treasury'])} days")
        print(rates['treasury'].tail(3).to_string())

    if not rates['fed_funds'].empty:
        print(f"\nFed funds rate: {len(rates['fed_funds'])} days")
        print(rates['fed_funds'].tail(3).to_string())

    if not rates['fomc_dots'].empty:
        print(f"\nFOMC dot plot: {len(rates['fomc_dots'])} entries")
        print(rates['fomc_dots'].tail(5).to_string())

    if not rates['fed_futures'].empty:
        print(f"\nFed futures (market-implied path):")
        print(rates['fed_futures'].to_string(index=False))

    hist = load_rate_history()
    if not hist.empty:
        print(f"\nRate history: {len(hist)} snapshots")
