# -*- coding: utf-8 -*-
"""
Fetch historical financial statements from yfinance.

3 statements × 2 frequencies = 6 datasets per ticker:
  - Income Statement (annual + quarterly)
  - Balance Sheet (annual + quarterly)
  - Cash Flow Statement (annual + quarterly)

Cached as CSV in financials_cache/{ticker}/

Usage:
    from financials_fetcher import load_all_financials
    FIN = load_all_financials(['AAPL', 'NVDA'])
    FIN['NVDA']['income_annual']       # Annual income statement
    FIN['NVDA']['balance_quarterly']   # Quarterly balance sheet
"""

import yfinance as yf
import pandas as pd
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'financials_cache')

TICKERS = ['MSFT', 'AAPL', 'NVDA', 'AMD', 'GOOG', 'META',
           'TSM', 'TSLA', 'PLTR', 'APP', 'MCD', 'COST']

# Statement types: (attribute_name, csv_name)
STATEMENTS = {
    'income_annual':      ('income_stmt',           'income_annual.csv'),
    'income_quarterly':   ('quarterly_income_stmt',  'income_quarterly.csv'),
    'balance_annual':     ('balance_sheet',          'balance_annual.csv'),
    'balance_quarterly':  ('quarterly_balance_sheet', 'balance_quarterly.csv'),
    'cashflow_annual':    ('cashflow',               'cashflow_annual.csv'),
    'cashflow_quarterly': ('quarterly_cashflow',      'cashflow_quarterly.csv'),
}


def fetch_financials(ticker_symbol):
    """Fetch all 6 financial statements for a single ticker.

    Returns dict: {statement_key: DataFrame}
    DataFrame has items as rows, dates as columns (newest first).
    """
    stock = yf.Ticker(ticker_symbol)
    result = {}

    for key, (attr_name, _) in STATEMENTS.items():
        try:
            df = getattr(stock, attr_name)
            if df is not None and not df.empty:
                # Columns are Timestamps → convert to date strings
                df.columns = [c.strftime('%Y-%m-%d') for c in df.columns]
                result[key] = df
            else:
                result[key] = pd.DataFrame()
        except Exception as e:
            print(f"  Warning: {ticker_symbol} {key}: {e}")
            result[key] = pd.DataFrame()

    return result


def load_all_financials(tickers=None, cache_hours=24, force_refresh=False):
    """Load financial statements for all tickers. Uses CSV cache if fresh.

    Args:
        tickers: list of ticker symbols (default: TICKERS)
        cache_hours: cache TTL in hours (default 24)
        force_refresh: if True, ignore cache and re-fetch

    Returns dict: {ticker: {statement_key: DataFrame}}
    """
    if tickers is None:
        tickers = TICKERS

    result = {}

    for ticker in tickers:
        ticker_dir = os.path.join(CACHE_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Check if any cache exists and is fresh
        use_cache = False
        if not force_refresh:
            sample_file = os.path.join(ticker_dir, 'income_annual.csv')
            if os.path.exists(sample_file):
                age = (time.time() - os.path.getmtime(sample_file)) / 3600
                if age < cache_hours:
                    use_cache = True

        if use_cache:
            data = {}
            for key, (_, csv_name) in STATEMENTS.items():
                path = os.path.join(ticker_dir, csv_name)
                if os.path.exists(path):
                    data[key] = pd.read_csv(path, index_col=0)
                else:
                    data[key] = pd.DataFrame()
            result[ticker] = data
        else:
            print(f"Fetching financials for {ticker}...")
            data = fetch_financials(ticker)

            # Save to cache
            for key, (_, csv_name) in STATEMENTS.items():
                path = os.path.join(ticker_dir, csv_name)
                if not data[key].empty:
                    data[key].to_csv(path)

            result[ticker] = data
            time.sleep(0.5)

    return result


# ─── Summary helpers ──────────────────────────────────────────────────

def print_summary(ticker, data):
    """Print a concise summary of key financial metrics."""
    inc = data.get('income_annual', pd.DataFrame())
    bs = data.get('balance_annual', pd.DataFrame())
    cf = data.get('cashflow_annual', pd.DataFrame())

    def fmt(val):
        if pd.isna(val):
            return '    --'
        if abs(val) >= 1e9:
            return f'{val/1e9:6.1f}B'
        if abs(val) >= 1e6:
            return f'{val/1e6:6.0f}M'
        return f'{val:6.2f}'

    print(f"\n{'='*70}")
    print(f"  {ticker} Financial Summary")
    print(f"{'='*70}")

    if not inc.empty:
        dates = inc.columns.tolist()
        print(f"\n{'':30s} {'  '.join([d[:7] for d in dates])}")
        print(f"{'-'*70}")

        # Income Statement
        print("損益表 (Income Statement)")
        for item in ['Total Revenue', 'Gross Profit', 'Operating Income',
                      'Net Income', 'Diluted EPS', 'EBITDA']:
            if item in inc.index:
                vals = '  '.join([fmt(inc.loc[item, d]) for d in dates])
                print(f"  {item:28s} {vals}")

        # Margins
        if 'Total Revenue' in inc.index and 'Gross Profit' in inc.index:
            rev = inc.loc['Total Revenue']
            gp = inc.loc['Gross Profit']
            oi = inc.loc['Operating Income'] if 'Operating Income' in inc.index else None
            ni = inc.loc['Net Income'] if 'Net Income' in inc.index else None

            print(f"  {'Gross Margin %':28s}", end='')
            for d in dates:
                r = rev[d]
                g = gp[d]
                if pd.notna(r) and r > 0 and pd.notna(g):
                    print(f' {g/r*100:5.1f}%', end='')
                else:
                    print('    --', end='')
            print()

            if oi is not None:
                print(f"  {'Operating Margin %':28s}", end='')
                for d in dates:
                    r = rev[d]
                    o = oi[d]
                    if pd.notna(r) and r > 0 and pd.notna(o):
                        print(f' {o/r*100:5.1f}%', end='')
                    else:
                        print('    --', end='')
                print()

    if not bs.empty:
        dates = bs.columns.tolist()
        print(f"\n資產負債表 (Balance Sheet)")
        for item in ['Total Assets', 'Total Liabilities Net Minority Interest',
                      'Stockholders Equity', 'Total Debt',
                      'Cash And Cash Equivalents', 'Working Capital']:
            if item in bs.index:
                vals = '  '.join([fmt(bs.loc[item, d]) for d in dates])
                print(f"  {item:28s} {vals}")

        # D/E ratio
        if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
            print(f"  {'Debt/Equity':28s}", end='')
            for d in dates:
                debt = bs.loc['Total Debt', d]
                eq = bs.loc['Stockholders Equity', d]
                if pd.notna(debt) and pd.notna(eq) and eq > 0:
                    print(f' {debt/eq:5.2f}x', end='')
                else:
                    print('    --', end='')
            print()

    if not cf.empty:
        dates = cf.columns.tolist()
        print(f"\n現金流量表 (Cash Flow)")
        for item in ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow',
                      'Repurchase Of Capital Stock']:
            if item in cf.index:
                vals = '  '.join([fmt(cf.loc[item, d]) for d in dates])
                print(f"  {item:28s} {vals}")

        # FCF margin
        if 'Free Cash Flow' in cf.index and not inc.empty and 'Total Revenue' in inc.index:
            rev = inc.loc['Total Revenue']
            fcf = cf.loc['Free Cash Flow']
            print(f"  {'FCF Margin %':28s}", end='')
            for d in cf.columns:
                f_val = fcf[d] if d in fcf.index else None
                r_val = rev[d] if d in rev.index else None
                if pd.notna(f_val) and pd.notna(r_val) and r_val > 0:
                    print(f' {f_val/r_val*100:5.1f}%', end='')
                else:
                    print('    --', end='')
            print()


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    FIN = load_all_financials()

    for ticker in TICKERS:
        print_summary(ticker, FIN[ticker])
