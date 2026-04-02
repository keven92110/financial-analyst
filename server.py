# -*- coding: utf-8 -*-
"""
Financial Analyst Web Server

FastAPI backend serving JSON APIs for the frontend.
Reuses existing data-fetching modules unchanged.

Usage:
    pip install fastapi uvicorn
    python server.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio

# Local modules
from eps_fetcher import load_all_eps
from rates_fetcher import fetch_all_rates
from financials_fetcher import load_all_financials, STATEMENTS
from dcf_valuation import (
    compute_dcf_direct, build_annual_estimates,
    fetch_breakeven_inflation, ERP_LEVELS
)
from iv_risk import compute_stock_iv

# ─── Constants ────────────────────────────────────────────────────────

TICKERS = ['MSFT', 'AAPL', 'NVDA', 'AMD', 'GOOG', 'META',
           'TSM', 'TSLA', 'PLTR', 'APP', 'MCD', 'COST']

INDEX_SYMBOLS = {
    '^GSPC': 'S&P 500',
    '^DJI':  'Dow Jones',
    '^IXIC': 'Nasdaq',
    '^RUT':  'Russell 2000',
}

PLOT_START = '2022-01-01'

# ─── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Financial Analyst")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SCRIPT_DIR, 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory cache with TTL
import time as _time

_cache = {}
_cache_ts = {}
CACHE_TTL = 600  # 10 minutes


def _get_cached(key):
    """Get cached value if not expired."""
    if key in _cache and (_time.time() - _cache_ts.get(key, 0)) < CACHE_TTL:
        return _cache[key]
    return None


def _set_cached(key, value):
    """Set cached value with timestamp."""
    _cache[key] = value
    _cache_ts[key] = _time.time()
    return value


def _ts_to_str(idx):
    """Convert pandas Timestamp index to string list."""
    return [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in idx]


def _series_to_dict(s):
    """Convert pandas Series to {dates, values}."""
    s = s.dropna()
    return {
        'dates': _ts_to_str(s.index),
        'values': [round(float(v), 4) if pd.notna(v) else None for v in s.values],
    }


def _df_to_dict(df):
    """Convert DataFrame to {index, columns, data}."""
    return {
        'index': df.index.tolist(),
        'columns': df.columns.tolist(),
        'data': {col: [round(float(v), 4) if pd.notna(v) else None
                       for v in df[col].values]
                 for col in df.columns},
    }


# ─── Frontend ─────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))


# ─── Dashboard APIs ──────────────────────────────────────────────────

@app.get("/api/dashboard/rates")
def get_dashboard_rates():
    cached = _get_cached('dashboard_rates_response')
    if cached:
        return cached

    rates = fetch_all_rates()
    _set_cached('rates', rates)

    result = {}

    # Fed funds
    ff = rates.get('fed_funds', pd.DataFrame())
    if not ff.empty:
        ff_data = {'dates': _ts_to_str(ff.index)}
        for col in ff.columns:
            ff_data[col] = [round(float(v), 4) if pd.notna(v) else None for v in ff[col].values]
        # Derive target_upper if missing
        if 'target_upper' not in ff.columns and 'target_lower' in ff.columns:
            ff_data['target_upper'] = [round(v + 0.25, 4) if v is not None else None
                                        for v in ff_data['target_lower']]
        result['fed_funds'] = ff_data

    # Treasury
    treasury = rates.get('treasury', pd.DataFrame())
    if not treasury.empty:
        latest = treasury.dropna(how='all').iloc[-1]
        result['yield_curve'] = {
            'maturities': latest.index.tolist(),
            'yields': [round(float(v), 4) if pd.notna(v) else None for v in latest.values],
        }

    # FOMC dots
    dots = rates.get('fomc_dots', pd.DataFrame())
    if not dots.empty:
        dots_future = dots[dots['median'].notna()].copy()
        today = pd.Timestamp(date.today())
        dots_future.index = pd.to_datetime(dots_future.index)
        dots_future = dots_future[dots_future.index.map(
            lambda d: pd.Timestamp(f"{d.year}-12-31")) > today]
        result['fomc_dots'] = {
            'dates': [f"{d.year}-12-31" for d in dots_future.index],
            'median': [round(float(v), 4) for v in dots_future['median'].values],
            'range_low': [round(float(v), 4) for v in dots_future['range_low'].values]
            if 'range_low' in dots_future.columns else [],
            'range_high': [round(float(v), 4) for v in dots_future['range_high'].values]
            if 'range_high' in dots_future.columns else [],
        }

    # Fed futures
    futures = rates.get('fed_futures', pd.DataFrame())
    if not futures.empty:
        result['fed_futures'] = {
            'dates': [m + '-15' for m in futures['month'].values],
            'implied_rate': [round(float(v), 4) for v in futures['implied_rate'].values],
        }

    # Breakeven inflation
    try:
        be = fetch_breakeven_inflation()
        _cache['breakeven'] = be
        if be is not None and not be.empty:
            result['breakeven_latest'] = round(float(be.dropna().iloc[-1]), 4)
    except Exception:
        pass

    return _set_cached('dashboard_rates_response', result)


@app.get("/api/dashboard/indices")
def get_indices():
    cached = _get_cached('indices_response')
    if cached:
        return cached
    result = {}
    for sym, name in INDEX_SYMBOLS.items():
        try:
            t = yf.Ticker(sym)
            h = t.history(period='6mo')
            if not h.empty:
                close = h['Close']
                result[sym] = {
                    'name': name,
                    'dates': _ts_to_str(close.index),
                    'values': [round(float(v), 2) for v in close.values],
                }
        except Exception:
            pass
    return _set_cached('indices_response', result)


@app.get("/api/dashboard/gold")
def get_gold():
    cached = _get_cached('gold_response')
    if cached:
        return cached
    try:
        gold = yf.Ticker('GC=F')
        close = gold.history(period='6mo')['Close']
        return _set_cached('gold_response', _series_to_dict(close))
    except Exception:
        return {'dates': [], 'values': []}


# ─── Stock Combined API ──────────────────────────────────────────────

@app.get("/api/stock/{ticker}/all")
def get_stock_all(ticker: str):
    """Combined endpoint: fetches EPS, price, financials, IV once and computes all views."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in list")

    cached = _get_cached(f'stock_all_{ticker}')
    if cached:
        return cached

    print(f"[Stock-All] Fetching all data for {ticker}...")
    t0 = _time.time()

    # Shared data fetches (done once)
    eps_data = load_all_eps([ticker])[ticker]
    stock = yf.Ticker(ticker)
    price_series = stock.history(period='5y')['Close']
    fin_data = load_all_financials([ticker])[ticker]

    rates = _get_cached('rates')
    if rates is None:
        rates = fetch_all_rates()
        _set_cached('rates', rates)

    breakeven_raw = _get_cached('breakeven')
    if breakeven_raw is None:
        try:
            breakeven_raw = fetch_breakeven_inflation()
            _set_cached('breakeven', breakeven_raw)
        except Exception:
            breakeven_raw = pd.Series()

    try:
        iv_data = compute_stock_iv(ticker, start_date=PLOT_START)
    except Exception:
        iv_data = None

    result = {}

    # --- EPS ---
    result['eps'] = _build_eps_response(eps_data)

    # --- Financials ---
    result['financials'] = _build_financials_response(fin_data)

    # --- PE River ---
    try:
        result['pe_river'] = _build_pe_river(eps_data, price_series, ticker)
    except Exception as e:
        print(f"[Stock-All] PE river error for {ticker}: {e}")
        result['pe_river'] = {'error': str(e)}

    # --- DCF River ---
    try:
        result['dcf_river'] = _build_dcf_river(eps_data, price_series, rates, breakeven_raw, iv_data, ticker)
    except Exception as e:
        print(f"[Stock-All] DCF error for {ticker}: {e}")
        result['dcf_river'] = {'error': str(e)}

    print(f"[Stock-All] {ticker} done in {_time.time()-t0:.1f}s")
    return _set_cached(f'stock_all_{ticker}', result)


def _build_eps_response(eps_data):
    """Build EPS response dict from eps DataFrame."""
    result = {
        'quarters': eps_data['Name'].tolist(),
        'dates': _ts_to_str(eps_data.index),
        'estimate': [round(float(v), 4) if pd.notna(v) else None for v in eps_data['Estimate EPS'].values],
        'actual': [round(float(v), 4) if pd.notna(v) else None for v in eps_data['EPS'].values],
    }
    if 'EstimateEPSnext4Q' in eps_data.columns:
        fwd = eps_data['EstimateEPSnext4Q']
        result['forward_4q'] = {'dates': _ts_to_str(fwd.dropna().index), 'values': [round(float(v), 4) for v in fwd.dropna().values]}
    if 'EPSpast4Q' in eps_data.columns:
        trailing = eps_data['EPSpast4Q']
        result['trailing_4q'] = {'dates': _ts_to_str(trailing.dropna().index), 'values': [round(float(v), 4) for v in trailing.dropna().values]}
    return result


def _build_financials_response(fin_data):
    """Build financials response."""
    def fmt(val):
        if pd.isna(val): return '--'
        val = float(val)
        if abs(val) >= 1e9: return f'{val/1e9:.1f}B'
        if abs(val) >= 1e6: return f'{val/1e6:.0f}M'
        return f'{val:.2f}'

    def pct(num, den):
        if pd.isna(num) or pd.isna(den) or den == 0: return '--'
        return f'{float(num)/float(den)*100:.1f}%'

    def ratio(num, den):
        if pd.isna(num) or pd.isna(den) or den == 0: return '--'
        return f'{float(num)/float(den):.2f}x'

    def safe_get(df, key, d):
        if d in df.columns and key in df.index: return df.loc[key, d]
        return float('nan')

    def build_table(freq):
        inc = fin_data.get(f'income_{freq}', pd.DataFrame())
        bs = fin_data.get(f'balance_{freq}', pd.DataFrame())
        cf = fin_data.get(f'cashflow_{freq}', pd.DataFrame())
        all_dates = set()
        for df in [inc, bs, cf]:
            if not df.empty: all_dates.update(df.columns.tolist())
        if not all_dates: return None
        dates = sorted(all_dates, reverse=True)
        rows = []
        if not inc.empty:
            rows.append({'label': '--- Income Statement ---', 'values': {}, 'is_header': True})
            for label, key in [('Total Revenue','Total Revenue'),('Gross Profit','Gross Profit'),
                               ('Operating Income','Operating Income'),('Net Income','Net Income'),
                               ('Diluted EPS','Diluted EPS'),('EBITDA','EBITDA')]:
                if key in inc.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(inc, key, d)) for d in dates}})
            if 'Total Revenue' in inc.index and 'Gross Profit' in inc.index:
                rows.append({'label': 'Gross Margin %', 'values': {d: pct(safe_get(inc,'Gross Profit',d), safe_get(inc,'Total Revenue',d)) for d in dates}})
            if 'Total Revenue' in inc.index and 'Operating Income' in inc.index:
                rows.append({'label': 'Operating Margin %', 'values': {d: pct(safe_get(inc,'Operating Income',d), safe_get(inc,'Total Revenue',d)) for d in dates}})
        if not bs.empty:
            rows.append({'label': '--- Balance Sheet ---', 'values': {}, 'is_header': True})
            for label, key in [('Total Assets','Total Assets'),('Total Liabilities','Total Liabilities Net Minority Interest'),
                               ('Stockholders Equity','Stockholders Equity'),('Total Debt','Total Debt'),
                               ('Cash & Equivalents','Cash And Cash Equivalents'),('Working Capital','Working Capital')]:
                if key in bs.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(bs, key, d)) for d in dates}})
            if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
                rows.append({'label': 'Debt/Equity', 'values': {d: ratio(safe_get(bs,'Total Debt',d), safe_get(bs,'Stockholders Equity',d)) for d in dates}})
        if not cf.empty:
            rows.append({'label': '--- Cash Flow ---', 'values': {}, 'is_header': True})
            for label, key in [('Operating Cash Flow','Operating Cash Flow'),('Capital Expenditure','Capital Expenditure'),
                               ('Free Cash Flow','Free Cash Flow'),('Stock Buybacks','Repurchase Of Capital Stock')]:
                if key in cf.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(cf, key, d)) for d in dates}})
            if not inc.empty and 'Free Cash Flow' in cf.index and 'Total Revenue' in inc.index:
                rows.append({'label': 'FCF Margin %', 'values': {d: pct(safe_get(cf,'Free Cash Flow',d), safe_get(inc,'Total Revenue',d)) for d in dates}})
        return {'dates': [d[:7] for d in dates], 'rows': rows}

    return {'annual': build_table('annual'), 'quarterly': build_table('quarterly')}


def _build_pe_river(eps_data, price_series, ticker):
    """Compute PE river bands."""
    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')
    pe_levels = list(range(55, 5, -5))

    last_valid = eps_data['EstimateEPSnext4Q'].dropna().index.max()
    if pd.isna(last_valid):
        return {'error': 'No forward EPS data'}

    full_range = pd.date_range(start=PLOT_START, end=last_valid.tz_localize(None)).tz_localize('America/New_York')
    eps_daily = eps_data[['EstimateEPSnext4Q', 'EPSpast4Q']].reindex(full_range)
    eps_daily = eps_daily.interpolate(method='time').bfill()

    temp_df = eps_daily.copy()
    temp_df['Close'] = price_series.reindex(full_range)
    temp_df = temp_df.dropna(subset=['EstimateEPSnext4Q'])

    dates_str = _ts_to_str(temp_df.index)
    today_idx = temp_df.index.get_indexer([today_tz], method='nearest')[0]

    bands = []
    for i in range(len(pe_levels) - 1):
        lower_pe, upper_pe = pe_levels[i], pe_levels[i + 1]
        lower_prices = (temp_df['EstimateEPSnext4Q'] * lower_pe).values
        upper_prices = (temp_df['EstimateEPSnext4Q'] * upper_pe).values
        lp = round(float(lower_prices[today_idx]), 1)
        up = round(float(upper_prices[today_idx]), 1)
        bands.append({
            'label': f'PE:{upper_pe}-{lower_pe}, ${up}-${lp}',
            'lower': [round(float(v), 2) for v in lower_prices],
            'upper': [round(float(v), 2) for v in upper_prices],
        })

    return {
        'dates': dates_str, 'today': today.strftime('%Y-%m-%d'),
        'bands': bands,
        'price': [round(float(v), 2) if pd.notna(v) else None for v in temp_df['Close'].values],
    }


def _build_dcf_river(eps_data, price_series, rates, breakeven_raw, iv_data, ticker):
    """Compute DCF river bands."""
    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')

    y1_q, y2_q, y3_q = build_annual_estimates(eps_data)
    last_valid = y1_q.dropna().index.max()
    if pd.isna(last_valid):
        return {'error': 'No forward EPS data'}

    full_range = pd.date_range(start=PLOT_START, end=last_valid.tz_localize(None)).tz_localize('America/New_York')
    annual_df = pd.DataFrame({'year1': y1_q, 'year2': y2_q, 'year3': y3_q})
    daily = annual_df.reindex(full_range).interpolate(method='time').bfill()

    naive_range = full_range.tz_localize(None)
    treasury = rates.get('treasury', pd.DataFrame())
    t10y_raw = treasury['10Y'].dropna() if '10Y' in treasury.columns else pd.Series()
    risk_free = (t10y_raw.reindex(naive_range).ffill().bfill() / 100).values

    if breakeven_raw is not None and not breakeven_raw.empty:
        inflation = (breakeven_raw.reindex(naive_range).ffill().bfill() / 100).values
    else:
        inflation = np.full(len(full_range), 0.025)

    y1, y2, y3 = daily['year1'].values, daily['year2'].values, daily['year3'].values
    erp_sorted = sorted(ERP_LEVELS)
    dcf_results = {}
    for erp in erp_sorted:
        dcf_results[erp] = compute_dcf_direct(y1, y2, y3, inflation, risk_free + erp)

    today_idx = full_range.get_indexer([today_tz], method='nearest')[0]
    dates_str = _ts_to_str(full_range)

    bands = []
    n_bands = len(erp_sorted) - 1
    for i in range(n_bands):
        upper = dcf_results[erp_sorted[i]]
        lower = dcf_results[erp_sorted[i + 1]]
        uv, lv = float(upper[today_idx]), float(lower[today_idx])
        bands.append({
            'label': f'ERP {erp_sorted[i]*100:.0f}%-{erp_sorted[i+1]*100:.0f}%, ${lv:.0f}-${uv:.0f}',
            'upper': [round(float(v), 2) if not np.isnan(v) else None for v in upper],
            'lower': [round(float(v), 2) if not np.isnan(v) else None for v in lower],
        })

    # IV line
    iv_line = None
    iv_erp_now = None
    if iv_data is not None and not iv_data.empty:
        iv_erp = iv_data['implied_erp'].reindex(full_range).ffill().bfill()
        dcf_iv = compute_dcf_direct(y1, y2, y3, inflation, risk_free + iv_erp.values)
        iv_line = [round(float(v), 2) if not np.isnan(v) else None for v in dcf_iv]
        iv_erp_now = round(float(iv_erp.iloc[today_idx]), 4)

    # Price + implied ERP
    closing_prices = price_series.reindex(full_range)
    price_vals = [round(float(v), 2) if pd.notna(v) else None for v in closing_prices.values]
    current_price = closing_prices.dropna().iloc[-1] if not closing_prices.dropna().empty else np.nan
    implied_erp = None
    if not np.isnan(current_price) and y1[today_idx] > 0:
        lo, hi = 0.001, 0.50
        for _ in range(50):
            mid = (lo + hi) / 2
            fv = compute_dcf_direct(np.array([y1[today_idx]]), np.array([y2[today_idx]]),
                np.array([y3[today_idx]]), np.array([inflation[today_idx]]),
                np.array([risk_free[today_idx] + mid]))[0]
            if np.isnan(fv) or fv < current_price: hi = mid
            else: lo = mid
        implied_erp = round((lo + hi) / 2 * 100, 1)

    # EPS for right axis
    actual_eps = eps_data['EPSpast4Q'].dropna()
    actual_eps = actual_eps[actual_eps.index >= full_range[0]]
    fwd_eps = eps_data['EstimateEPSnext4Q'].dropna()
    fwd_future = fwd_eps[fwd_eps.index > today_tz]

    return {
        'dates': dates_str, 'today': today.strftime('%Y-%m-%d'),
        'bands': bands, 'price': price_vals,
        'iv_line': iv_line, 'iv_erp_now': iv_erp_now, 'implied_erp': implied_erp,
        'actual_eps': _series_to_dict(actual_eps),
        'estimate_eps': _series_to_dict(fwd_future),
        'y_top': round(float(np.nanpercentile(dcf_results[erp_sorted[-1]], 95) * 2.5), 0),
    }


# ─── Stock Individual APIs (kept for backward compat) ────────────────

@app.get("/api/stock/{ticker}/eps")
def get_stock_eps(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in list")

    eps = load_all_eps([ticker])[ticker]

    result = {
        'quarters': eps['Name'].tolist(),
        'dates': _ts_to_str(eps.index),
        'estimate': [round(float(v), 4) if pd.notna(v) else None
                     for v in eps['Estimate EPS'].values],
        'actual': [round(float(v), 4) if pd.notna(v) else None
                   for v in eps['EPS'].values],
    }

    # Forward/trailing 4Q sums
    if 'EstimateEPSnext4Q' in eps.columns:
        fwd = eps['EstimateEPSnext4Q']
        result['forward_4q'] = {
            'dates': _ts_to_str(fwd.dropna().index),
            'values': [round(float(v), 4) for v in fwd.dropna().values],
        }
    if 'EPSpast4Q' in eps.columns:
        trailing = eps['EPSpast4Q']
        result['trailing_4q'] = {
            'dates': _ts_to_str(trailing.dropna().index),
            'values': [round(float(v), 4) for v in trailing.dropna().values],
        }

    return result


@app.get("/api/stock/{ticker}/financials")
def get_stock_financials(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in list")

    fin = load_all_financials([ticker])[ticker]

    def fmt(val):
        if pd.isna(val):
            return '--'
        val = float(val)
        if abs(val) >= 1e9:
            return f'{val/1e9:.1f}B'
        if abs(val) >= 1e6:
            return f'{val/1e6:.0f}M'
        return f'{val:.2f}'

    def pct(num, den):
        if pd.isna(num) or pd.isna(den) or den == 0:
            return '--'
        return f'{float(num)/float(den)*100:.1f}%'

    def ratio(num, den):
        if pd.isna(num) or pd.isna(den) or den == 0:
            return '--'
        return f'{float(num)/float(den):.2f}x'

    def safe_get(df, key, d):
        if d in df.columns and key in df.index:
            return df.loc[key, d]
        return float('nan')

    def build_table(freq):
        inc = fin.get(f'income_{freq}', pd.DataFrame())
        bs = fin.get(f'balance_{freq}', pd.DataFrame())
        cf = fin.get(f'cashflow_{freq}', pd.DataFrame())

        all_dates = set()
        for df in [inc, bs, cf]:
            if not df.empty:
                all_dates.update(df.columns.tolist())
        if not all_dates:
            return None
        dates = sorted(all_dates, reverse=True)

        rows = []

        if not inc.empty:
            rows.append({'label': '--- Income Statement ---', 'values': {}, 'is_header': True})
            for label, key in [('Total Revenue', 'Total Revenue'),
                               ('Gross Profit', 'Gross Profit'),
                               ('Operating Income', 'Operating Income'),
                               ('Net Income', 'Net Income'),
                               ('Diluted EPS', 'Diluted EPS'),
                               ('EBITDA', 'EBITDA')]:
                if key in inc.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(inc, key, d)) for d in dates}})
            if 'Total Revenue' in inc.index and 'Gross Profit' in inc.index:
                rows.append({'label': 'Gross Margin %', 'values': {
                    d: pct(safe_get(inc, 'Gross Profit', d), safe_get(inc, 'Total Revenue', d)) for d in dates}})
            if 'Total Revenue' in inc.index and 'Operating Income' in inc.index:
                rows.append({'label': 'Operating Margin %', 'values': {
                    d: pct(safe_get(inc, 'Operating Income', d), safe_get(inc, 'Total Revenue', d)) for d in dates}})

        if not bs.empty:
            rows.append({'label': '--- Balance Sheet ---', 'values': {}, 'is_header': True})
            for label, key in [('Total Assets', 'Total Assets'),
                               ('Total Liabilities', 'Total Liabilities Net Minority Interest'),
                               ('Stockholders Equity', 'Stockholders Equity'),
                               ('Total Debt', 'Total Debt'),
                               ('Cash & Equivalents', 'Cash And Cash Equivalents'),
                               ('Working Capital', 'Working Capital')]:
                if key in bs.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(bs, key, d)) for d in dates}})
            if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
                rows.append({'label': 'Debt/Equity', 'values': {
                    d: ratio(safe_get(bs, 'Total Debt', d), safe_get(bs, 'Stockholders Equity', d)) for d in dates}})

        if not cf.empty:
            rows.append({'label': '--- Cash Flow ---', 'values': {}, 'is_header': True})
            for label, key in [('Operating Cash Flow', 'Operating Cash Flow'),
                               ('Capital Expenditure', 'Capital Expenditure'),
                               ('Free Cash Flow', 'Free Cash Flow'),
                               ('Stock Buybacks', 'Repurchase Of Capital Stock')]:
                if key in cf.index:
                    rows.append({'label': label, 'values': {d: fmt(safe_get(cf, key, d)) for d in dates}})
            if not inc.empty and 'Free Cash Flow' in cf.index and 'Total Revenue' in inc.index:
                rows.append({'label': 'FCF Margin %', 'values': {
                    d: pct(safe_get(cf, 'Free Cash Flow', d), safe_get(inc, 'Total Revenue', d)) for d in dates}})

        return {'dates': [d[:7] for d in dates], 'rows': rows}

    return {
        'annual': build_table('annual'),
        'quarterly': build_table('quarterly'),
    }


@app.get("/api/stock/{ticker}/pe-river")
def get_pe_river(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in list")

    print(f"[PE-River] Fetching EPS for {ticker}...")
    eps_data = load_all_eps([ticker])[ticker]
    print(f"[PE-River] Fetching price for {ticker}...")
    stock = yf.Ticker(ticker)
    price_series = stock.history(period='5y')['Close']
    print(f"[PE-River] Computing for {ticker}, price points: {len(price_series)}")

    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')
    pe_levels = list(range(55, 5, -5))

    last_valid = eps_data['EstimateEPSnext4Q'].dropna().index.max()
    if pd.isna(last_valid):
        return {'error': 'No forward EPS data'}

    full_range = pd.date_range(
        start=PLOT_START, end=last_valid.tz_localize(None)
    ).tz_localize('America/New_York')

    eps_daily = eps_data[['EstimateEPSnext4Q', 'EPSpast4Q']].reindex(full_range)
    eps_daily = eps_daily.interpolate(method='time').bfill()

    temp_df = eps_daily.copy()
    temp_df['Close'] = price_series.reindex(full_range)
    temp_df = temp_df.dropna(subset=['EstimateEPSnext4Q'])

    dates_str = _ts_to_str(temp_df.index)
    today_idx = temp_df.index.get_indexer([today_tz], method='nearest')[0]

    bands = []
    for i in range(len(pe_levels) - 1):
        lower_pe, upper_pe = pe_levels[i], pe_levels[i + 1]
        lower_prices = (temp_df['EstimateEPSnext4Q'] * lower_pe).values
        upper_prices = (temp_df['EstimateEPSnext4Q'] * upper_pe).values
        lp = round(float(lower_prices[today_idx]), 1)
        up = round(float(upper_prices[today_idx]), 1)
        bands.append({
            'label': f'PE:{upper_pe}-{lower_pe}, ${up}-${lp}',
            'lower': [round(float(v), 2) for v in lower_prices],
            'upper': [round(float(v), 2) for v in upper_prices],
        })

    price_vals = temp_df['Close'].values
    return {
        'dates': dates_str,
        'today': today.strftime('%Y-%m-%d'),
        'bands': bands,
        'price': [round(float(v), 2) if pd.notna(v) else None for v in price_vals],
        'n_bands': len(bands),
    }


@app.get("/api/stock/{ticker}/dcf-river")
def get_dcf_river(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in list")

    print(f"[DCF] Fetching data for {ticker}...")
    eps_data = load_all_eps([ticker])[ticker]
    stock = yf.Ticker(ticker)
    price_series = stock.history(period='5y')['Close']
    print(f"[DCF] Data ready for {ticker}")

    # Rates
    rates = _cache.get('rates') or fetch_all_rates()
    breakeven_raw = _cache.get('breakeven')
    if breakeven_raw is None:
        try:
            breakeven_raw = fetch_breakeven_inflation()
        except Exception:
            breakeven_raw = pd.Series()

    # IV
    try:
        iv_data = compute_stock_iv(ticker, start_date=PLOT_START)
    except Exception:
        iv_data = None

    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')

    y1_q, y2_q, y3_q = build_annual_estimates(eps_data)
    last_valid = y1_q.dropna().index.max()
    if pd.isna(last_valid):
        return {'error': 'No forward EPS data'}

    full_range = pd.date_range(
        start=PLOT_START, end=last_valid.tz_localize(None)
    ).tz_localize('America/New_York')

    annual_df = pd.DataFrame({'year1': y1_q, 'year2': y2_q, 'year3': y3_q})
    daily = annual_df.reindex(full_range).interpolate(method='time').bfill()

    naive_range = full_range.tz_localize(None)
    treasury = rates.get('treasury', pd.DataFrame())
    t10y_raw = treasury['10Y'].dropna() if '10Y' in treasury.columns else pd.Series()
    risk_free = (t10y_raw.reindex(naive_range).ffill().bfill() / 100).values

    if breakeven_raw is not None and not breakeven_raw.empty:
        inflation = (breakeven_raw.reindex(naive_range).ffill().bfill() / 100).values
    else:
        inflation = np.full(len(full_range), 0.025)

    y1, y2, y3 = daily['year1'].values, daily['year2'].values, daily['year3'].values

    erp_sorted = sorted(ERP_LEVELS)
    dcf_results = {}
    for erp in erp_sorted:
        dcf_results[erp] = compute_dcf_direct(y1, y2, y3, inflation, risk_free + erp)

    today_idx = full_range.get_indexer([today_tz], method='nearest')[0]
    dates_str = _ts_to_str(full_range)

    # Bands
    bands = []
    n_bands = len(erp_sorted) - 1
    for i in range(n_bands):
        upper = dcf_results[erp_sorted[i]]
        lower = dcf_results[erp_sorted[i + 1]]
        uv, lv = float(upper[today_idx]), float(lower[today_idx])
        bands.append({
            'label': f'ERP {erp_sorted[i]*100:.0f}%-{erp_sorted[i+1]*100:.0f}%, ${lv:.0f}-${uv:.0f}',
            'upper': [round(float(v), 2) if not np.isnan(v) else None for v in upper],
            'lower': [round(float(v), 2) if not np.isnan(v) else None for v in lower],
        })

    # IV line
    iv_line = None
    iv_erp_now = None
    if iv_data is not None and not iv_data.empty:
        iv_erp = iv_data['implied_erp'].reindex(full_range).ffill().bfill()
        dcf_iv = compute_dcf_direct(y1, y2, y3, inflation, risk_free + iv_erp.values)
        iv_line = [round(float(v), 2) if not np.isnan(v) else None for v in dcf_iv]
        iv_erp_now = round(float(iv_erp.iloc[today_idx]), 4)

    # Price
    closing_prices = price_series.reindex(full_range)
    price_vals = [round(float(v), 2) if pd.notna(v) else None for v in closing_prices.values]

    # Implied ERP
    current_price = closing_prices.dropna().iloc[-1] if not closing_prices.dropna().empty else np.nan
    implied_erp = None
    if not np.isnan(current_price) and y1[today_idx] > 0:
        lo, hi = 0.001, 0.50
        for _ in range(50):
            mid = (lo + hi) / 2
            fv = compute_dcf_direct(
                np.array([y1[today_idx]]), np.array([y2[today_idx]]),
                np.array([y3[today_idx]]), np.array([inflation[today_idx]]),
                np.array([risk_free[today_idx] + mid])
            )[0]
            if np.isnan(fv) or fv < current_price:
                hi = mid
            else:
                lo = mid
        implied_erp = round((lo + hi) / 2 * 100, 1)

    # EPS for right axis
    actual_eps = eps_data['EPSpast4Q'].dropna()
    actual_eps = actual_eps[actual_eps.index >= full_range[0]]
    fwd_eps = eps_data['EstimateEPSnext4Q'].dropna()
    fwd_future = fwd_eps[fwd_eps.index > today_tz]

    return {
        'dates': dates_str,
        'today': today.strftime('%Y-%m-%d'),
        'bands': bands,
        'price': price_vals,
        'iv_line': iv_line,
        'iv_erp_now': iv_erp_now,
        'implied_erp': implied_erp,
        'actual_eps': _series_to_dict(actual_eps),
        'estimate_eps': _series_to_dict(fwd_future),
        'y_top': round(float(np.nanpercentile(dcf_results[erp_sorted[-1]], 95) * 2.5), 0),
    }


# ─── Run ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
