# -*- coding: utf-8 -*-
"""
Implied Volatility & Options-Implied Risk Premium

Quantifies equity risk premium from market-implied volatility:
  1. VIX × Beta: market IV scaled to individual stock
  2. Realized Vol: 30-day historical volatility
  3. Blended IV: weighted average of (1) and (2)
  4. IV → ERP conversion: ERP ≈ 0.5 × IV² (variance risk premium)

Also snapshots current options IV when market is open.

Usage:
    from iv_risk import compute_stock_iv, get_implied_erp
    iv_data = compute_stock_iv('NVDA')
    # Returns daily DataFrame with: vix, beta_iv, realized_vol, blended_iv, implied_erp
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import date, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IV_CACHE_DIR = os.path.join(SCRIPT_DIR, 'iv_cache')
IV_HISTORY_DIR = os.path.join(SCRIPT_DIR, 'iv_history')


# ─── VIX & SPY cache ─────────────────────────────────────────────────

_vix_cache = {}
_spy_cache = {}


def _get_vix(period='10y'):
    """Get VIX history (cached)."""
    if 'data' not in _vix_cache:
        vix = yf.Ticker('^VIX')
        _vix_cache['data'] = vix.history(period=period)['Close']
    return _vix_cache['data']


def _get_spy_returns(period='10y'):
    """Get SPY log returns (cached)."""
    if 'data' not in _spy_cache:
        spy = yf.Ticker('SPY')
        prices = spy.history(period=period)['Close']
        _spy_cache['data'] = np.log(prices / prices.shift(1))
    return _spy_cache['data']


# ─── Core computation ────────────────────────────────────────────────

def compute_stock_iv(ticker, start_date='2020-01-01'):
    """Compute implied volatility proxy for a stock over time.

    Returns DataFrame with columns:
        vix:          VIX level (market IV, annualized %)
        realized_vol: 30-day realized vol (annualized %)
        rolling_beta: 120-day rolling beta vs SPY
        beta_iv:      VIX × beta (stock-specific IV proxy, %)
        blended_iv:   weighted average of beta_iv and realized_vol (%)
        implied_erp:  options-implied ERP from blended IV (decimal)

    Index: DatetimeIndex (tz-aware, America/New_York)
    """
    stock = yf.Ticker(ticker)
    prices = stock.history(period='10y')['Close']
    log_returns = np.log(prices / prices.shift(1))

    # Realized vol (30-day rolling, annualized)
    realized_vol = log_returns.rolling(30, min_periods=20).std() * np.sqrt(252) * 100

    # VIX
    vix = _get_vix()

    # SPY returns for beta computation
    spy_ret = _get_spy_returns()

    # Rolling beta (120-day window)
    common_idx = log_returns.dropna().index.intersection(spy_ret.dropna().index)
    stock_ret_aligned = log_returns.reindex(common_idx)
    spy_ret_aligned = spy_ret.reindex(common_idx)

    # Use expanding then rolling covariance for beta
    cov_rolling = stock_ret_aligned.rolling(120, min_periods=60).cov(spy_ret_aligned)
    var_rolling = spy_ret_aligned.rolling(120, min_periods=60).var()
    rolling_beta = (cov_rolling / var_rolling).clip(0.5, 5.0)  # reasonable bounds

    # Build result DataFrame
    start_ts = pd.Timestamp(start_date)
    result = pd.DataFrame(index=prices.index)
    result['vix'] = vix.reindex(result.index, method='ffill')
    result['realized_vol'] = realized_vol
    result['rolling_beta'] = rolling_beta.reindex(result.index, method='ffill')

    # Beta-adjusted IV = VIX × beta
    result['beta_iv'] = result['vix'] * result['rolling_beta']

    # Blended IV: 60% beta_iv + 40% realized_vol
    # (beta_iv has forward-looking info, realized_vol is backward-looking)
    result['blended_iv'] = (
        0.6 * result['beta_iv'] + 0.4 * result['realized_vol']
    )

    # IV → ERP conversion
    # Academic: ERP ≈ 0.5 × σ² (variance risk premium)
    # Practical adjustment: scale down slightly as pure variance overestimates
    iv_decimal = result['blended_iv'] / 100
    result['implied_erp'] = 0.5 * iv_decimal ** 2

    # Filter to start_date (handle tz-aware index)
    if result.index.tz is not None:
        start_ts = start_ts.tz_localize(result.index.tz)
    result = result[result.index >= start_ts]
    result = result.dropna(subset=['blended_iv'])

    return result


def get_implied_erp(ticker, start_date='2020-01-01'):
    """Get just the implied ERP time series for a stock.

    Returns Series of ERP values (decimal, e.g. 0.05 = 5%).
    """
    iv = compute_stock_iv(ticker, start_date)
    return iv['implied_erp']


# ─── Options IV snapshot ──────────────────────────────────────────────

def snapshot_options_iv(ticker):
    """Snapshot current ATM implied volatility from options chain.

    Saves to iv_history/{ticker}.csv for building true IV history.
    Only saves if market appears open (bid/ask > 0).
    """
    os.makedirs(IV_HISTORY_DIR, exist_ok=True)
    history_path = os.path.join(IV_HISTORY_DIR, f'{ticker}.csv')

    stock = yf.Ticker(ticker)
    price = stock.history(period='1d')['Close'].iloc[-1]
    exps = stock.options

    today_str = date.today().strftime('%Y-%m-%d')
    snapshot = {'date': today_str, 'stock_price': round(float(price), 2)}

    for exp in exps:
        days = (pd.Timestamp(exp) - pd.Timestamp('today')).days
        if days < 14:
            continue

        try:
            chain = stock.option_chain(exp)
            calls = chain.calls
            puts = chain.puts

            # ATM call
            calls['dist'] = abs(calls['strike'] - price)
            atm_call = calls.sort_values('dist').iloc[0]

            # ATM put
            puts['dist'] = abs(puts['strike'] - price)
            atm_put = puts.sort_values('dist').iloc[0]

            # Only save if bid/ask exist (market open)
            if atm_call['bid'] > 0 and atm_call['ask'] > 0:
                iv_call = atm_call['impliedVolatility']
                iv_put = atm_put['impliedVolatility'] if atm_put['bid'] > 0 else iv_call
                iv_avg = (iv_call + iv_put) / 2

                snapshot[f'iv_{days}d'] = round(float(iv_avg * 100), 1)

                # Keep first 5 expirations with data
                if len([k for k in snapshot if k.startswith('iv_')]) >= 5:
                    break
        except Exception:
            continue

    # Only save if we got actual IV data
    iv_cols = [k for k in snapshot if k.startswith('iv_')]
    if not iv_cols:
        return None

    # Save/append to history
    new_row = pd.DataFrame([snapshot])
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        hist = hist[hist['date'] != today_str]  # replace today
        hist = pd.concat([hist, new_row], ignore_index=True)
    else:
        hist = new_row

    hist.sort_values('date').reset_index(drop=True).to_csv(history_path, index=False)
    return snapshot


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tickers = ['AAPL', 'NVDA', 'AMD', 'GOOG', 'META', 'TSM',
               'TSLA', 'PLTR', 'APP', 'MCD', 'COST', 'MSFT']

    print("=" * 60)
    print("  IV-based Risk Premium")
    print("=" * 60)

    for t in tickers:
        iv_data = compute_stock_iv(t)
        latest = iv_data.iloc[-1]
        print(f"\n{t}:")
        print(f"  VIX:          {latest['vix']:.1f}%")
        print(f"  Beta:         {latest['rolling_beta']:.2f}")
        print(f"  Beta IV:      {latest['beta_iv']:.1f}%")
        print(f"  Realized Vol: {latest['realized_vol']:.1f}%")
        print(f"  Blended IV:   {latest['blended_iv']:.1f}%")
        print(f"  Implied ERP:  {latest['implied_erp']*100:.1f}%")

    # Try options snapshot
    print("\n\nAttempting options IV snapshots...")
    for t in tickers[:3]:
        snap = snapshot_options_iv(t)
        if snap:
            print(f"  {t}: {snap}")
        else:
            print(f"  {t}: market closed or no data")
        time.sleep(0.5)
