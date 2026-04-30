"""Fetch CBOE volatility indices (= market-implied 30d / 9d annualized vol).

We use these as proxies for option-implied vol on each underlying:
  ^VIX     SPX 30-day IV
  ^VXN     NDX 30-day IV
  ^VXD     DJI 30-day IV
  ^VIX9D   SPX  9-day IV       (proxy for ~5-day horizon)

RUT (^RVX) and SOX have no reliable vol index in yfinance; skipped.

All values are annualized percent (e.g. 18.5 = 18.5% annualized).
"""
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

# Map our index name → yfinance vol-index symbol for each horizon
VOL_INDEX_30D = {
    'SPX': '^VIX',
    'NDX': '^VXN',
    'DJI': '^VXD',
    # 'RUT': '^RVX',   # delisted in yfinance
    # 'SOX': none
}
VOL_INDEX_9D = {
    'SPX': '^VIX9D',   # the only 9-day index available
}


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f'{symbol.lstrip("^")}.csv'


def fetch_history(symbol: str, start: str = '1994-01-01',
                  refresh: bool = False) -> pd.Series:
    """Get a vol-index Close price series. Cached on disk."""
    path = _cache_path(symbol)
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
        last = df.index.max()
        if (pd.Timestamp.today().normalize() - last).days <= 1:
            return df['Close']
        # Incremental fetch from last_date+1
        new = yf.download(symbol,
                          start=(last + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                          progress=False, auto_adjust=False)
        if new is None or len(new) == 0:
            return df['Close']
        if hasattr(new.columns, 'get_level_values'):
            new.columns = new.columns.get_level_values(0)
        merged = pd.concat([df, new[['Close']]])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        merged.to_csv(path)
        return merged['Close']

    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if hasattr(df.columns, 'get_level_values'):
        df.columns = df.columns.get_level_values(0)
    df.index.name = 'Date'
    df = df[['Close']]
    df.to_csv(path)
    return df['Close']


def fetch_current_iv(index_name: str, horizon: str = '30d') -> dict:
    """Most recent vol-index value for a given index/horizon."""
    table = VOL_INDEX_30D if horizon == '30d' else VOL_INDEX_9D
    sym = table.get(index_name)
    if sym is None:
        return {'available': False, 'reason': f'No vol-index for {index_name}/{horizon}'}
    series = fetch_history(sym)
    if series.empty:
        return {'available': False, 'reason': 'fetch failed'}
    last = series.dropna()
    if last.empty:
        return {'available': False, 'reason': 'all NaN'}
    return {
        'available': True,
        'symbol': sym,
        'date':   last.index[-1],
        'iv_pct': float(last.iloc[-1]),
        'horizon': horizon,
    }


def get_all_current_iv() -> dict:
    """Latest 30-day and 9-day IV for each index (when available)."""
    out = {}
    for idx in ['SPX', 'NDX', 'DJI', 'RUT', 'SOX']:
        out[idx] = {
            '30d': fetch_current_iv(idx, '30d'),
            '9d':  fetch_current_iv(idx, '9d'),
        }
    return out


__all__ = [
    'VOL_INDEX_30D', 'VOL_INDEX_9D',
    'fetch_history', 'fetch_current_iv', 'get_all_current_iv',
]


if __name__ == '__main__':
    print('=== Current IV per index ===')
    out = get_all_current_iv()
    for idx, h in out.items():
        for k, v in h.items():
            if v['available']:
                print(f'  {idx:5s} {k:>3s}: {v["iv_pct"]:.2f}%  '
                      f'({v["symbol"]}, {v["date"].date()})')
            else:
                print(f'  {idx:5s} {k:>3s}: --   ({v["reason"]})')
