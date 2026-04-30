"""Fetch and cache daily index data from yfinance."""
import os
import warnings
from pathlib import Path
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

CACHE_DIR = Path(__file__).parent / 'data_cache'
CACHE_DIR.mkdir(exist_ok=True)

INDICES = {
    'SPX':  '^GSPC',   # S&P 500
    'DJI':  '^DJI',    # Dow Jones
    'NDX':  '^IXIC',   # Nasdaq Composite
    'RUT':  '^RUT',    # Russell 2000
    'SOX':  '^SOX',    # PHLX Semiconductor
    'TWII': '^TWII',   # Taiwan Weighted Index
}

START_DATE = '1994-01-01'


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f'{symbol}.csv'


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
    return df


def fetch_index(name: str, ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch one index. Cache on disk; incremental update if cache exists."""
    path = _cache_path(name)

    if path.exists() and not refresh:
        cached = _read_cache(path)
        last_date = cached.index.max()
        if (pd.Timestamp.today().normalize() - last_date).days <= 1:
            return cached
        start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        new = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        if new is None or len(new) == 0:
            return cached
        new = _flatten(new)
        merged = pd.concat([cached, new])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        merged.to_csv(path)
        return merged

    df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=False)
    df = _flatten(df)
    df.to_csv(path)
    return df


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns; flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = 'Date'
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]


def load_all(refresh: bool = False) -> dict:
    """Load all indices, return dict of {name: DataFrame}."""
    result = {}
    for name, ticker in INDICES.items():
        df = fetch_index(name, ticker, refresh=refresh)
        result[name] = df
        print(f'  {name:5s} ({ticker:6s}): {len(df):,} rows  '
              f'{df.index.min().date()} → {df.index.max().date()}')
    return result


def build_returns_panel(data: dict) -> pd.DataFrame:
    """Build a panel of daily log returns (using Adj Close).

    Returns DataFrame indexed by date, columns = index names.
    """
    closes = pd.DataFrame({name: df['Adj Close'] for name, df in data.items()})
    return closes


if __name__ == '__main__':
    print('Loading indices...')
    data = load_all()
    panel = build_returns_panel(data)
    print(f'\nPanel shape: {panel.shape}')
    print(panel.tail(3))
