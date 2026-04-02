# -*- coding: utf-8 -*-
"""
Financial Analyst Desktop Application

PyQt5 + Matplotlib dashboard for:
  - Fed funds rate, Treasury yield curve, rate expectations
  - US indices & Gold overview
  - Per-stock: PE river, DCF river, EPS, financial statements

Usage:
    python app.py
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import date, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QComboBox, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QScrollArea, QFrame, QStatusBar,
    QHeaderView, QSplitter, QGroupBox, QSizePolicy, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPalette

# Local modules
from eps_fetcher import load_all_eps
from rates_fetcher import fetch_all_rates, _get_fred
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

plt.style.use('seaborn-v0_8-whitegrid')


# ─── Matplotlib Canvas ────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    """Reusable matplotlib canvas widget for PyQt5."""

    def __init__(self, parent=None, width=8, height=4.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)


# ─── Data Workers (background threads) ────────────────────────────────

class DashboardWorker(QThread):
    """Fetch dashboard data in background."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)

    def run(self):
        data = {}
        try:
            self.progress.emit("Fetching interest rates...")
            data['rates'] = fetch_all_rates()

            self.progress.emit("Fetching breakeven inflation...")
            data['breakeven'] = fetch_breakeven_inflation()

            self.progress.emit("Fetching US indices...")
            indices = {}
            for sym, name in INDEX_SYMBOLS.items():
                try:
                    t = yf.Ticker(sym)
                    h = t.history(period='6mo')
                    if not h.empty:
                        indices[sym] = {
                            'name': name,
                            'history': h['Close'],
                        }
                except Exception:
                    pass
            data['indices'] = indices

            self.progress.emit("Fetching gold price...")
            try:
                gold = yf.Ticker('GC=F')
                data['gold'] = gold.history(period='6mo')['Close']
            except Exception:
                data['gold'] = pd.Series()

            self.progress.emit("Dashboard data ready.")
        except Exception as e:
            self.progress.emit(f"Error: {e}")

        self.finished.emit(data)


class StockWorker(QThread):
    """Fetch per-stock data in background."""
    finished = pyqtSignal(str, dict)
    progress = pyqtSignal(str)

    def __init__(self, ticker, rates_data=None, breakeven_data=None):
        super().__init__()
        self.ticker = ticker
        self.rates_data = rates_data
        self.breakeven_data = breakeven_data

    def run(self):
        t = self.ticker
        data = {}
        try:
            self.progress.emit(f"Fetching EPS for {t}...")
            data['eps'] = load_all_eps([t])[t]

            self.progress.emit(f"Fetching financials for {t}...")
            data['financials'] = load_all_financials([t])[t]

            self.progress.emit(f"Fetching price history for {t}...")
            stock = yf.Ticker(t)
            data['price'] = stock.history(period='5y')['Close']

            self.progress.emit(f"Computing IV for {t}...")
            data['iv'] = compute_stock_iv(t, start_date=PLOT_START)

            # Pass through shared data
            data['rates'] = self.rates_data
            data['breakeven'] = self.breakeven_data

            self.progress.emit(f"{t} data ready.")
        except Exception as e:
            self.progress.emit(f"Error loading {t}: {e}")

        self.finished.emit(t, data)


# ─── Chart Plotting Functions ─────────────────────────────────────────

def plot_yield_curve(ax, treasury_df):
    """Plot latest Treasury yield curve."""
    latest = treasury_df.dropna(how='all').iloc[-1]
    maturities = latest.index.tolist()
    yields = latest.values.astype(float)

    ax.plot(maturities, yields, 'o-', color='#2980b9', linewidth=2, markersize=6)
    ax.fill_between(range(len(maturities)), yields, alpha=0.15, color='#2980b9')
    ax.set_xticks(range(len(maturities)))
    ax.set_xticklabels(maturities, rotation=45, fontsize=8)
    ax.set_ylabel('Yield (%)')
    ax.set_title('US Treasury Yield Curve', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_rates_expectations(ax, fed_funds, fomc_dots, fed_futures):
    """Plot fed funds history + futures + FOMC dots."""
    today = pd.Timestamp(date.today())
    plot_start = pd.Timestamp('2020-01-01')

    ff = fed_funds.copy()
    ff.index = pd.to_datetime(ff.index)
    if 'target_upper' not in ff.columns:
        ff['target_upper'] = ff['target_lower'] + 0.25
    ff['target_upper'] = ff['target_upper'].ffill()
    ff['target_lower'] = ff['target_lower'].ffill()

    # Target range band
    hist = ff[(ff.index >= plot_start) & (ff.index <= today)]
    ax.fill_between(hist.index, hist['target_lower'], hist['target_upper'],
                    color='#3498db', alpha=0.25, label='Target range', step='post')

    # Effective rate
    eff = ff['effective'].dropna()
    eff = eff[(eff.index >= plot_start) & (eff.index <= today)]
    ax.plot(eff.index, eff, color='#2c3e50', linewidth=1.0, label='Effective rate')

    # Today line
    ax.axvline(x=today, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Futures
    if fed_futures is not None and not fed_futures.empty:
        futures = fed_futures.copy()
        futures['date'] = pd.to_datetime(futures['month'] + '-15')
        current_rate = eff.iloc[-1] if not eff.empty else None
        if current_rate is not None:
            fut_dates = [today] + futures['date'].tolist()
            fut_rates = [current_rate] + futures['implied_rate'].tolist()
        else:
            fut_dates = futures['date'].tolist()
            fut_rates = futures['implied_rate'].tolist()
        ax.plot(fut_dates, fut_rates, color='#e74c3c', linewidth=2, linestyle='--',
                marker='o', markersize=3, label='Market-implied path', zorder=5)

    # FOMC dots
    dots = fomc_dots.copy()
    dots.index = pd.to_datetime(dots.index)
    year_dots = dots[dots['median'].notna()].copy()
    year_dots.index = year_dots.index.map(lambda d: pd.Timestamp(f"{d.year}-12-31"))
    year_dots = year_dots[year_dots.index > today]

    if not year_dots.empty:
        if 'range_low' in year_dots.columns and 'range_high' in year_dots.columns:
            ax.errorbar(year_dots.index, year_dots['median'],
                        yerr=[year_dots['median'] - year_dots['range_low'],
                              year_dots['range_high'] - year_dots['median']],
                        fmt='D', color='#27ae60', markersize=6, capsize=4, capthick=1.5,
                        elinewidth=1.5, label='FOMC dots', zorder=6)
        elif 'range_low' in year_dots.columns:
            ax.errorbar(year_dots.index, year_dots['median'],
                        yerr=[year_dots['median'] - year_dots['range_low'],
                              year_dots['median'] * 0],
                        fmt='D', color='#27ae60', markersize=6, capsize=4, capthick=1.5,
                        elinewidth=1.5, label='FOMC dots', zorder=6)
        else:
            ax.plot(year_dots.index, year_dots['median'], 'D', color='#27ae60',
                    markersize=6, label='FOMC dots', zorder=6)

    # Long-run neutral
    if 'median_longrun' in dots.columns:
        longrun = dots['median_longrun'].dropna()
        if not longrun.empty:
            lr = longrun.iloc[-1]
            ax.axhline(y=lr, color='#8e44ad', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.text(0.98, lr, f' Neutral: {lr:.1f}%', transform=ax.get_yaxis_transform(),
                    fontsize=8, color='#8e44ad', va='bottom', ha='right')

    ax.set_ylim(bottom=0)
    ax.set_title('Fed Funds Rate & Market Expectations', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.tick_params(axis='x', rotation=30, labelsize=8)


def plot_gold(ax, gold_series):
    """Plot gold price line chart."""
    ax.plot(gold_series.index, gold_series.values, color='#DAA520', linewidth=1.5)
    ax.fill_between(gold_series.index, gold_series.values, alpha=0.1, color='#DAA520')
    ax.set_title('Gold (GC=F) 6-Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='x', rotation=30, labelsize=8)


def plot_pe_river(ax, ticker, eps_data, price_series):
    """Plot forward PE river bands."""
    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')
    pe_levels = np.arange(55, 5, -5)

    last_valid = eps_data['EstimateEPSnext4Q'].dropna().index.max()
    if pd.isna(last_valid):
        ax.text(0.5, 0.5, 'No forward EPS data', transform=ax.transAxes, ha='center')
        return

    full_range = pd.date_range(
        start=PLOT_START, end=last_valid.tz_localize(None)
    ).tz_localize('America/New_York')

    eps_daily = eps_data[['EstimateEPSnext4Q', 'EPSpast4Q']].reindex(full_range)
    eps_daily = eps_daily.interpolate(method='time').bfill()

    temp_df = eps_daily.copy()
    temp_df['Close'] = price_series.reindex(full_range)
    temp_df = temp_df.dropna(subset=['EstimateEPSnext4Q'])

    for i in range(len(pe_levels) - 1):
        lower_pe, upper_pe = pe_levels[i], pe_levels[i + 1]
        lower_prices = temp_df['EstimateEPSnext4Q'] * lower_pe
        upper_prices = temp_df['EstimateEPSnext4Q'] * upper_pe

        today_idx = temp_df.index.get_indexer([today_tz], method='nearest')[0]
        lp = round(lower_prices.iloc[today_idx], 1)
        up = round(upper_prices.iloc[today_idx], 1)
        color = plt.cm.RdYlGn_r((len(pe_levels) - i) / len(pe_levels))
        ax.fill_between(temp_df.index, lower_prices, upper_prices, color=color, alpha=0.5,
                        label=f'PE:{upper_pe}-{lower_pe}, ${up}-${lp}')

    price_df = temp_df['Close'].dropna()
    ax.plot(price_df.index, price_df, color='black', linewidth=1.5, label='Price')
    ax.axvline(x=today_tz, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_title(f'{ticker} Forward P/E River', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)


def plot_dcf_river(ax, ticker, eps_data, rates_data, breakeven_raw, iv_data, price_series):
    """Plot DCF fair value river with ERP sweep + IV line."""
    today = pd.Timestamp(date.today())
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')

    y1_q, y2_q, y3_q = build_annual_estimates(eps_data)
    last_valid = y1_q.dropna().index.max()
    if pd.isna(last_valid):
        ax.text(0.5, 0.5, 'No forward EPS data', transform=ax.transAxes, ha='center')
        return

    full_range = pd.date_range(
        start=PLOT_START, end=last_valid.tz_localize(None)
    ).tz_localize('America/New_York')

    annual_df = pd.DataFrame({'year1': y1_q, 'year2': y2_q, 'year3': y3_q})
    daily = annual_df.reindex(full_range).interpolate(method='time').bfill()

    naive_range = full_range.tz_localize(None)
    treasury = rates_data.get('treasury', pd.DataFrame())
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

    # IV-implied DCF
    dcf_iv = None
    iv_erp_now = None
    if iv_data is not None and not iv_data.empty:
        iv_erp = iv_data['implied_erp'].reindex(full_range).ffill().bfill()
        dcf_iv = compute_dcf_direct(y1, y2, y3, inflation, risk_free + iv_erp.values)
        today_idx = full_range.get_indexer([today_tz], method='nearest')[0]
        iv_erp_now = iv_erp.iloc[today_idx]

    today_idx = full_range.get_indexer([today_tz], method='nearest')[0]

    # Price-implied ERP
    closing_prices = price_series.reindex(full_range)
    price_plot = closing_prices.dropna()
    current_price = price_plot.iloc[-1] if not price_plot.empty else np.nan
    implied_erp = np.nan
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
        implied_erp = (lo + hi) / 2

    # Plot bands
    n_bands = len(erp_sorted) - 1
    for i in range(n_bands):
        upper = dcf_results[erp_sorted[i]]
        lower = dcf_results[erp_sorted[i + 1]]
        color = plt.cm.RdYlGn_r(i / n_bands)
        uv, lv = upper[today_idx], lower[today_idx]
        ax.fill_between(full_range, upper, lower, color=color, alpha=0.5,
                        label=f'ERP {erp_sorted[i]*100:.0f}%-{erp_sorted[i+1]*100:.0f}%, ${lv:.0f}-${uv:.0f}')

    # IV line
    if dcf_iv is not None:
        s = pd.Series(dcf_iv, index=full_range).dropna()
        ax.plot(s.index, s.values, color='purple', linewidth=2, linestyle='--',
                label=f'IV fair value (ERP={iv_erp_now*100:.1f}%)')

    # Stock price
    ax.plot(price_plot.index, price_plot.values, 'k-', linewidth=1.5, label='Price')
    ax.axvline(x=today_tz, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Y-axis scaling
    highest = dcf_results[erp_sorted[-1]]
    y_top = np.nanpercentile(highest, 95) * 2.5
    if not np.isnan(current_price):
        y_top = max(y_top, current_price * 2.0)
    ax.set_ylim(bottom=0, top=y_top)

    # EPS on right axis
    ax2 = ax.twinx()
    actual = eps_data['EPSpast4Q'].dropna()
    actual = actual[actual.index >= full_range[0]]
    ax2.plot(actual.index, actual.values, color='#8B4513', linewidth=2,
             marker='s', markersize=3, label='Actual EPS (trailing 4Q)')

    fwd = eps_data['EstimateEPSnext4Q'].dropna()
    est_future = fwd[fwd.index > today_tz]
    if not est_future.empty:
        ax2.plot(est_future.index, est_future.values, color='#4169E1', linewidth=2,
                 linestyle='--', marker='D', markersize=3, label='Estimate EPS (fwd 4Q)')
    ax2.set_ylabel('EPS ($)', color='#8B4513', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#8B4513')

    title_parts = [f'{ticker} DCF Fair Value']
    if not np.isnan(implied_erp):
        title_parts.append(f'Price ERP: {implied_erp*100:.1f}%')
    if iv_erp_now is not None:
        title_parts.append(f'IV ERP: {iv_erp_now*100:.1f}%')
    ax.set_title(' | '.join(title_parts), fontsize=11, fontweight='bold')
    ax.set_ylabel('Price ($)')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
              loc='center left', bbox_to_anchor=(1.08, 0.5))
    ax.grid(True, alpha=0.3)


def plot_eps_bars(ax, eps_data):
    """Plot quarterly EPS: actual vs estimate bar chart."""
    today_tz = pd.Timestamp(date.today(), tz='America/New_York')

    df = eps_data[['Name', 'Estimate EPS', 'EPS']].copy()
    # Past: has actual EPS
    past = df[df['EPS'].notna()].tail(12)
    # Future: estimate only
    future = df[(df['EPS'].isna()) & (df['Estimate EPS'].notna())].head(8)
    show = pd.concat([past, future])

    if show.empty:
        ax.text(0.5, 0.5, 'No EPS data', transform=ax.transAxes, ha='center')
        return

    names = show['Name'].values
    x = np.arange(len(names))
    width = 0.35

    est = show['Estimate EPS'].values.astype(float)
    act = show['EPS'].values.astype(float)

    # Estimate bars
    ax.bar(x - width / 2, est, width, color='#95a5a6', alpha=0.7)

    # Actual bars (color by beat/miss)
    for i, (a, e) in enumerate(zip(act, est)):
        if np.isnan(a):
            continue
        color = '#2ecc71' if a >= e else '#e74c3c'
        ax.bar(x[i] + width / 2, a, width, color=color, alpha=0.9)

    # Legend with colored patches
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#95a5a6', alpha=0.7, label='Estimate'),
        Patch(facecolor='#2ecc71', alpha=0.9, label='Actual (beat)'),
        Patch(facecolor='#e74c3c', alpha=0.9, label='Actual (miss)'),
    ]
    ax.legend(handles=handles, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, fontsize=8, ha='right')
    ax.set_ylabel('EPS ($)')
    ax.set_title('Quarterly EPS: Actual vs Estimate', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Divider line between past and future
    if len(past) > 0 and len(future) > 0:
        div_x = len(past) - 0.5
        ax.axvline(x=div_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(div_x, ax.get_ylim()[1] * 0.95, '  Future →',
                fontsize=8, color='gray', va='top')


# ─── Financial Statements Table ───────────────────────────────────────

def build_financials_table(fin_data):
    """Build a QTableWidget from financial statement data."""
    inc = fin_data.get('income_annual', pd.DataFrame())
    bs = fin_data.get('balance_annual', pd.DataFrame())
    cf = fin_data.get('cashflow_annual', pd.DataFrame())

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

    # Build unified date list (union of all statements, sorted newest first)
    all_dates = set()
    for df in [inc, bs, cf]:
        if not df.empty:
            all_dates.update(df.columns.tolist())
    if not all_dates:
        return None
    dates = sorted(all_dates, reverse=True)

    def safe_get(df, key, d):
        """Safely get value from DataFrame, return NaN if missing."""
        if d in df.columns and key in df.index:
            return df.loc[key, d]
        return float('nan')

    # Define rows
    sections = []

    if not inc.empty:
        items = [
            ('--- Income Statement ---', None),
            ('Total Revenue', 'Total Revenue'),
            ('Gross Profit', 'Gross Profit'),
            ('Operating Income', 'Operating Income'),
            ('Net Income', 'Net Income'),
            ('Diluted EPS', 'Diluted EPS'),
            ('EBITDA', 'EBITDA'),
        ]
        for label, key in items:
            if key is None:
                sections.append((label, [''] * len(dates)))
            elif key in inc.index:
                sections.append((label, [fmt(safe_get(inc, key, d)) for d in dates]))

        # Margins
        if 'Total Revenue' in inc.index and 'Gross Profit' in inc.index:
            sections.append(('Gross Margin %',
                             [pct(safe_get(inc, 'Gross Profit', d), safe_get(inc, 'Total Revenue', d)) for d in dates]))
        if 'Total Revenue' in inc.index and 'Operating Income' in inc.index:
            sections.append(('Operating Margin %',
                             [pct(safe_get(inc, 'Operating Income', d), safe_get(inc, 'Total Revenue', d)) for d in dates]))

    if not bs.empty:
        items = [
            ('--- Balance Sheet ---', None),
            ('Total Assets', 'Total Assets'),
            ('Total Liabilities', 'Total Liabilities Net Minority Interest'),
            ('Stockholders Equity', 'Stockholders Equity'),
            ('Total Debt', 'Total Debt'),
            ('Cash & Equivalents', 'Cash And Cash Equivalents'),
            ('Working Capital', 'Working Capital'),
        ]
        for label, key in items:
            if key is None:
                sections.append((label, [''] * len(dates)))
            elif key in bs.index:
                sections.append((label, [fmt(safe_get(bs, key, d)) for d in dates]))

        if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
            sections.append(('Debt/Equity',
                             [ratio(safe_get(bs, 'Total Debt', d), safe_get(bs, 'Stockholders Equity', d)) for d in dates]))

    if not cf.empty:
        items = [
            ('--- Cash Flow ---', None),
            ('Operating Cash Flow', 'Operating Cash Flow'),
            ('Capital Expenditure', 'Capital Expenditure'),
            ('Free Cash Flow', 'Free Cash Flow'),
            ('Stock Buybacks', 'Repurchase Of Capital Stock'),
        ]
        for label, key in items:
            if key is None:
                sections.append((label, [''] * len(dates)))
            elif key in cf.index:
                sections.append((label, [fmt(safe_get(cf, key, d)) for d in dates]))

        if not inc.empty and 'Free Cash Flow' in cf.index and 'Total Revenue' in inc.index:
            sections.append(('FCF Margin %',
                             [pct(safe_get(cf, 'Free Cash Flow', d), safe_get(inc, 'Total Revenue', d)) for d in dates]))

    if not sections:
        return None

    table = QTableWidget()
    table.setRowCount(len(sections))
    table.setColumnCount(len(dates) + 1)
    table.setHorizontalHeaderLabels(['Item'] + [d[:7] for d in dates])

    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    for i in range(1, len(dates) + 1):
        header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

    for row, (label, vals) in enumerate(sections):
        # Label
        item = QTableWidgetItem(label)
        if label.startswith('---'):
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setBackground(QColor(230, 230, 240))
        table.setItem(row, 0, item)

        # Values
        if vals:
            for col, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, col + 1, cell)

    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.verticalHeader().setVisible(False)

    return table


# ─── Dashboard Page ───────────────────────────────────────────────────

class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)

        # Row 0: Fed status bar
        self.fed_frame = QGroupBox("Federal Funds Rate")
        fed_layout = QHBoxLayout(self.fed_frame)
        self.fed_effective = QLabel("--")
        self.fed_effective.setFont(QFont('Arial', 24, QFont.Bold))
        self.fed_target = QLabel("Target: -- ~ --")
        self.fed_target.setFont(QFont('Arial', 12))
        fed_layout.addWidget(self.fed_effective)
        fed_layout.addWidget(self.fed_target)
        fed_layout.addStretch()
        layout.addWidget(self.fed_frame, 0, 0, 1, 2)

        # Row 1: Yield curve + Rate expectations
        self.yield_canvas = MplCanvas(self, width=6, height=3.5)
        self.rates_canvas = MplCanvas(self, width=8, height=3.5)
        layout.addWidget(self.yield_canvas, 1, 0)
        layout.addWidget(self.rates_canvas, 1, 1)

        # Row 2: Indices chart + Gold
        self.indices_canvas = MplCanvas(self, width=6, height=3)
        self.gold_canvas = MplCanvas(self, width=6, height=3)
        layout.addWidget(self.indices_canvas, 2, 0)
        layout.addWidget(self.gold_canvas, 2, 1)

        layout.setRowStretch(1, 3)
        layout.setRowStretch(2, 2)

    def update_data(self, data):
        """Populate dashboard with fetched data."""
        rates = data.get('rates', {})

        # Fed status
        ff = rates.get('fed_funds', pd.DataFrame())
        if not ff.empty:
            # Find most recent row with some data
            valid = ff.dropna(how='all')
            if not valid.empty:
                lower = valid['target_lower'].dropna().iloc[-1] if 'target_lower' in valid.columns else None
                upper = valid['target_upper'].dropna().iloc[-1] if 'target_upper' in valid.columns else None
                eff_s = valid['effective'].dropna() if 'effective' in valid.columns else pd.Series()
                eff = eff_s.iloc[-1] if not eff_s.empty else None
                # Derive missing upper from lower
                if upper is None and lower is not None:
                    upper = lower + 0.25
                if eff is not None:
                    self.fed_effective.setText(f"{eff:.2f}%")
                elif upper is not None and lower is not None:
                    self.fed_effective.setText(f"{(upper+lower)/2:.2f}%")
                if upper is not None and lower is not None:
                    self.fed_target.setText(f"Target: {lower:.2f}% ~ {upper:.2f}%")

        # Yield curve
        try:
            treasury = rates.get('treasury', pd.DataFrame())
            if not treasury.empty:
                self.yield_canvas.clear()
                plot_yield_curve(self.yield_canvas.ax, treasury)
                self.yield_canvas.fig.tight_layout()
                self.yield_canvas.draw()
        except Exception as e:
            print(f"Dashboard: yield curve error: {e}")

        # Rate expectations
        try:
            if not ff.empty:
                self.rates_canvas.clear()
                plot_rates_expectations(
                    self.rates_canvas.ax, ff,
                    rates.get('fomc_dots', pd.DataFrame()),
                    rates.get('fed_futures', pd.DataFrame())
                )
                self.rates_canvas.fig.tight_layout()
                self.rates_canvas.draw()
        except Exception as e:
            print(f"Dashboard: rate expectations error: {e}")

        # Indices — normalized line chart
        try:
            indices = data.get('indices', {})
            if indices:
                self.indices_canvas.clear()
                ax_idx = self.indices_canvas.ax
                colors = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']
                for i, (sym, name) in enumerate(INDEX_SYMBOLS.items()):
                    info = indices.get(sym)
                    if info and not info['history'].empty:
                        series = info['history']
                        # Normalize to percentage change from first value
                        normalized = (series / series.iloc[0] - 1) * 100
                        label = f"{info['name']} ({normalized.iloc[-1]:+.1f}%)"
                        ax_idx.plot(normalized.index, normalized.values,
                                    color=colors[i % len(colors)], linewidth=1.5,
                                    label=label)
                ax_idx.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                ax_idx.set_title('US Indices (6M % Change)', fontsize=11, fontweight='bold')
                ax_idx.set_ylabel('Change (%)')
                ax_idx.legend(fontsize=7, loc='upper left')
                ax_idx.grid(True, alpha=0.3)
                ax_idx.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax_idx.tick_params(axis='x', rotation=30, labelsize=8)
                self.indices_canvas.fig.tight_layout()
                self.indices_canvas.draw()
        except Exception as e:
            print(f"Dashboard: indices error: {e}")

        # Gold
        try:
            gold = data.get('gold', pd.Series())
            if not gold.empty:
                self.gold_canvas.clear()
                plot_gold(self.gold_canvas.ax, gold)
                self.gold_canvas.fig.tight_layout()
                self.gold_canvas.draw()
        except Exception as e:
            print(f"Dashboard: gold error: {e}")


# ─── Stock Detail Page ────────────────────────────────────────────────

class StockDetailPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_ticker = None

        # Main layout with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.addWidget(scroll)

        # Loading label (shown while data loads)
        self.loading_label = QLabel("Select a stock to view details...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont('Arial', 14))
        self.content_layout.addWidget(self.loading_label)

        # Placeholder widgets (created on first load)
        self.fin_table = None
        self.pe_canvas = None
        self.dcf_canvas = None
        self.eps_canvas = None

    def _init_widgets(self):
        """Create chart widgets on first use."""
        # Clear layout
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Title
        self.title_label = QLabel()
        self.title_label.setFont(QFont('Arial', 16, QFont.Bold))
        self.content_layout.addWidget(self.title_label)

        # Tabs for organization
        tabs = QTabWidget()

        # Tab 1: Financial Statements (annual + quarterly sub-tabs)
        self.fin_tabs = QTabWidget()
        self.fin_annual_container = QWidget()
        self.fin_annual_layout = QVBoxLayout(self.fin_annual_container)
        self.fin_quarterly_container = QWidget()
        self.fin_quarterly_layout = QVBoxLayout(self.fin_quarterly_container)
        self.fin_tabs.addTab(self.fin_annual_container, "年報 (Annual)")
        self.fin_tabs.addTab(self.fin_quarterly_container, "季報 (Quarterly)")
        tabs.addTab(self.fin_tabs, "Financial Statements")

        # Tab 2: PE River
        self.pe_canvas = MplCanvas(self, width=12, height=6)
        tabs.addTab(self.pe_canvas, "Forward P/E River")

        # Tab 3: DCF River
        self.dcf_canvas = MplCanvas(self, width=12, height=6)
        tabs.addTab(self.dcf_canvas, "DCF Valuation")

        # Tab 4: EPS
        self.eps_canvas = MplCanvas(self, width=12, height=5)
        tabs.addTab(self.eps_canvas, "EPS Analysis")

        self.content_layout.addWidget(tabs)

    def update_data(self, ticker, data):
        """Populate stock detail page."""
        if self.pe_canvas is None:
            self._init_widgets()

        self.current_ticker = ticker
        self.title_label.setText(f"{ticker} - Stock Analysis")

        eps_data = data.get('eps')
        fin_data = data.get('financials')
        price = data.get('price')
        iv_data = data.get('iv')
        rates = data.get('rates', {})
        breakeven = data.get('breakeven')

        # Financial Statements (annual + quarterly)
        if fin_data:
            for layout, freq in [(self.fin_annual_layout, 'annual'),
                                 (self.fin_quarterly_layout, 'quarterly')]:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                try:
                    freq_data = {k.replace('_annual', '').replace('_quarterly', ''): v
                                 for k, v in fin_data.items() if freq in k}
                    # Remap keys to match build_financials_table expectations
                    mapped = {}
                    for k, v in fin_data.items():
                        if freq in k:
                            base = k.replace(f'_{freq}', '')
                            mapped[f'{base}_annual'] = v  # reuse _annual keys
                    table = build_financials_table(mapped)
                    if table:
                        layout.addWidget(table)
                except Exception as e:
                    print(f"Stock {ticker}: {freq} financials error: {e}")

        # PE River
        if eps_data is not None and price is not None:
            self.pe_canvas.clear()
            try:
                plot_pe_river(self.pe_canvas.ax, ticker, eps_data, price)
                self.pe_canvas.fig.tight_layout()
            except Exception as e:
                self.pe_canvas.ax.text(0.5, 0.5, f'Error: {e}',
                                       transform=self.pe_canvas.ax.transAxes, ha='center')
            self.pe_canvas.draw()

        # DCF River
        if eps_data is not None and price is not None:
            self.dcf_canvas.clear()
            try:
                plot_dcf_river(self.dcf_canvas.ax, ticker, eps_data,
                               rates, breakeven, iv_data, price)
                self.dcf_canvas.fig.tight_layout()
            except Exception as e:
                self.dcf_canvas.ax.text(0.5, 0.5, f'Error: {e}',
                                        transform=self.dcf_canvas.ax.transAxes, ha='center')
            self.dcf_canvas.draw()

        # EPS
        if eps_data is not None:
            self.eps_canvas.clear()
            try:
                plot_eps_bars(self.eps_canvas.ax, eps_data)
                self.eps_canvas.fig.tight_layout()
            except Exception as e:
                self.eps_canvas.ax.text(0.5, 0.5, f'Error: {e}',
                                        transform=self.eps_canvas.ax.transAxes, ha='center')
            self.eps_canvas.draw()

    def show_loading(self, ticker):
        if self.pe_canvas is None:
            self._init_widgets()
        self.title_label.setText(f"Loading {ticker}...")


# ─── Main Window ──────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial Analyst")
        self.setMinimumSize(1400, 900)

        # Data cache
        self.dashboard_data = None
        self.stock_cache = {}
        self.active_worker = None

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top bar
        top_bar = QHBoxLayout()

        self.btn_dashboard = QPushButton("Dashboard")
        self.btn_dashboard.setFont(QFont('Arial', 11))
        self.btn_dashboard.clicked.connect(self.show_dashboard)
        top_bar.addWidget(self.btn_dashboard)

        top_bar.addWidget(QLabel("  Stock: "))
        self.ticker_combo = QComboBox()
        self.ticker_combo.setFont(QFont('Arial', 11))
        self.ticker_combo.addItems(TICKERS)
        self.ticker_combo.setCurrentIndex(-1)
        self.ticker_combo.setPlaceholderText("Select a stock...")
        self.ticker_combo.currentTextChanged.connect(self.on_ticker_selected)
        top_bar.addWidget(self.ticker_combo)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_data)
        top_bar.addWidget(self.btn_refresh)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Stacked pages
        self.stack = QStackedWidget()
        self.dashboard_page = DashboardPage()
        self.stock_page = StockDetailPage()
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.stock_page)
        main_layout.addWidget(self.stack)

        # Status bar
        self.statusBar().showMessage("Starting up...")

        # Load dashboard data on startup
        self.load_dashboard()

    def load_dashboard(self):
        self.statusBar().showMessage("Loading dashboard data...")
        self.worker = DashboardWorker()
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.on_dashboard_loaded)
        self.worker.start()

    def on_dashboard_loaded(self, data):
        self.dashboard_data = data
        self.dashboard_page.update_data(data)
        self.statusBar().showMessage("Dashboard ready.", 5000)

    def show_dashboard(self):
        self.stack.setCurrentIndex(0)
        self.ticker_combo.setCurrentIndex(-1)

    def on_ticker_selected(self, ticker):
        if not ticker or ticker not in TICKERS:
            return

        self.stack.setCurrentIndex(1)

        # Use cache if available
        if ticker in self.stock_cache:
            self.stock_page.update_data(ticker, self.stock_cache[ticker])
            self.statusBar().showMessage(f"{ticker} loaded from cache.", 3000)
            return

        # Load in background
        self.stock_page.show_loading(ticker)
        self.statusBar().showMessage(f"Loading {ticker}...")

        rates = self.dashboard_data.get('rates') if self.dashboard_data else None
        breakeven = self.dashboard_data.get('breakeven') if self.dashboard_data else None

        self.active_worker = StockWorker(ticker, rates_data=rates, breakeven_data=breakeven)
        self.active_worker.progress.connect(self.statusBar().showMessage)
        self.active_worker.finished.connect(self.on_stock_loaded)
        self.active_worker.start()

    def on_stock_loaded(self, ticker, data):
        self.stock_cache[ticker] = data
        # Only update if this is still the selected ticker
        if self.ticker_combo.currentText() == ticker:
            self.stock_page.update_data(ticker, data)
        self.statusBar().showMessage(f"{ticker} ready.", 5000)

    def refresh_data(self):
        self.stock_cache.clear()
        self.dashboard_data = None
        self.load_dashboard()
        self.statusBar().showMessage("Refreshing all data...")


# ─── Entry Point ──────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)

    # Dark/light system theme
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
