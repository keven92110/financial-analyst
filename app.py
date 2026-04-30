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
    QHeaderView, QSplitter, QGroupBox, QSizePolicy, QTabWidget,
    QDoubleSpinBox
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


# ─── Market Windows Page ──────────────────────────────────────────────
#
# Two sub-tabs:
#   - 今日訊號  (today's classification + historical fwd 5d/20d distribution)
#   - 歷史回測  (user picks method/category/index/horizon → 7-panel grid)
#
# Backend modules live under research/. We import them lazily to avoid
# circular issues during app startup.

class _MarketWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, refresh: bool = False):
        super().__init__()
        self.refresh = refresh

    def run(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research'))
            import predict
            predict.reset_caches()
            data = predict.get_today_signals(refresh=self.refresh)
            self.finished.emit(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class KMeansReferenceWidget(QWidget):
    """Compact reference panel showing the 6 K-means cluster mean paths.

    Today's cluster is drawn bold; others are faded. Used in TodaySignalsTab
    so users can visually compare today's window against the canonical
    cluster shapes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        title = QLabel("<b>K-means reference</b>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 9pt;")
        layout.addWidget(title)

        self.canvas = MplCanvas(self, width=3.0, height=4.0, dpi=90)
        layout.addWidget(self.canvas)
        self.setMaximumWidth(360)
        self.setMinimumWidth(280)

        self.cluster_paths = None  # dict[str -> np.ndarray(30,)]
        self.cluster_meta = None   # dict[str -> dict(N, win_pct)]
        self.highlighted = None
        self._loaded = False

    def ensure_loaded(self):
        if self._loaded:
            return
        try:
            sys.path.insert(0,
                os.path.join(os.path.dirname(__file__), 'research'))
            import predict
            panel = predict.load_panel()
        except Exception as e:
            print(f'KMeansReferenceWidget load error: {e}')
            return
        path_cols = [f'p{i}' for i in range(30)]
        self.cluster_paths = {}
        self.cluster_meta = {}
        clusters = sorted(panel['cat_c'].dropna().unique(),
                          key=lambda x: int(x[1:]))
        for cl in clusters:
            sub = panel[panel['cat_c'] == cl]
            self.cluster_paths[cl] = sub[path_cols].mean().values
            self.cluster_meta[cl] = {
                'n': len(sub),
                'win_pct': float(sub['window_ret'].mean() * 100),
            }
        self._loaded = True
        self._redraw()

    def highlight(self, cluster_label):
        """Highlight one cluster (e.g. 'C4'). None = no highlight."""
        self.highlighted = cluster_label
        if self._loaded:
            self._redraw()

    def _redraw(self):
        if not self.cluster_paths:
            return
        self.canvas.clear()
        ax = self.canvas.ax
        cmap = plt.get_cmap('RdYlGn')
        n = len(self.cluster_paths)
        for i, (cl, path) in enumerate(self.cluster_paths.items()):
            color = cmap(i / max(1, n - 1))
            is_today = (cl == self.highlighted)
            lw = 3.0 if is_today else 1.2
            alpha = 1.0 if is_today else 0.45
            meta = self.cluster_meta.get(cl, {})
            label = f'{cl}  win {meta.get("win_pct", 0):+.0f}%'
            if is_today:
                label += '  ←TODAY'
            ax.plot(range(30), path, color=color, lw=lw, alpha=alpha,
                    label=label)
        ax.axhline(1.0, color='gray', lw=0.4, ls=':')
        ax.set_xlabel('day', fontsize=8)
        ax.set_ylabel('rel. price', fontsize=8)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class _RetrainWorker(QThread):
    """Background worker that re-runs research/run.py to refit K-means
    and rebuild the historical window panel."""
    finished = pyqtSignal(str)   # path to new bundle
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def run(self):
        try:
            import subprocess
            self.progress.emit('Retraining models (this takes ~1-2 min)...')
            research_dir = os.path.join(os.path.dirname(__file__), 'research')
            result = subprocess.run(
                [sys.executable, 'run.py'],
                cwd=research_dir,
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                self.error.emit(f'run.py failed:\n{result.stderr[-2000:]}')
                return
            # Reset caches in predict so new bundle is picked up
            sys.path.insert(0, research_dir)
            import predict
            predict.reset_caches()
            self.finished.emit('OK')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


class TodaySignalsTab(QWidget):
    """Single-screen view: pick an index, see today's classification by all
    three methods overlaid on shared-x fwd_5d / fwd_20d distributions, with
    a K-means cluster reference panel on the right."""

    METHOD_LABELS = {
        'cat_b': 'B · Shape',
        'cat_c': 'C · K-means',
        'cat_d': 'D · RSI',
    }
    # 7 trailing-range overlay colors / alphas / line widths (same as Backtest)
    RANGE_LABELS = ['6M', '12M', '2Y', '5Y', '10Y', '20Y', 'Full']
    RANGE_COLORS = ['#d73027', '#fc8d59', '#fdae61', '#a6d96a',
                    '#66bd63', '#1a9850', '#2c3e50']
    RANGE_ALPHAS = [1.00, 0.92, 0.82, 0.72, 0.60, 0.48, 0.40]
    RANGE_LWS    = [2.4,  2.2,  2.0,  1.8,  1.6,  1.4,  1.6]

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # ── Header row 1: as_of + dropdowns + buttons
        head1 = QHBoxLayout()
        self.as_of_label = QLabel("As of: --  (loading...)")
        self.as_of_label.setFont(QFont('Arial', 10))
        head1.addWidget(self.as_of_label)
        self.model_age_label = QLabel("")
        self.model_age_label.setFont(QFont('Arial', 9))
        self.model_age_label.setStyleSheet("color: #888;")
        head1.addWidget(self.model_age_label)
        head1.addStretch()
        head1.addWidget(QLabel("Index:"))
        self.idx_combo = QComboBox()
        for i in ['ALL_5IDX', 'DJI', 'NDX', 'SOX', 'SPX', 'RUT', 'TWII']:
            self.idx_combo.addItem(i, i)
        self.idx_combo.setCurrentText('ALL_5IDX')
        self.idx_combo.currentIndexChanged.connect(self._on_view_change)
        head1.addWidget(self.idx_combo)
        head1.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem('B · Shape', 'cat_b')
        self.method_combo.addItem('C · K-means', 'cat_c')
        self.method_combo.addItem('D · RSI', 'cat_d')
        # Default to D (RSI) — short-term mean reversion is the most actionable
        self.method_combo.setCurrentIndex(2)
        self.method_combo.currentIndexChanged.connect(self._on_view_change)
        head1.addWidget(self.method_combo)
        self.btn_reload = QPushButton("Refresh data")
        self.btn_reload.clicked.connect(lambda: self.reload(refresh=True))
        head1.addWidget(self.btn_reload)
        self.btn_retrain = QPushButton("Retrain K-means")
        self.btn_retrain.setToolTip(
            "Re-runs research/run.py to refit K-means and rebuild the "
            "historical window panel using all data through today.\n"
            "Takes ~1-2 minutes."
        )
        self.btn_retrain.clicked.connect(self._on_retrain)
        head1.addWidget(self.btn_retrain)
        outer.addLayout(head1)
        self._update_model_age()

        # ── Header row 2: today's classifications
        self.classif_label = QLabel("Today's classification will appear here.")
        self.classif_label.setStyleSheet(
            "padding: 4px 8px; background:#fafbfc; border:1px solid #e0e0e0; "
            "border-radius:4px; font-size: 10pt;")
        self.classif_label.setWordWrap(True)
        outer.addWidget(self.classif_label)

        # ── Body: charts (left) + K-means reference (right)
        body = QHBoxLayout()
        body.setSpacing(6)

        # Left: stacked 5d/20d charts on a single Figure (sharex)
        chart_holder = QWidget()
        ch_layout = QVBoxLayout(chart_holder)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_canvas = MplCanvas(self, width=10, height=7, dpi=100)
        ch_layout.addWidget(self.chart_canvas)
        body.addWidget(chart_holder, stretch=4)

        # Right: K-means reference (compact)
        self.kmeans_ref = KMeansReferenceWidget()
        body.addWidget(self.kmeans_ref, stretch=1)

        outer.addLayout(body, stretch=1)

        self.worker = None
        self._data = None
        self._loaded = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            # Don't call kmeans_ref.ensure_loaded() here — it synchronously
            # reads the 36MB window_panel.csv on the GUI thread and freezes
            # the UI for several seconds. The reload() worker loads the panel
            # in the background; ensure_loaded() is then called from _on_data
            # where it reads from the populated cache (fast).
            self.reload(refresh=False)

    # ── Model-age / retrain plumbing
    def _update_model_age(self):
        try:
            from datetime import datetime
            bundle_path = os.path.join(
                os.path.dirname(__file__),
                'research', 'models', 'classifier_bundle.pkl')
            if not os.path.exists(bundle_path):
                self.model_age_label.setText("·  K-means: not trained")
                return
            mtime = datetime.fromtimestamp(os.path.getmtime(bundle_path))
            age_days = (datetime.now() - mtime).days
            tag = "" if age_days < 30 else " (consider retrain)"
            self.model_age_label.setText(
                f"·  K-means trained {mtime:%Y-%m-%d} ({age_days}d ago){tag}"
            )
        except Exception:
            self.model_age_label.setText("")

    def _on_retrain(self):
        self.btn_retrain.setEnabled(False)
        self.btn_reload.setEnabled(False)
        self.model_age_label.setText("·  Retraining models...")
        self._retrain_worker = _RetrainWorker()
        self._retrain_worker.progress.connect(
            lambda msg: self.model_age_label.setText('·  ' + msg)
        )
        self._retrain_worker.error.connect(self._on_retrain_error)
        self._retrain_worker.finished.connect(self._on_retrain_done)
        self._retrain_worker.start()

    def _on_retrain_error(self, msg: str):
        self.btn_retrain.setEnabled(True)
        self.btn_reload.setEnabled(True)
        self.model_age_label.setText(f"·  Retrain failed: {msg[:100]}")

    def _on_retrain_done(self, _: str):
        self.btn_retrain.setEnabled(True)
        self.btn_reload.setEnabled(True)
        self._update_model_age()
        # Re-fetch today's signals AND refresh K-means reference
        self.kmeans_ref._loaded = False
        self.kmeans_ref.ensure_loaded()
        self.reload(refresh=False)

    # ── Data fetch / render
    def reload(self, refresh: bool = False):
        self.btn_reload.setEnabled(False)
        self.classif_label.setText("Loading today's signals...")
        self.chart_canvas.clear()
        self.chart_canvas.draw()

        self.worker = _MarketWorker(refresh=refresh)
        self.worker.finished.connect(self._on_data)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_error(self, msg: str):
        self.btn_reload.setEnabled(True)
        self.classif_label.setText(
            f"<span style='color:#c0392b'>Error loading data: {msg}</span>")

    def _on_data(self, data: dict):
        self.btn_reload.setEnabled(True)
        self._data = data
        as_of = data.get('as_of_date')
        last_per_idx = data.get('last_date_per_index', {})
        as_of_str = as_of.strftime('%Y-%m-%d') if as_of is not None else '--'
        per_idx_str = ', '.join(f"{k}={v.strftime('%m-%d')}"
                                for k, v in last_per_idx.items())
        self.as_of_label.setText(
            f"As of: {as_of_str}  (per-index latest: {per_idx_str})")
        # Panel is now in predict._panel_cache — safe to load reference widget
        # (reads from cache, doesn't re-parse CSV)
        self.kmeans_ref.ensure_loaded()
        self._render_for_view()

    def _on_view_change(self):
        if self._data is None:
            return
        self._render_for_view()

    def _render_for_view(self):
        """Render charts for the currently selected (Index, Method).
        Uses cached today_data; no network fetch."""
        data = self._data
        if data is None:
            return
        idx_name = self.idx_combo.currentData() or 'ALL_5IDX'
        method = self.method_combo.currentData() or 'cat_d'

        entry = data['indices'].get(idx_name)
        if entry is None:
            self.classif_label.setText(f"<i>No data for {idx_name}.</i>")
            self.chart_canvas.clear(); self.chart_canvas.draw()
            return

        cls = entry.get('classification')   # None for ALL_5IDX

        # K-means reference always tracks today's C cluster of the selected
        # index, regardless of which method is plotted.
        cat_c_for_highlight = None
        if cls is not None:
            cat_c_for_highlight = cls.get('cat_c')
        self.kmeans_ref.highlight(cat_c_for_highlight)

        # Compute today's category for the selected method, plus per-range
        # historical fwd_5d/20d distributions
        try:
            sys.path.insert(0,
                os.path.join(os.path.dirname(__file__), 'research'))
            import predict
        except Exception as e:
            self.classif_label.setText(f"<span style='color:#c0392b'>Import error: {e}</span>")
            return
        try:
            distributions, today_cat = predict.get_today_distributions_by_range(
                method, idx_name, data
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.classif_label.setText(
                f"<span style='color:#c0392b'>Lookup error: {e}</span>")
            return

        # Header label
        if cls is not None:
            head_text = (f'<b>{idx_name}</b>  ·  30d {cls["window_ret_pct"]:+.2f}%  '
                         f'·  vol(yr) {cls["window_vol_pct"]:.1f}%  '
                         f'·  RSI14 {cls["rsi14"]:.1f}'
                         f'  ·  window {cls["dates"][0].strftime("%m/%d")}–'
                         f'{cls["dates"][-1].strftime("%m/%d")}')
        else:
            head_text = f'<b>{idx_name}</b>  ·  combined view (5 indices, pooled)'

        cat_disp = today_cat or '—'
        if len(cat_disp) > 90:
            cat_disp = cat_disp[:87] + '...'
        method_label = self.METHOD_LABELS[method]
        cat_html = (f"<span style='font-size:11pt'>"
                    f"<b>Today's {method_label}:</b> "
                    f"<span style='color:#c0392b'>{cat_disp}</span>"
                    f"</span>")
        self.classif_label.setText(head_text + '<br>' + cat_html)

        # Draw 7-range overlay charts
        self._draw_range_overlay(distributions, idx_name, method, today_cat)

    def _draw_range_overlay(self, distributions: dict, idx_name: str,
                            method: str, today_cat):
        """fwd_5d on top, fwd_20d on bottom, shared x-axis;
        overlay 7 trailing ranges (recent=warm, older=cool)."""
        fig = self.chart_canvas.fig
        fig.clear()
        ax5 = fig.add_subplot(2, 1, 1)
        ax20 = fig.add_subplot(2, 1, 2, sharex=ax5)

        # Determine common x-range across BOTH horizons & all 7 ranges
        all_vals = []
        for label in self.RANGE_LABELS:
            stats = distributions.get(label, {})
            for hkey in ('fwd_5d', 'fwd_20d'):
                v = stats.get(hkey, {}).get('vals')
                if v is not None and len(v):
                    all_vals.append(v)
        if all_vals:
            comb = np.concatenate(all_vals) * 100
            xlo = np.percentile(comb, 0.5)
            xhi = np.percentile(comb, 99.5)
            span = max(abs(xlo), abs(xhi))
            xlo, xhi = -span, span
        else:
            xlo, xhi = -10, 10

        n_bins = 45
        edges = np.linspace(xlo, xhi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        for ax, hkey, title in ((ax5, 'fwd_5d', 'Forward 5-day return  (probability)'),
                                (ax20, 'fwd_20d', 'Forward 20-day return  (probability)')):
            plotted = 0
            for i, label in enumerate(self.RANGE_LABELS):
                stats = distributions.get(label, {}).get(hkey, {})
                vals = stats.get('vals')
                n = stats.get('n', 0)
                if vals is None or len(vals) == 0:
                    continue
                arr = vals * 100
                counts, _ = np.histogram(arr, bins=edges)
                if counts.sum() == 0:
                    continue
                pmf = counts / counts.sum()
                warn = ' ⚠️' if 0 < n < 30 else ''
                leg = (f'{label:>4s}{warn}  N={n:>5,}  '
                       f'μ={stats["mean_pct"]:+.2f}%  '
                       f'hit={stats["hit_rate"]:.0f}%')
                ax.plot(centers, pmf, color=self.RANGE_COLORS[i],
                        lw=self.RANGE_LWS[i], alpha=self.RANGE_ALPHAS[i],
                        label=leg, drawstyle='steps-mid')
                ax.fill_between(centers, 0, pmf, color=self.RANGE_COLORS[i],
                                alpha=self.RANGE_ALPHAS[i] * 0.15, step='mid')
                plotted += 1
            ax.axvline(0, color='gray', lw=0.7, ls=':')
            ax.set_ylabel('probability', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)
            if plotted:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.92,
                          title='trailing range  (warm=recent, cool=older)',
                          title_fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No matching historical windows.',
                        ha='center', va='center',
                        transform=ax.transAxes, color='#888')
            ax.set_title(title, fontsize=10, loc='left')

        ax5.tick_params(axis='x', labelbottom=False)
        ax20.set_xlabel('forward return (%)', fontsize=10)
        # Big overall title with what's being plotted
        cat_disp = (today_cat or '—')
        if len(cat_disp) > 60:
            cat_disp = cat_disp[:57] + '...'
        fig.suptitle(
            f'{idx_name}  ·  {self.METHOD_LABELS[method]}  ·  '
            f'today = {cat_disp}',
            fontsize=11, y=0.995
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.chart_canvas.draw()


class _BacktestInitWorker(QThread):
    """One-shot worker that loads the historical window panel into the
    predict module's cache, then returns the category list for the
    initially-selected (method, index)."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, method: str, idx: str):
        super().__init__()
        self.method = method
        self.idx = idx

    def run(self):
        try:
            sys.path.insert(0,
                os.path.join(os.path.dirname(__file__), 'research'))
            import predict
            panel = predict.load_panel()    # populates cache
            cats = predict.get_categories_for_method(panel, self.method,
                                                     index_name=self.idx)
            self.finished.emit(list(cats))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


class BacktestTab(QWidget):
    """User-driven backtest viewer with 7 trailing windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)

        # Top control bar
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel('Method:'))
        self.method_combo = QComboBox()
        self.method_combo.addItem('B · Shape', 'cat_b')
        self.method_combo.addItem('C · K-means', 'cat_c')
        self.method_combo.addItem('D · RSI', 'cat_d')
        self.method_combo.currentIndexChanged.connect(self._refresh_categories)
        ctrl.addWidget(self.method_combo)

        ctrl.addWidget(QLabel('Category:'))
        self.cat_combo = QComboBox()
        ctrl.addWidget(self.cat_combo)

        ctrl.addWidget(QLabel('Index:'))
        self.idx_combo = QComboBox()
        for i in ['ALL_5IDX', 'SPX', 'DJI', 'NDX', 'RUT', 'SOX', 'TWII']:
            self.idx_combo.addItem(i, i)
        self.idx_combo.currentIndexChanged.connect(self._refresh_categories)
        ctrl.addWidget(self.idx_combo)

        ctrl.addWidget(QLabel('Horizon:'))
        self.horizon_combo = QComboBox()
        self.horizon_combo.addItem('5d', 'fwd_5d')
        self.horizon_combo.addItem('20d', 'fwd_20d')
        ctrl.addWidget(self.horizon_combo)

        self.btn_run = QPushButton('Plot')
        self.btn_run.clicked.connect(self._run)
        ctrl.addWidget(self.btn_run)
        ctrl.addStretch()
        outer.addLayout(ctrl)

        # Plot area
        self.canvas = MplCanvas(self, width=12, height=7, dpi=100)
        outer.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: #888; font-size: 9pt;')
        outer.addWidget(self.status_label)

        # Init
        self._loaded = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            # First load uses an init worker so we don't block the UI thread
            # on the 36MB window_panel.csv read. After init, _refresh_categories
            # reads from the cached panel (fast, can stay synchronous).
            self.btn_run.setEnabled(False)
            self.status_label.setText('Loading window panel...')
            self._init_worker = _BacktestInitWorker(
                self.method_combo.currentData(),
                self.idx_combo.currentData(),
            )
            self._init_worker.finished.connect(self._on_init_done)
            self._init_worker.error.connect(self._on_init_error)
            self._init_worker.start()

    def _on_init_done(self, cats: list):
        self.btn_run.setEnabled(True)
        self.status_label.setText('')
        self.cat_combo.clear()
        self.cat_combo.addItems(cats)
        # Now panel is cached → trigger initial plot (uses fast in-memory path)
        self._run()

    def _on_init_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.status_label.setText(f'Init error: {msg}')

    def _refresh_categories(self):
        """Synchronous; only safe to call AFTER first init has populated cache."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research'))
            import predict
        except Exception:
            return
        method = self.method_combo.currentData()
        idx = self.idx_combo.currentData()
        try:
            panel = predict.load_panel()   # reads from cache after first init
            cats = predict.get_categories_for_method(panel, method,
                                                     index_name=idx)
        except Exception:
            cats = []
        prev = self.cat_combo.currentText()
        self.cat_combo.clear()
        self.cat_combo.addItems(cats)
        if prev in cats:
            self.cat_combo.setCurrentText(prev)

    def _run(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research'))
            import predict
        except Exception as e:
            self.status_label.setText(f'Import error: {e}')
            return
        method = self.method_combo.currentData()
        cat = self.cat_combo.currentText()
        idx = self.idx_combo.currentData()
        horizon = self.horizon_combo.currentData()
        if not cat:
            self.status_label.setText('No category selected.')
            return
        try:
            data = predict.get_backtest_distributions(method, cat, idx, horizon)
        except Exception as e:
            self.status_label.setText(f'Error: {e}')
            return
        self._plot(data, method, cat, idx, horizon)

    def _plot(self, data: dict, method: str, cat: str,
              idx: str, horizon: str):
        """All time-ranges overlaid as PMF curves (sum to 1).

        Older ranges are more transparent so the recent ones stand out.
        """
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        labels = ['6M', '12M', '2Y', '5Y', '10Y', '20Y', 'Full']
        # Color: warm (red/orange) = recent → cool (blue) = old
        cmap = ['#d73027', '#fc8d59', '#fdae61', '#a6d96a',
                '#66bd63', '#1a9850', '#2c3e50']
        # Transparency: recent = solid, old = lighter
        alphas = [1.00, 0.92, 0.82, 0.72, 0.60, 0.48, 0.40]
        # Line widths: recent thicker, old thinner
        lws = [2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.6]

        # Common x-range (1–99 percentile of all combined data)
        all_vals = []
        for lbl in labels:
            v = data.get(lbl, {}).get('vals')
            if v is not None and len(v):
                all_vals.append(v)
        if all_vals:
            comb = np.concatenate(all_vals) * 100
            xmin = np.percentile(comb, 0.5)
            xmax = np.percentile(comb, 99.5)
            # Symmetrize a bit so 0 doesn't sit at edge
            span = max(abs(xmin), abs(xmax))
            xmin, xmax = -span, span
        else:
            xmin, xmax = -5, 5

        n_bins = 40
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        plotted = 0
        for i, lbl in enumerate(labels):
            stats = data.get(lbl, {})
            vals = stats.get('vals')
            n = stats.get('n', 0)
            if vals is None or len(vals) == 0:
                continue
            arr = vals * 100
            counts, _ = np.histogram(arr, bins=edges)
            pmf = counts / counts.sum()  # probabilities sum to 1

            warn = ' ⚠️' if 0 < n < 30 else ''
            label = (f'{lbl:>4s}{warn}  N={n:>5,}  '
                     f'μ={stats["mean_pct"]:+.2f}%  '
                     f'hit={stats["hit_rate"]:.0f}%')

            ax.plot(centers, pmf,
                    color=cmap[i], lw=lws[i], alpha=alphas[i],
                    label=label, drawstyle='steps-mid')
            ax.fill_between(centers, 0, pmf, color=cmap[i],
                            alpha=alphas[i] * 0.15, step='mid')
            plotted += 1

        ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
        ax.set_xlabel(f'fwd {horizon.replace("fwd_", "")} return (%)',
                      fontsize=11)
        ax.set_ylabel('Probability  (sum = 1)', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.92,
                  title='trailing range  (recent = warm, older = cooler)',
                  title_fontsize=9)

        title = f'{method.upper()} · {cat}   ·   index = {idx}   ·   horizon = {horizon}'
        ax.set_title(title, fontsize=12, pad=12)
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        self.status_label.setText(
            f'Plotted {plotted}/7 ranges · {method.upper()} · {cat} · {idx} · {horizon}'
        )


class _IVWorker(QThread):
    """Background worker that fetches today's IV (CBOE vol indices) and
    today's σ_emp from our model, then builds a comparison table."""
    finished = pyqtSignal(object, dict)   # (DataFrame, today_data)
    error = pyqtSignal(str)

    def run(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research'))
            import predict
            from iv import iv_compare
            predict.reset_caches()
            today_data = predict.get_today_signals()
            df = iv_compare.compare_today(today_data)
            self.finished.emit(df, today_data)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


class IVComparisonTab(QWidget):
    """Tab showing market IV (CBOE vol indices) vs our model's σ_emp.

    Useful for spotting potential variance risk premium (VRP) opportunities,
    though the historical backtest shows our σ does NOT meaningfully filter
    'good days' to sell vol — the broad VIX-vs-RV pattern dominates.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Header
        head = QHBoxLayout()
        title = QLabel("<b>IV vs Empirical σ Comparison</b>")
        title.setStyleSheet("font-size: 12pt;")
        head.addWidget(title)
        head.addStretch()
        self.btn_reload = QPushButton("Refresh")
        self.btn_reload.clicked.connect(self.reload)
        head.addWidget(self.btn_reload)
        outer.addLayout(head)

        # Status / explanation
        explain = QLabel(
            "<i>VRP = market IV − our σ_emp.  "
            "Positive VRP suggests market is pricing more vol than our historical "
            "RSI-bucket distribution.  σ_emp uses cat_d (RSI) full-history match "
            "and is annualized.  Note: VIX historically captures realized vol "
            "much better than our σ_emp (correlation 0.71 vs 0.27), so this is "
            "informational rather than a clear edge signal.</i>"
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: #555; font-size: 9pt; padding: 4px;")
        outer.addWidget(explain)

        # Table
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        outer.addWidget(self.table, stretch=1)

        # Status line
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")
        outer.addWidget(self.status_label)

        self._loaded = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            self.reload()

    def reload(self):
        self.btn_reload.setEnabled(False)
        self.status_label.setText("Loading IV data...")
        self.worker = _IVWorker()
        self.worker.finished.connect(self._render)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_error(self, msg: str):
        self.btn_reload.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")

    def _render(self, df, today_data):
        self.btn_reload.setEnabled(True)
        as_of = today_data.get('as_of_date')
        as_of_str = as_of.strftime('%Y-%m-%d') if as_of is not None else '--'

        cols = ['Index', "Today's RSI bucket",
                'σ_emp 5d (ann %)', 'IV 9d (%)', 'VRP 5d (%)',
                'σ_emp 20d (ann %)', 'IV 30d (%)', 'VRP 20d (%)',
                'N (5d / 20d)']
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(df))
        for r in range(len(df)):
            row = df.iloc[r]
            def make_item(text, color=None, bold=False, align_right=True):
                it = QTableWidgetItem(text)
                if align_right:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                if color:
                    it.setForeground(QColor(color))
                if bold:
                    f = it.font(); f.setBold(True); it.setFont(f)
                return it

            self.table.setItem(r, 0, make_item(row['index'], bold=True, align_right=False))
            self.table.setItem(r, 1, make_item(str(row['cat_d']), align_right=False))
            self.table.setItem(r, 2, make_item(_fmt_pct(row['sigma_emp_5d_pct'])))
            self.table.setItem(r, 3, make_item(_fmt_pct(row['iv_9d_pct'])))
            self.table.setItem(r, 4, make_item(_fmt_pct(row['vrp_5d_pct'], signed=True),
                                                color=_vrp_color(row['vrp_5d_pct'])))
            self.table.setItem(r, 5, make_item(_fmt_pct(row['sigma_emp_20d_pct'])))
            self.table.setItem(r, 6, make_item(_fmt_pct(row['iv_30d_pct'])))
            self.table.setItem(r, 7, make_item(_fmt_pct(row['vrp_20d_pct'], signed=True),
                                                color=_vrp_color(row['vrp_20d_pct'])))
            n_str = f"{int(row['n_5d']):,} / {int(row['n_20d']):,}"
            self.table.setItem(r, 8, make_item(n_str, align_right=False))
        self.table.resizeColumnsToContents()

        # Status / summary line
        valid_vrp_20 = df['vrp_20d_pct'].dropna()
        if len(valid_vrp_20):
            mean_vrp = valid_vrp_20.mean()
            sign = "MORE" if mean_vrp > 0 else "LESS"
            self.status_label.setText(
                f"As of: {as_of_str}    Mean VRP_20d (where IV available) = "
                f"{mean_vrp:+.2f}%  →  market expects {sign} vol than our model on average. "
                f"Historical mean VRP (VIX-RV) ≈ +3.8%.  "
                f"RUT/SOX have no CBOE vol-index in yfinance, IV shown as --."
            )
        else:
            self.status_label.setText(f"As of: {as_of_str}")


def _fmt_pct(v, signed=False) -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '--'
        return f'{v:+.2f}' if signed else f'{v:.2f}'
    except Exception:
        return '--'


def _vrp_color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if v > 2:
        return '#1a7e1a'   # green = market richer (sell vol potential)
    if v < -2:
        return '#c0392b'   # red = market cheaper (buy vol potential)
    return None


class _StrategyWorker(QThread):
    """Background worker that builds today's distribution + ranks strategies."""
    finished = pyqtSignal(dict, dict, str, str)   # eval_result, today_data, idx, horizon
    error = pyqtSignal(str)

    def __init__(self, idx_name: str, horizon: str,
                 method: str = 'cat_d', sigma_mult: float = 1.0):
        super().__init__()
        self.idx_name = idx_name
        self.horizon = horizon
        self.method = method
        self.sigma_mult = sigma_mult

    def run(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research'))
            import predict
            from iv import strategy_eval

            data = predict.get_today_signals()

            # Get distribution samples for the chosen index/method/horizon
            entry = data['indices'].get(self.idx_name)
            if entry is None:
                raise RuntimeError(f'No data for {self.idx_name}')
            preds = entry.get('predictions', {})
            mp = preds.get(self.method, {})
            stats = mp.get(self.horizon, {})
            samples = stats.get('vals')
            if samples is None or len(samples) == 0:
                raise RuntimeError(f'No samples for {self.idx_name}/{self.method}/{self.horizon}')

            # Use last_close from this index (or any) as spot
            cls = entry.get('classification') or {}
            S0 = cls.get('last_close')
            if S0 is None:
                # ALL_5IDX has no single spot; use SPX as proxy
                spx_entry = data['indices'].get('SPX', {})
                S0 = (spx_entry.get('classification') or {}).get('last_close', 100)

            result = strategy_eval.find_best_strategies(
                np.asarray(samples), float(S0),
                pricing='empirical', top_k=15, metric='pop',
                primary_sigma_mult=self.sigma_mult,
            )
            # Also fetch per-range distributions so the chart background can
            # show recent vs older history as overlaid PMFs (matches the
            # "Today" / "Backtest" tabs).
            try:
                range_dists, _ = predict.get_today_distributions_by_range(
                    self.method, self.idx_name, data
                )
                result['range_dists'] = range_dists
            except Exception as e:
                print(f'range_dists fetch failed: {e}')
                result['range_dists'] = {}
            result['horizon_key'] = self.horizon
            self.finished.emit(result, data, self.idx_name, self.horizon)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


class OptionStrategyTab(QWidget):
    """Show the empirical-fair-priced option strategies under our distribution.

    Pivot insight: under empirical fair pricing every strategy has E[P&L]=0,
    so what's interesting is the SHAPE of P&L (PoP, max gain, max loss).
    The distribution skew/kurtosis from our model directly implies which
    payoff shapes are 'attractive' for the user's risk preference.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        # Header controls
        head = QHBoxLayout()
        head.addWidget(QLabel('<b>Option Strategy Builder</b>'))
        head.addSpacing(20)
        head.addWidget(QLabel('Index:'))
        self.idx_combo = QComboBox()
        for i in ['ALL_5IDX', 'SPX', 'NDX', 'DJI', 'RUT', 'SOX', 'TWII']:
            self.idx_combo.addItem(i, i)
        self.idx_combo.setCurrentText('SPX')
        head.addWidget(self.idx_combo)
        head.addWidget(QLabel('Method:'))
        self.method_combo = QComboBox()
        self.method_combo.addItem('B · Shape',  'cat_b')
        self.method_combo.addItem('C · K-means', 'cat_c')
        self.method_combo.addItem('D · RSI',    'cat_d')
        self.method_combo.setCurrentIndex(2)
        head.addWidget(self.method_combo)
        head.addWidget(QLabel('Horizon:'))
        self.h_combo = QComboBox()
        self.h_combo.addItem('5d', 'fwd_5d')
        self.h_combo.addItem('20d', 'fwd_20d')
        self.h_combo.setCurrentIndex(1)
        head.addWidget(self.h_combo)
        head.addWidget(QLabel('Sort by:'))
        self.sort_combo = QComboBox()
        self.sort_combo.addItem('PoP %',  'pop')
        self.sort_combo.addItem('Sharpe', 'sharpe')
        self.sort_combo.addItem('E[P&L]', 'e_pnl')
        head.addWidget(self.sort_combo)
        # σ multiplier input — strikes are placed at ±k×σ_log of our distribution
        head.addWidget(QLabel('σ mult:'))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setRange(0.10, 5.00)
        self.sigma_spin.setSingleStep(0.25)
        self.sigma_spin.setValue(1.00)
        self.sigma_spin.setSuffix('  (× σ_log)')
        self.sigma_spin.setToolTip(
            'Strikes used by candidates are placed at multiples of this σ.\n'
            '  1.0  → ±0.5σ, ±1σ, ±1.5σ, ±2σ around spot\n'
            '  0.5  → ±0.25σ, ±0.5σ, ±0.75σ, ±1σ\n'
            '  2.0  → ±1σ, ±2σ, ±3σ, ±4σ\n'
            '\n(changes auto-trigger re-search)'
        )
        # Auto re-run when sigma changes (uses cached today_data so it's fast)
        self.sigma_spin.valueChanged.connect(self._on_sigma_change)
        head.addWidget(self.sigma_spin)
        self.btn_run = QPushButton('Find strategies')
        self.btn_run.clicked.connect(self._run)
        head.addWidget(self.btn_run)
        head.addStretch()
        outer.addLayout(head)

        # Status / dist info
        self.info_label = QLabel(
            '<i>Empirical fair pricing: option price = E[payoff] under our '
            'distribution.  All strategies have E[P&L]=0 by construction; '
            'compare them by SHAPE (PoP, max gain/loss, breakevens).</i>')
        self.info_label.setStyleSheet('color: #555; font-size: 9pt; padding: 2px;')
        self.info_label.setWordWrap(True)
        outer.addWidget(self.info_label)

        # Splitter: top (table left, charts right), bottom (P&L histogram)
        body = QHBoxLayout()
        body.setSpacing(8)

        # Left: strategy table
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(450)
        body.addWidget(self.table, stretch=2)

        # Right: single canvas with two stacked sharex subplots
        right = QVBoxLayout()
        right.setSpacing(4)
        self.canvas_combined = MplCanvas(self, width=8, height=8, dpi=100)
        right.addWidget(self.canvas_combined)
        body.addLayout(right, stretch=3)

        outer.addLayout(body, stretch=1)

        self._loaded = False
        self._result = None
        self._S0 = None
        self._last_today_data = None

    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self._loaded = True
            self._run()

    def _run(self):
        idx = self.idx_combo.currentData() or 'SPX'
        method = self.method_combo.currentData() or 'cat_d'
        horizon = self.h_combo.currentData() or 'fwd_20d'
        sigma_mult = float(self.sigma_spin.value())
        self.btn_run.setEnabled(False)
        self.info_label.setText('Computing strategies...')
        self.worker = _StrategyWorker(idx, horizon, method, sigma_mult)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_sigma_change(self):
        """Cheap re-run: reuse cached today_data (no yfinance fetch) and
        just rebuild candidates with the new sigma multiplier."""
        if self._result is None:
            # Not loaded yet → defer to full _run
            return
        # Inline re-evaluate using cached samples
        try:
            sys.path.insert(0,
                os.path.join(os.path.dirname(__file__), 'research'))
            from iv import strategy_eval
        except Exception:
            return
        idx = self.idx_combo.currentData() or 'SPX'
        method = self.method_combo.currentData() or 'cat_d'
        horizon = self.h_combo.currentData() or 'fwd_20d'
        sigma_mult = float(self.sigma_spin.value())
        # We need today_data to fetch samples - it was passed to _on_done
        if not hasattr(self, '_last_today_data') or self._last_today_data is None:
            return
        entry = self._last_today_data['indices'].get(idx)
        if entry is None: return
        preds = entry.get('predictions', {})
        samples = preds.get(method, {}).get(horizon, {}).get('vals')
        if samples is None or len(samples) == 0: return
        result = strategy_eval.find_best_strategies(
            np.asarray(samples), float(self._S0),
            pricing='empirical', top_k=15, metric='pop',
            primary_sigma_mult=sigma_mult,
        )
        # Range distributions don't change with sigma — copy from cached result
        if self._result is not None:
            result['range_dists'] = self._result.get('range_dists', {})
        result['horizon_key'] = horizon
        self._on_done(result, self._last_today_data, idx, horizon)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.info_label.setText(f"<span style='color:#c0392b'>Error: {msg}</span>")

    def _on_done(self, result, today_data, idx, horizon):
        self.btn_run.setEnabled(True)
        self._result = result
        self._last_today_data = today_data   # cache for sigma-change re-runs
        self._S0 = result['S0']
        S_T = result['S_T']
        chain = result['chain']
        idx_name = idx; h = horizon
        method = self.method_combo.currentData()

        # Distribution stats
        log_returns = np.log(S_T / self._S0)
        mu = log_returns.mean() * 100
        sigma = log_returns.std() * 100
        p_up = (S_T > self._S0).mean() * 100
        sigma_mult = float(self.sigma_spin.value())
        # σ_mult corresponds to a price band: spot × exp(±σ_mult × σ_log)
        band_lo = self._S0 * float(np.exp(-sigma_mult * (sigma / 100)))
        band_hi = self._S0 * float(np.exp(+sigma_mult * (sigma / 100)))
        # Today's category
        entry = today_data['indices'].get(idx_name) or {}
        cls = entry.get('classification') or {}
        cat = (cls.get(method) or '—')
        info = (
            f"<b>{idx_name}</b>  ·  S0=<b>{self._S0:,.2f}</b>  ·  horizon={h}  "
            f"·  today's cat={cat}<br>"
            f"<i>Distribution:</i> μ_log={mu:+.2f}%  σ_log={sigma:.2f}%  "
            f"P(up)={p_up:.1f}%  N={len(S_T):,}  "
            f"<br><i>Strikes</i>: ±{sigma_mult:g}σ → "
            f"[<b>{band_lo:,.0f}</b>, <b>{band_hi:,.0f}</b>]  "
            f"(also at 0.5×, 1.5×, 2× of this).  "
            f"Pricing: empirical fair (E[payoff] under our distribution)"
        )
        self.info_label.setText(info)

        # Re-sort by selected metric
        metric = self.sort_combo.currentData()
        results = sorted(result['all'],
                         key=lambda r: r[metric], reverse=True)[:15]
        # Clear selection first so selectRow(0) fires the change signal even
        # when row 0 was already selected (otherwise charts won't redraw).
        self.table.clearSelection()
        self._populate_table(results)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
            # Belt-and-braces: force redraw even if signal fires twice
            top_item = self.table.item(0, 0)
            if top_item is not None:
                top_res = top_item.data(Qt.UserRole)
                if top_res is not None:
                    self._draw_payoff(top_res)

    def _populate_table(self, results):
        cols = ['Strategy', 'Cost', 'PoP %', 'E[P&L]',
                'Max gain', 'Max loss', 'P10', 'P90']
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(results))
        for r, res in enumerate(results):
            def make(text, align_right=True, color=None, bold=False):
                it = QTableWidgetItem(text)
                if align_right:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if color:
                    it.setForeground(QColor(color))
                if bold:
                    f = it.font(); f.setBold(True); it.setFont(f)
                return it
            self.table.setItem(r, 0, make(res['name'], align_right=False))
            self.table.setItem(r, 1, make(f"{res['cost']:+.2f}"))
            self.table.setItem(r, 2, make(f"{res['pop']:.1f}"))
            self.table.setItem(r, 3, make(f"{res['e_pnl']:+.4f}", bold=True))
            self.table.setItem(r, 4, make(f"{res['max_gain']:+.2f}",
                                          color='#1a7e1a' if res['max_gain']>0 else None))
            self.table.setItem(r, 5, make(f"{res['max_loss']:+.2f}",
                                          color='#c0392b' if res['max_loss']<-1 else None))
            self.table.setItem(r, 6, make(f"{res['p10']:+.2f}"))
            self.table.setItem(r, 7, make(f"{res['p90']:+.2f}"))
            self.table.item(r, 0).setData(Qt.UserRole, res)
        self.table.resizeColumnsToContents()

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        r = rows[0].row()
        item = self.table.item(r, 0)
        if not item: return
        res = item.data(Qt.UserRole)
        if not res: return
        self._draw_payoff(res)

    def _draw_payoff(self, res: dict):
        """Draw payoff diagram (top) + expected-contribution curve (bottom)
        on a single shared-x figure."""
        if self._result is None: return
        S_T = self._result['S_T']
        S0 = self._S0
        strat = res['strategy']
        from iv.strategy_eval import expected_contribution_curve

        fig = self.canvas_combined.fig
        fig.clear()
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

        # ── Common x-range based on price distribution
        s_lo = float(np.percentile(S_T, 0.5)) * 0.97
        s_hi = float(np.percentile(S_T, 99.5)) * 1.03
        grid = np.linspace(s_lo, s_hi, 400)
        pnl_grid = strat.pnl(grid)

        # ── TOP: Payoff diagram with distribution overlay
        ax_top.fill_between(grid, 0, pnl_grid, where=(pnl_grid >= 0),
                            color='#27ae60', alpha=0.25, label='profit')
        ax_top.fill_between(grid, 0, pnl_grid, where=(pnl_grid < 0),
                            color='#c0392b', alpha=0.25, label='loss')
        ax_top.plot(grid, pnl_grid, color='black', lw=1.5)
        ax_top.axhline(0, color='gray', lw=0.6)
        ax_top.axvline(S0, color='blue', lw=0.8, ls='--',
                       label=f'spot {S0:.0f}')
        for be in res.get('breakevens', []):
            ax_top.axvline(be, color='purple', lw=0.6, ls=':', alpha=0.7)

        # ── Per-leg strike markers + labels.
        # Group by strike so legs at the same strike (e.g. iron butterfly's
        # short put + short call both at K_atm) don't overlap visually.
        from collections import defaultdict
        legs_by_strike = defaultdict(list)
        for leg in strat.legs:
            legs_by_strike[leg.strike].append(leg)

        for strike, legs_here in legs_by_strike.items():
            # Mixed sides → grey line; uniform side → side-tinted line
            sides = {l.side for l in legs_here}
            if len(sides) == 1:
                line_color = '#1a7e1a' if 'short' in sides else '#c0392b'
            else:
                line_color = '#555'
            ax_top.axvline(strike, color=line_color, lw=0.8,
                           ls='-.', alpha=0.45)

            # Build combined label, one line per leg at this strike
            label_lines = [f'K={strike:.0f}']
            for leg in legs_here:
                sign = '+L' if leg.side == 'long' else '-S'
                typ = leg.type[0].upper()
                label_lines.append(f'{sign}{typ} @ {leg.premium:.2f}')
            label = '\n'.join(label_lines)

            # Color of bbox: side-specific if uniform, else neutral
            box_color = (line_color if len(sides) == 1 else '#666')
            ax_top.annotate(
                label,
                xy=(strike, 0),
                xytext=(0, 14), textcoords='offset points',
                fontsize=7, ha='center', va='bottom',
                color=box_color,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor=box_color,
                          linewidth=0.7, alpha=0.9),
            )

        # ── Per-range distribution overlay on twin axis (same colour scheme
        #    as Today / Backtest tabs: recent ranges opaque, older transparent)
        ax_top2 = ax_top.twinx()
        range_dists = self._result.get('range_dists', {}) if self._result else {}
        horizon_key = self._result.get('horizon_key', 'fwd_20d') if self._result else 'fwd_20d'

        bins = np.linspace(s_lo, s_hi, 50)
        centers = (bins[:-1] + bins[1:]) / 2
        # Same palette / alphas used elsewhere in the app
        R_LABELS = ['6M', '12M', '2Y', '5Y', '10Y', '20Y', 'Full']
        R_COLORS = ['#d73027', '#fc8d59', '#fdae61', '#a6d96a',
                    '#66bd63', '#1a9850', '#2c3e50']
        R_ALPHAS = [1.00, 0.92, 0.82, 0.72, 0.60, 0.48, 0.40]
        R_LWS    = [1.6,  1.5,  1.4,  1.3,  1.2,  1.1,  1.2]

        plotted_any = False
        for i, lbl in enumerate(R_LABELS):
            rd = range_dists.get(lbl, {}).get(horizon_key, {})
            vals = rd.get('vals')
            if vals is None or len(vals) == 0:
                continue
            n = len(vals)
            range_S_T = S0 * np.exp(np.asarray(vals))
            counts, _ = np.histogram(range_S_T, bins=bins)
            if counts.sum() == 0:
                continue
            pmf = counts / counts.sum()
            ax_top2.plot(centers, pmf,
                         color=R_COLORS[i], lw=R_LWS[i],
                         alpha=R_ALPHAS[i],
                         drawstyle='steps-mid',
                         label=f'{lbl} (N={n:,})')
            ax_top2.fill_between(centers, 0, pmf,
                                 color=R_COLORS[i],
                                 alpha=R_ALPHAS[i] * 0.12,
                                 step='mid')
            plotted_any = True

        if not plotted_any:
            # Fallback: single hist of all S_T (e.g. range_dists empty)
            ax_top2.hist(S_T, bins=60, density=True, alpha=0.18,
                         color='#4575b4', edgecolor='none')

        ax_top2.set_ylabel('probability per range', fontsize=8, color='#444')
        ax_top2.tick_params(axis='y', labelsize=7, colors='#444')
        if plotted_any:
            ax_top2.legend(loc='upper right', fontsize=7,
                           framealpha=0.85, ncol=2,
                           title='trailing range',
                           title_fontsize=7)
        ax_top.set_ylabel('P&L per share')

        # Two-line title: line 1 = strategy stats, line 2 = leg breakdown
        leg_strs = []
        for leg in strat.legs:
            sign = '+L' if leg.side == 'long' else '-S'
            typ = leg.type[0].upper()
            leg_strs.append(f"{sign} {typ}{leg.strike:.0f} @ {leg.premium:.2f}")
        cost = res['cost']
        if abs(cost) < 0.01:
            cost_label = '≈0'
        elif cost < 0:
            cost_label = 'credit'
        else:
            cost_label = 'debit'
        legs_line = ('   |   '.join(leg_strs)
                     + f"   →   net {cost:+.2f} ({cost_label})")
        ax_top.set_title(
            f"{res['name']}   PoP={res['pop']:.1f}%   "
            f"max±=({res['max_gain']:+.1f}, {res['max_loss']:+.1f})\n"
            f"{legs_line}",
            fontsize=10
        )
        ax_top.legend(loc='upper left', fontsize=8)
        ax_top.grid(alpha=0.25)
        # Hide top x-axis tick labels (shared with bottom)
        ax_top.tick_params(axis='x', labelbottom=False)

        # ── BOTTOM: Expected-contribution curve  P&L(S_T) × pdf(S_T)
        # Sum of this curve = E[P&L]  (always ≈0 under empirical fair pricing)
        ec = expected_contribution_curve(strat, S_T)
        contrib = ec['contribution']
        ax_bot.fill_between(ec['grid'], 0, contrib, where=(contrib >= 0),
                            color='#27ae60', alpha=0.55,
                            label='profit contribution')
        ax_bot.fill_between(ec['grid'], 0, contrib, where=(contrib < 0),
                            color='#c0392b', alpha=0.55,
                            label='loss contribution')
        ax_bot.plot(ec['grid'], contrib, color='black', lw=1.0)
        ax_bot.axhline(0, color='gray', lw=0.6)
        ax_bot.axvline(S0, color='blue', lw=0.6, ls='--', alpha=0.6)
        for be in res.get('breakevens', []):
            ax_bot.axvline(be, color='purple', lw=0.5, ls=':', alpha=0.5)
        ax_bot.set_xlabel('underlying price at expiry')
        ax_bot.set_ylabel('P&L × probability')
        ax_bot.set_title(
            f"Expected contribution per price   "
            f"(sum of green + red = E[P&L] = {res['e_pnl']:+.4f})",
            fontsize=10
        )
        ax_bot.legend(loc='best', fontsize=8)
        ax_bot.grid(alpha=0.25)

        # Force a clean x-range
        ax_top.set_xlim(s_lo, s_hi)

        fig.tight_layout()
        self.canvas_combined.draw()


class MarketWindowsPage(QWidget):
    """Container for sub-tabs:
       1. 今日訊號  — today's classification + 7-range overlay charts
       2. 歷史回測  — user-driven backtest
       3. IV vs σ — market IV vs our empirical σ
       4. Strategy Builder — option strategies under our empirical distribution
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.today_tab = TodaySignalsTab()
        self.backtest_tab = BacktestTab()
        self.iv_tab = IVComparisonTab()
        self.strategy_tab = OptionStrategyTab()
        self.tabs.addTab(self.today_tab, "今日訊號 (Today)")
        self.tabs.addTab(self.backtest_tab, "歷史回測 (Backtest)")
        self.tabs.addTab(self.iv_tab, "IV vs σ (Options)")
        self.tabs.addTab(self.strategy_tab, "Strategy Builder")
        layout.addWidget(self.tabs)


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

        self.btn_market = QPushButton("Market Windows")
        self.btn_market.setFont(QFont('Arial', 11))
        self.btn_market.clicked.connect(self.show_market_windows)
        top_bar.addWidget(self.btn_market)

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
        self.market_page = MarketWindowsPage()
        self.stack.addWidget(self.dashboard_page)   # 0
        self.stack.addWidget(self.stock_page)       # 1
        self.stack.addWidget(self.market_page)      # 2
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

    def show_market_windows(self):
        self.stack.setCurrentIndex(2)
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
