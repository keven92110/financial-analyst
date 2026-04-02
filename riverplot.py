# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:33:05 2025

@author: keven
"""

import tkinter as tk
from tkinter import ttk
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import date
import time

today = date.today()
today_str = today.strftime("%Y-%m-%d")

TICKERS = ['MSFT','AAPL','NVDA','AMD','GOOG','META','TSM','TSLA','PLTR','APP','MCD','COST']

# Auto-fetch EPS data (cached, refreshes every 12 hours)
# 加 force_refresh=True 可強制重抓最新資料
from eps_fetcher import load_all_eps
FR = load_all_eps(TICKERS)


# 設定 PE 區間
pe_levels = np.arange(55, 5, -5)  # 10, 15, 20, ..., 50


for ticker_symbol in TICKERS:
    # ticker_symbol = 'AAPL'
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="5y")
    closing_prices = hist['Close']

    # 動態決定 PE 河流帶的結束日期：EstimateEPSnext4Q 最後一筆有效資料
    eps_data = FR[ticker_symbol]
    last_valid = eps_data['EstimateEPSnext4Q'].dropna().index.max()
    full_range = pd.date_range(
        start='2022-01-01', end=last_valid.tz_localize(None)
    ).tz_localize('America/New_York')

    # 將 EPS 資料 reindex 到每日，線性插值讓河流帶平滑
    eps_daily = eps_data[['EstimateEPSnext4Q', 'EPSpast4Q']].reindex(full_range).interpolate(method='time')
    eps_daily = eps_daily.bfill()

    # 合併股價（只到今天）與 EPS（延伸到未來）
    temp_df = eps_daily.copy()
    temp_df['Close'] = closing_prices.reindex(full_range)

    # 只保留有 EPS 資料的部分
    temp_df = temp_df.dropna(subset=['EstimateEPSnext4Q'])

    fig, ax = plt.subplots(figsize=(12, 6))

    # PE 河流帶（延伸到 2027）
    for i in range(len(pe_levels) - 1):
        lower_pe, upper_pe = pe_levels[i], pe_levels[i + 1]
        lower_prices = temp_df['EstimateEPSnext4Q'] * lower_pe
        upper_prices = temp_df['EstimateEPSnext4Q'] * upper_pe
        # legend 顯示「今天」的價格區間
        today_idx = temp_df.index.get_indexer([pd.Timestamp(today_str, tz='America/New_York')], method='nearest')[0]
        lower_price = round(lower_prices.iloc[today_idx], 1)
        upper_price = round(upper_prices.iloc[today_idx], 1)
        color = plt.cm.RdYlGn_r((len(pe_levels) - i) / len(pe_levels))
        ax.fill_between(temp_df.index, lower_prices, upper_prices, color=color, alpha=0.5,
                        label=f'PE:{upper_pe}-{lower_pe}, price:{upper_price}-{lower_price}')

    # 股價線（只到今天）
    price_df = temp_df['Close'].dropna()
    ax.plot(price_df.index, price_df, color='black', linewidth=1.5, label="price")

    # 今天的分界線
    ax.axvline(x=pd.Timestamp(today_str, tz='America/New_York'),
               color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("price")
    ax.set_title(f"{ticker_symbol} riverplot from forward P/E")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    
    
    
    
    
    