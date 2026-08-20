import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 設定 Streamlit 頁面標題與配置
st.set_page_config(page_title="台股 Delta 理論分析", layout="wide")

st.title("📈 台股個股 Delta 週期與動能分析")

# 1. 側邊欄輸入參數
st.sidebar.header("參數設定")
stock_id = st.sidebar.text_input("股票代碼 (台股請加 .TW 或 .TWO)", value="2330.TW")
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-08-01"))

if st.sidebar.button("開始分析"):
    with st.spinner("讀取數據中..."):
        # 下載歷史資料
        df = yf.download(stock_id, start=start_date, end=end_date)

        if df.empty:
            st.error("查無資料，請確認股票代碼或日期範圍。")
        else:
            # 解決 yfinance MultiIndex 欄位問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 確保欄位為一維 Series，避免 ValueError 欄位對齊錯誤
            close_s = df['Close'].squeeze()
            volume_s = df['Volume'].squeeze()

            # 2. 計算 Delta 相關指標
            df['Delta_Price'] = close_s.diff()
            df['Vol_Change'] = volume_s.diff()
            df['Volume_Delta_Ratio'] = df['Delta_Price'] / (volume_s / 1000)

            # 5日 Delta 與極限通道
            df['Delta_5D'] = close_s - close_s.shift(5)
            rolling_mean = df['Delta_5D'].rolling(60).mean()
            rolling_std = df['Delta_5D'].rolling(60).std()

            df['Upper_Threshold'] = rolling_mean + (2 * rolling_std)
            df['Lower_Threshold'] = rolling_mean - (2 * rolling_std)

            # 訊號判定
            df['Signal'] = 0
            df.loc[df['Delta_5D'] > df['Upper_Threshold'], 'Signal'] = -1
            df.loc[df['Delta_5D'] < df['Lower_Threshold'], 'Signal'] = 1

            # 3. 繪製圖表 (針對手機深色模式優化背景)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

            # 上圖：價格與轉折預警
            ax1.plot(df.index, close_s, label=f'{stock_id} Close', color='blue', alpha=0.7)
            buy_signals = df[df['Signal'] == 1]
            sell_signals = df[df['Signal'] == -1]
            
            ax1.scatter(buy_signals.index, close_s.loc[buy_signals.index], marker='^', color='red', s=80, label='Delta Bottom Alert (Buy)')
            ax1.scatter(sell_signals.index, close_s.loc[sell_signals.index], marker='v', color='green', s=80, label='Delta Top Alert (Sell)')
            ax1.set_ylabel('Price (TWD)', color='black')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # 下圖：5日 Delta 變化與軌道
            ax2.plot(df.index, df['Delta_5D'], label='5-Day Delta', color='purple')
            ax2.plot(df.index, df['Upper_Threshold'], label='Upper Limit', color='red', linestyle='--')
            ax2.plot(df.index, df['Lower_Threshold'], label='Lower Limit', color='green', linestyle='--')
            ax2.axhline(0, color='gray', linestyle=':')
            ax2.set_ylabel('5-Day Price Change', color='black')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.5)

            # 顯示圖表
            st.pyplot(fig)

            # 4. 顯示最後 10 筆數據
            st.subheader("最新 10 日分析數據")
            show_cols = ['Close', 'Delta_Price', 'Delta_5D', 'Volume_Delta_Ratio', 'Signal']
            st.dataframe(df.tail(10)[show_cols])
