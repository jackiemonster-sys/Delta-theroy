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
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-02-01"))

if st.sidebar.button("開始分析"):
    with st.spinner("讀取資料中..."):
        # 下載資料
        df = yf.download(stock_id, start=start_date, end=end_date)

        if df.empty:
            st.error("查無資料，請檢查股票代碼或日期範圍。")
        else:
            # 清理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 算 Delta 指標
            df['Delta_Price'] = df['Close'].diff()
            df['Vol_Change'] = df['Volume'].diff()
            df['Volume_Delta_Ratio'] = df['Delta_Price'] / (df['Volume'] / 1000)

            df['Delta_5D'] = df['Close'] - df['Close'].shift(5)
            df['Upper_Threshold'] = df['Delta_5D'].rolling(60).mean() + (2 * df['Delta_5D'].rolling(60).std())
            df['Lower_Threshold'] = df['Delta_5D'].rolling(60).mean() - (2 * df['Delta_5D'].rolling(60).std())

            df['Signal'] = 0
            df.loc[df['Delta_5D'] > df['Upper_Threshold'], 'Signal'] = -1
            df.loc[df['Delta_5D'] < df['Lower_Threshold'], 'Signal'] = 1

            # 2. 繪製圖表 (針對黑屏與深色模式優化)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # 設定白色背景避免手機深色模式黑屏
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

            # 子圖 1：股價與轉折訊號
            ax1.plot(df.index, df['Close'], label=f'{stock_id} Close', color='blue', alpha=0.7)
            buy_signals = df[df['Signal'] == 1]
            sell_signals = df[df['Signal'] == -1]
            ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='red', s=80, label='Delta Bottom Alert (Buy)')
            ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='green', s=80, label='Delta Top Alert (Sell)')
            ax1.set_ylabel('Price (TWD)', color='black')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # 子圖 2：5日 Delta 變化
            ax2.plot(df.index, df['Delta_5D'], label='5-Day Delta', color='purple')
            ax2.plot(df.index, df['Upper_Threshold'], label='Upper Limit', color='red', linestyle='--')
            ax2.plot(df.index, df['Lower_Threshold'], label='Lower Limit', color='green', linestyle='--')
            ax2.axhline(0, color='gray', linestyle=':')
            ax2.set_ylabel('5-Day Price Change', color='black')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.5)

            # 3. 使用 st.pyplot 輸出至網頁 (關鍵步驟)
            st.pyplot(fig)

            # 4. 顯示數據表格
            st.subheader("最新 10 日分析數據")
            st.dataframe(df.tail(10)[['Close', 'Delta_Price', 'Delta_5D', 'Volume_Delta_Ratio', 'Signal']])
