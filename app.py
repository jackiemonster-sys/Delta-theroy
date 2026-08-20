import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 設定 Streamlit 頁面標題與配置
st.set_page_config(page_title="台股 Delta 理論分析與預測", layout="wide")

st.title("📈 台股 Delta 週期動能分析與明日高低點預測")
st.markdown("""
本 App 結合 **Delta 動能理論**（統計波動轉折）與**明日預測系統**。
> **注意：** 預測結果基於統計波動率，僅供參考，不構成投資建議。
""")

# 1. 側邊欄輸入參數
st.sidebar.header("參數設定")
stock_id = st.sidebar.text_input("股票代碼 (台股請加 .TW 或 .TWO)", value="2330.TW")
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-08-01"))

# 預測相關參數
prediction_days = 1 # 預測未來幾天
conf_interval = st.sidebar.slider("預測信賴區間 (%)", min_value=80, max_value=99, value=95)

if st.sidebar.button("開始分析與預測"):
    with st.spinner("讀取與計算數據中..."):
        # 下載歷史資料
        df = yf.download(stock_id, start=start_date, end=end_date)

        if df.empty or len(df) < 60:
            st.error("資料不足，無法分析。請確保日期範圍至少包含 60 個交易日，並檢查代碼。")
        else:
            # 解決 yfinance MultiIndex 欄位問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 確保欄位為一維 Series
            close_s = df['Close'].squeeze()
            volume_s = df['Volume'].squeeze()

            # --- [Part 1: 原有 Delta 轉折分析] ---
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

            # --- [Part 2: 明日高低點預測系統] ---
            # 基礎：利用今日收盤價、滾動平均收益率及滾動波動率
            
            # 計算單日收益率
            df['Return'] = close_s.pct_change()
            
            # 計算滾動參數 (近 20 日作為短期波動依據)
            window = 20
            df['Rolling_Mean_Ret'] = df['Return'].rolling(window).mean()
            df['Rolling_Std_Ret'] = df['Return'].rolling(window).std()
            
            # 獲取最新一天的數據用於預測明天
            last_price = close_s.iloc[-1]
            last_date = df.index[-1]
            last_std_ret = df['Rolling_Std_Ret'].iloc[-1]
            
            if np.isnan(last_std_ret):
                st.warning("波動率計算不足，預測功能可能不準確。")
                last_std_ret = 0.02 # 預設一個基本波動率 (2%)

            # 根據信賴區間計算 Z 分數 (95% -> 1.96)
            import scipy.stats as stats
            z_score = stats.norm.ppf(1 - (1 - conf_interval/100)/2)

            # 計算預測的價格波動幅度
            price_volatility = last_price * last_std_ret * z_score * np.sqrt(prediction_days)
            
            predict_high = last_price + price_volatility
            predict_low = last_price - price_volatility
            predict_date = last_date + pd.Timedelta(days=1)

            # 將預測結果加入 DataFrame 用於繪圖
            df['Predict_High'] = close_s + (close_s * df['Rolling_Std_Ret'] * z_score * np.sqrt(prediction_days))
            df['Predict_Low'] = close_s - (close_s * df['Rolling_Std_Ret'] * z_score * np.sqrt(prediction_days))

            # --- [Part 3: 顯示預測面版] ---
            st.subheader(f"🔮 明日預測範圍 ({conf_interval}% 信賴度)")
            cols = st.columns(3)
            cols[0].metric("明日預測日期", predict_date.strftime('%Y-%m-%d'))
            cols[1].metric("預測高點 ⬆️", f"{predict_high:.2f}", f"+{price_volatility:.2f}")
            cols[2].metric("預測低點 ⬇️", f"{predict_low:.2f}", f"-{price_volatility:.2f}")
            
            st.markdown(f"**分析解釋：** 根據近 {window} 日波動模式，**{stock_id}** 在 {predict_date.strftime('%Y-%m-%d')} 有 {conf_interval}% 的機率收盤在 **{predict_low:.2f}** 至 **{predict_high:.2f}** 之間。")

            # --- [Part 4: 繪製升級版圖表] ---
            # 優化手機顯示：增加陰影帶和時間軸格式
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            fig.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')

            # 下載預測陰影帶所需的未來日期線
            extended_index = df.index

            # 上圖：價格、轉折預警與預測陰影帶
            ax1.plot(df.index, close_s, label=f'{stock_id} Close', color='black', alpha=0.6, linewidth=1.5)
            
            # 【關鍵新功能】繪製預測高低點陰影帶
            ax1.fill_between(df.index, df['Predict_Low'], df['Predict_High'], color='#3498db', alpha=0.2, label=f'{conf_interval}% Predict Band')

            buy_signals = df[df['Signal'] == 1]
            sell_signals = df[df['Signal'] == -1]
            
            ax1.scatter(buy_signals.index, close_s.loc[buy_signals.index], marker='^', color='red', s=100, label='Delta Bottom (Buy)')
            ax1.scatter(sell_signals.index, close_s.loc[sell_signals.index], marker='v', color='green', s=100, label='Delta Top (Sell)')
            
            ax1.set_title(f'Price & Delta Cycle Prediction: {stock_id}')
            ax1.set_ylabel('Price (TWD)', color='black')
            ax1.legend(loc='upper left', fontsize='small')
            ax1.grid(True, linestyle='--', alpha=0.3)

            # 下圖：5日 Delta 變化與軌道
            ax2.plot(df.index, df['Delta_5D'], label='5-Day Delta', color='purple', linewidth=1.2)
            ax2.plot(df.index, df['Upper_Threshold'], label='Upper Limit', color='red', linestyle='--', alpha=0.6)
            ax2.plot(df.index, df['Lower_Threshold'], label='Lower Limit', color='green', linestyle='--', alpha=0.6)
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.8)
            
            ax2.set_ylabel('5-Day Delta', color='black')
            ax2.legend(loc='upper left', fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.3)

            # 時間軸格式優化
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            plt.xticks(rotation=45)

            # 顯示圖表
            st.pyplot(fig)

            # 5. 顯示最後 10 筆數據 (加入預測)
            st.subheader("最新 10 日分析與預測數據")
            show_cols = ['Close', 'Delta_Price', 'Signal', 'Predict_Low', 'Predict_High']
            st.dataframe(df.tail(10)[show_cols].style.format("{:.2f}"))
