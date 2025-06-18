import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from collections import Counter

st.title("Stock Analysis with Streamlit")

# Input widgets
ticker = st.text_input("Ticker Symbol", value="TA35.TA")
start_date = st.date_input("Start Date", value=datetime(2014, 1, 1))
end_date = st.date_input("End Date", value=datetime.now())
analysis_type = st.selectbox("Analysis Type", ["Price History", "Returns Analysis", "Price Forecast", "Streak Analysis"])

if st.button("Run Analysis"):
    try:
        if start_date > end_date or end_date > datetime.now().date():
            st.error("Start date must be before end date, and end date cannot be in the future.")
            return

        ta = yf.Ticker(ticker)

        if analysis_type == "Price History":
            days_back = st.number_input("Days Back", min_value=1, value=20, key="days_back")
            df = ta.history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                st.error(f"No data found for {ticker}.")
                return
            live_price = round(ta.history(period="1d")['Close'].iloc[-1], 2)
            df.reset_index(inplace=True)
            last_date = df['Date'].max()
            days_ago_date = last_date - timedelta(days=days_back)
            last_days_df = df[df['Date'] >= days_ago_date]
            if last_days_df.empty:
                st.error(f"No data in last {days_back} days.")
                return
            highest_price = round(last_days_df['High'].max(), 2)
            highest_date = last_days_df.loc[last_days_df['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')
            lowest_price = round(last_days_df['Low'].min(), 2)
            lowest_date = last_days_df.loc[last_days_df['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')
            df_filtered = df[['Date', 'Close']].copy()
            df_filtered['Date'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
            df_filtered.columns = ['Date', f'{ticker} Close']
            filename = f"{ticker.replace('.', '_')}_daily.csv"
            df_filtered.to_csv(filename, index=False)
            st.write(f"Printed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"Last Price for {ticker}: {live_price}")
            st.write(f"Highest Price in last {days_back} days: {highest_price} on {highest_date}")
            st.write(f"Lowest Price in last {days_back} days: {lowest_price} on {lowest_date}")
            st.write(f"Saved {len(df_filtered)} rows to: {filename}")
            st.write("Last 20 Days:")
            st.dataframe(df_filtered.tail(20))

        elif analysis_type == "Returns Analysis":
            interval_days = st.number_input("Interval (days)", min_value=1, value=1, key="interval")
            percentile = st.number_input("Percentile (0-100)", min_value=0.0, max_value=100.0, value=5.0, key="percentile")
            df = ta.history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                st.error(f"No data found for {ticker}.")
                return
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            returns_df = df[['Date', 'Close']].copy()
            returns_df.columns = ['Date', 'TA35 Close']
            returns_df['Returns (%)'] = returns_df['TA35 Close'].pct_change(periods=interval_days) * 100
            returns_df = returns_df.dropna().reset_index(drop=True)
            returns_df['Returns (%)'] = returns_df['Returns (%)'].round(2)
            returns_df['Year'] = returns_df['Date'].dt.year
            grouped = returns_df.groupby('Year')
            stats_list = []
            for year, group in grouped:
                returns = group['Returns (%)']
                stats_list.append({
                    'Year': year,
                    'Observations': len(returns),
                    'Skew': round(skew(returns), 2),
                    'Kurtosis': round(kurtosis(returns), 2),
                    'Percentile': round(returns.quantile(percentile / 100), 2)
                })
            stats = pd.DataFrame(stats_list)
            st.write("Yearly Statistics:")
            st.dataframe(stats)
            returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}day.csv", index=False)
            st.write(f"Saved to: {ticker.lower()}_returns_{interval_days}day.csv")

        elif analysis_type == "Price Forecast":
            interval_days = st.number_input("Interval (days)", min_value=1, value=1, key="interval2")
            percentile = st.number_input("Percentile (0-100)", min_value=0.0, max_value=100.0, value=5.0, key="percentile2")
            manual_price = st.number_input("Manual Price", min_value=0.01, value=2000.0, key="manual")
            df = ta.history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                st.error(f"No data found for {ticker}.")
                return
            df.reset_index(inplace=True)
            returns_df = df[['Date', 'Close']].copy()
            returns_df['Return (%)'] = returns_df['Close'].pct_change(periods=interval_days) * 100
            returns_df = returns_df.dropna().reset_index(drop=True)
            returns_df['Return (%)'] = returns_df['Return (%)'].round(2)
            forecast = manual_price * (1 + returns_df['Return (%)'].quantile(percentile/100)/100)
            st.write(f"Forecast Price: {round(forecast, 2)}")
            returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}d.csv", index=False)
            st.write(f"Saved to: {ticker.lower()}_returns_{interval_days}d.csv")

        elif analysis_type == "Streak Analysis":
            max_length = st.text_input("Max Streak Length ('all' or number)", value="all", key="length")
            df = ta.history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                st.error(f"No data found for {ticker}.")
                return
            df['Change'] = df['Close'].diff()
            df['Direction'] = df['Change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else None)
            streaks = []
            current_streak = 0
            current_dir = None
            for direction in df['Direction']:
                if direction not in ['up', 'down']:
                    if current_dir and current_streak > 0:
                        streaks.append((current_dir, current_streak))
                    current_dir = None
                    current_streak = 0
                    continue
                if direction == current_dir:
                    current_streak += 1
                else:
                    if current_dir and current_streak > 0:
                        streaks.append((current_dir, current_streak))
                    current_dir = direction
                    current_streak = 1
            if current_dir and current_streak > 0:
                streaks.append((current_dir, current_streak))
            st.write("Streaks:", streaks)
            streak_df = pd.DataFrame(streaks, columns=['Direction', 'Length'])
            streak_df.to_csv(f"{ticker.lower()}_streaks.csv", index=False)
            st.write(f"Saved to: {ticker.lower()}_streaks.csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Suggestions: Check ticker, internet connection, or update yfinance with `pip install --upgrade yfinance`.")
