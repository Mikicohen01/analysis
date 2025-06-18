import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import skew, kurtosis

# --- פונקציות עזר ---

def get_price_data(ticker, start_date, end_date, interval="1d"):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if df.empty:
        st.error(f"No data found for {ticker} in the selected date range.")
        return None
    df = df[['Close']].dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df

def price_history_analysis(ticker, start_date, end_date, days_back):
    ta = yf.Ticker(ticker)
    df = ta.history(start=start_date, end=end_date, interval="1d")
    if df.empty:
        st.error(f"No data found for {ticker}.")
        return
    live_price = round(ta.history(period="1d")['Close'].iloc[-1], 2)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
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
    df['Close'] = df['Close'].round(2)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df_filtered = df[['Date', 'Close']]
    df_filtered.columns = ['Date', f'{ticker} Close']

    st.write(f"### Price History for {ticker}")
    st.write(f"**Last Price:** {live_price}")
    st.write(f"**Highest Price in last {days_back} days:** {highest_price} on {highest_date}")
    st.write(f"**Lowest Price in last {days_back} days:** {lowest_price} on {lowest_date}")
    st.dataframe(df_filtered.tail(20))
    csv = df_filtered.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, f"{ticker}_daily.csv", "text/csv")

def returns_analysis(ticker, start_date, end_date, interval_days, percentile):
    ta = yf.Ticker(ticker)
    daily_prices = ta.history(start=start_date, end=end_date, interval="1d")
    if daily_prices.empty:
        st.error(f"No data found for {ticker}.")
        return
    daily_prices.reset_index(inplace=True)
    daily_prices['Date'] = pd.to_datetime(daily_prices['Date']).dt.normalize()
    data = daily_prices[['Date', 'Close']].copy()
    data.columns = ['Date', 'Close']
    data['Close'] = data['Close'].round(1)

    data['Returns (%)'] = data['Close'].pct_change(periods=interval_days) * 100
    data.dropna(inplace=True)
    data['Returns (%)'] = data['Returns (%)'].round(2)
    data['Year'] = data['Date'].dt.year

    stats_list = []
    grouped = data.groupby('Year')
    for year, group in grouped:
        returns = group['Returns (%)']
        max_return = returns.max()
        min_return = returns.min()
        max_date = group.loc[returns.idxmax(), 'Date'].strftime('%d/%m')
        min_date = group.loc[returns.idxmin(), 'Date'].strftime('%d/%m')
        stats_list.append({
            'Year': year,
            'Observations': len(returns),
            'Skew': round(skew(returns), 2),
            'Kurtosis': round(kurtosis(returns), 2),
            'Percentile': round(returns.quantile(percentile / 100), 2),
            'Best Day': max_date,
            'Best Return (%)': round(max_return, 2),
            'Worst Day': min_date,
            'Worst Return (%)': round(min_return, 2)
        })

    stats_df = pd.DataFrame(stats_list)

    st.write(f"### Returns Analysis for {ticker} ({interval_days}-day Interval)")
    st.write(data[['Date', 'Close', 'Returns (%)']].head())
    st.write("#### Yearly Statistics")
    st.dataframe(stats_df)

    # Confidence intervals
    confidence_levels = [90, 95, 99]
    confidence_data = []
    for cl in confidence_levels:
        lower = (100 - cl) / 2
        upper = 100 - lower
        low_ret = round(data['Returns (%)'].quantile(lower / 100), 2)
        high_ret = round(data['Returns (%)'].quantile(upper / 100), 2)
        confidence_data.append({"Confidence Level (%)": cl, "Lowest Return (%)": low_ret, "Highest Return (%)": high_ret})
    confidence_df = pd.DataFrame(confidence_data)
    st.write("#### Confidence Intervals for Returns")
    st.dataframe(confidence_df)

    st.download_button("Download Returns CSV", data.to_csv(index=False).encode(), f"{ticker.lower()}_returns.csv", "text/csv")
    st.download_button("Download Yearly Stats CSV", stats_df.to_csv(index=False).encode(), f"{ticker.lower()}_yearly_stats.csv", "text/csv")

def price_forecast_analysis(ticker, start_date, end_date, interval_days, percentile, manual_price):
    ta = yf.Ticker(ticker)
    hist = ta.history(start=start_date, end=end_date, interval="1d")
    if hist.empty:
        st.error(f"No data found for {ticker}.")
        return
    live_price = round(ta.history(period="1d")["Close"].iloc[-1], 2)
    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
    df = hist[['Date', 'Close']].copy()
    df['Close'] = df['Close'].round(2)

    df['Return (%)'] = df['Close'].pct_change(periods=interval_days) * 100
    df.dropna(inplace=True)
    df['Return (%)'] = df['Return (%)'].round(2)
    df['Year'] = df['Date'].dt.year

    stats = []
    grouped = df.groupby('Year')
    for year, group in grouped:
        returns = group['Return (%)']
        stats.append({
            'Year': year,
            'Observations': len(returns),
            'Skew': round(skew(returns), 2),
            'Kurtosis': round(kurtosis(returns), 2),
            f'{percentile}th Percentile': round(returns.quantile(percentile / 100), 2),
        })
    yearly_stats = pd.DataFrame(stats)

    best_year = yearly_stats.loc[yearly_stats[f'{percentile}th Percentile'].idxmax(), 'Year']
    worst_year = yearly_stats.loc[yearly_stats[f'{percentile}th Percentile'].idxmin(), 'Year']
    filtered_df = df[~df['Date'].dt.year.isin([best_year, worst_year])]

    all_pct = round(df['Return (%)'].quantile(percentile / 100), 2)
    filtered_pct = round(filtered_df['Return (%)'].quantile(percentile / 100), 2) if not filtered_df.empty else None

    st.write(f"### Price Forecast for {ticker}")
    st.write(f"Live Price: {live_price}")
    st.write(f"Manual Price: {manual_price}")
    st.write(f"Interval: {interval_days} days, Percentile: {percentile}%")

    forecast_all = round(manual_price * (1 + all_pct / 100), 2)
    st.write(f"{percentile}th Percentile Forecast (All Years): {forecast_all} ({all_pct}%)")
    if filtered_pct is not None:
        forecast_filtered = round(manual_price * (1 + filtered_pct / 100), 2)
        st.write(f"{percentile}th Percentile Forecast (Excl. Outliers): {forecast_filtered} ({filtered_pct}%)")

    # Confidence intervals
    confidence_levels = [90, 95, 99]
    ci_results = []
    for cl in confidence_levels:
        low = (100 - cl) / 2
        high = 100 - low
        low_ret = round(df['Return (%)'].quantile(low / 100), 2)
        high_ret = round(df['Return (%)'].quantile(high / 100), 2)
        low_price = round(manual_price * (1 + low_ret / 100), 2)
        high_price = round(manual_price * (1 + high_ret / 100), 2)
        ci_results.append((cl, low_price, high_price, low_ret, high_ret))

    st.write("#### Confidence Interval Forecasts")
    for cl, low_p, high_p, low_r, high_r in ci_results:
        st.write(f"{cl}% CI: Price between {low_p} ({low_r}%) and {high_p} ({high_r}%)")

    st.download_button("Download Returns CSV", df.to_csv(index=False).encode(), f"{ticker.lower()}_returns.csv", "text/csv")
    st.download_button("Download Yearly Stats CSV", yearly_stats.to_csv(index=False).encode(), f"{ticker.lower()}_stats.csv", "text/csv")

def streak_analysis(ticker, start_date, end_date, yearly, max_length):
    df = get_price_data(ticker, start_date, end_date)
    if df is None:
        return
    df['Change'] = df['Close'].diff()
    df['Direction'] = df['Change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else None)
    df = df[['Date', 'Direction']]

    def calculate_streaks(df_streak):
        streaks = []
        current_dir = None
        current_streak = 0
        for direction in df_streak['Direction']:
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
        return streaks

    def build_table(streaks, max_len):
        up_counts = Counter(l for d, l in streaks if d == 'up')
        down_counts = Counter(l for d, l in streaks if d == 'down')
        if max_len == 'all':
            max_len_val = max(max(up_counts.keys(), default=1), max(down_counts.keys(), default=1))
        else:
            max_len_val = int(max_len)
        data = {'Streak Length (Days)': [], 'Up Streaks': [], 'Down Streaks': []}
        for i in range(1, max_len_val + 1):
            up = up_counts.get(i, 0)
            down = down_counts.get(i, 0)
            if up > 0 or down > 0:
                data['Streak Length (Days)'].append(i)
                data['Up Streaks'].append(up)
                data['Down Streaks'].append(down)
        return pd.DataFrame(data) if data['Streak Length (Days)'] else pd.DataFrame(columns=['Streak Length (Days)', 'Up Streaks', 'Down Streaks'])

    df['Date'] = pd.to_datetime(df['Date'])
    if yearly:
        df['Year'] = df['Date'].dt.year
        tables = []
        for year, group in df.groupby('Year'):
            streaks = calculate_streaks(group)
            table = build_table(streaks, max_length)
            if not table.empty:
                table['Year'] = year
                tables.append(table)
        if not tables:
            st.warning("No streaks found.")
            return
        result = pd.concat(tables).set_index(['Year', 'Streak Length (Days)']).reset_index()
        st.write(f"### Streak Analysis Yearly for {ticker}")
        st.dataframe(result)
        csv = result.to_csv(index=False).encode()
        st.download_button("Download Streaks Yearly CSV", csv, f"{ticker}_streaks_yearly.csv", "text/csv")
    else:
        streaks = calculate_streaks(df)
        table = build_table(streaks, max_length)
        if table.empty:
            st.warning("No streaks found.")
            return
        st.write(f"### Streak Analysis Overall for {ticker}")
        st.dataframe(table)
        csv = table.to_csv(index=False).encode()
        st.download_button("Download Streaks CSV", csv, f"{ticker}_streaks.csv", "text/csv")

# --- ממשק משתמש ---

def main():
    st.title("Stock Data Analysis Tool")

    topic = st.selectbox("Select Analysis Topic:", 
                         ["Select Topic", "Price History", "Returns Analysis", "Price Forecast", "Streak Analysis"])

    ticker = st.text_input("Ticker (Yahoo Finance format):", value="TA35.TA")
    start_date = st.date_input("Start Date:", value=datetime(2014, 1, 1))
    end_date = st.date_input("End Date:", value=datetime.today())

    if start_date > end_date:
        st.error("Start Date must be before End Date.")
        return

    if topic == "Price History":
        days_back = st.number_input("Days Back:", min_value=1, max_value=365, value=20)
        if st.button("Run Price History Analysis"):
            price_history_analysis(ticker, start_date, end_date, days_back)

    elif topic == "Returns Analysis":
        interval_days = st.number_input("Interval Days:", min_value=1, value=1)
        percentile = st.number_input("Percentile (0-100):", min_value=0.0, max_value=100.0, value=5.0)
        if st.button("Run Returns Analysis"):
            returns_analysis(ticker, start_date, end_date, interval_days, percentile)

    elif topic == "Price Forecast":
        interval_days = st.number_input("Interval Days:", min_value=1, value=1)
        percentile = st.number_input("Percentile (0-100):", min_value=0.0, max_value=100.0, value=5.0)
        manual_price = st.number_input("Manual Price:", min_value=0.01, value=2000.0)
        if st.button("Run Price Forecast Analysis"):
            price_forecast_analysis(ticker, start_date, end_date, interval_days, percentile, manual_price)

    elif topic == "Streak Analysis":
        yearly = st.selectbox("Distribution:", ["Yearly", "All Years"]) == "Yearly"
        max_length = st.text_input("Max Streak Length (number or 'all'):", value="all")
        if st.button("Run Streak Analysis"):
            if max_length.lower() != "all" and not max_length.isdigit():
                st.error("Max Streak Length must be a positive integer or 'all'.")
            else:
                streak_analysis(ticker, start_date, end_date, yearly, max_length)

if __name__ == "__main__":
    main()
