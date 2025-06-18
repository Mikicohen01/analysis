import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import skew, kurtosis
import streamlit as st

# --- פונקציות ניתוח ---

def get_price_data(ticker: str, start_date: datetime.date, end_date: datetime.date, interval: str = "1d") -> pd.DataFrame:
    """Fetch daily closing prices from yfinance."""
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in period.")
    df = df[['Close']].dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df

def price_history_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, days_back: int):
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

    filename = f"{ticker.replace('.', '_')}_daily.csv"
    df_filtered.to_csv(filename, index=False)

    st.write(f"### Price History for {ticker}")
    st.write(f"**Last Price:** {live_price}")
    st.write(f"**Highest Price in last {days_back} days:** {highest_price} on {highest_date}")
    st.write(f"**Lowest Price in last {days_back} days:** {lowest_price} on {lowest_date}")
    st.write(f"Saved {len(df_filtered)} rows to: `{filename}`")
    st.dataframe(df_filtered.tail(20))

def returns_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, interval_days: int, percentile: float):
    ta35 = yf.Ticker(ticker)
    daily_prices = ta35.history(start=start_date, end=end_date, interval="1d")
    if daily_prices.empty:
        st.error(f"No data for {ticker}.")
        return

    daily_prices.reset_index(inplace=True)
    daily_prices['Date'] = pd.to_datetime(daily_prices['Date']).dt.normalize()
    ta35_data = daily_prices[['Date', 'Close']].copy()
    ta35_data.columns = ['Date', 'TA35 Close']
    ta35_data['TA35 Close'] = ta35_data['TA35 Close'].round(1)

    returns_df = ta35_data.copy()
    returns_df['Returns (%)'] = returns_df['TA35 Close'].pct_change(periods=interval_days) * 100
    returns_df = returns_df.dropna().reset_index(drop=True)
    returns_df['Returns (%)'] = returns_df['Returns (%)'].round(2)

    returns_df['Year'] = returns_df['Date'].dt.year
    grouped = returns_df.groupby('Year')
    stats_list = []
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
    stats = pd.DataFrame(stats_list)

    confidence_levels = [90, 95, 99]
    confidence_data = {
        'Confidence Level (%)': [],
        'Lowest Return (%)': [],
        'Highest Return (%)': []
    }
    for cl in confidence_levels:
        lower = (100 - cl) / 2
        upper = 100 - lower
        lower_return = round(returns_df['Returns (%)'].quantile(lower / 100), 2)
        upper_return = round(returns_df['Returns (%)'].quantile(upper / 100), 2)
        confidence_data['Confidence Level (%)'].append(cl)
        confidence_data['Lowest Return (%)'].append(lower_return)
        confidence_data['Highest Return (%)'].append(upper_return)
    confidence_df = pd.DataFrame(confidence_data)

    highest_percentile_year = stats.loc[stats['Percentile'].idxmax(), 'Year']
    lowest_percentile_year = stats.loc[stats['Percentile'].idxmin(), 'Year']
    outlier_years = [highest_percentile_year, lowest_percentile_year]

    all_data_percentile = round(returns_df['Returns (%)'].quantile(percentile/100), 2)
    all_data_skew = round(skew(returns_df['Returns (%)']), 2)
    all_data_kurtosis = round(kurtosis(returns_df['Returns (%)']), 2)

    filtered_returns = returns_df[~returns_df['Year'].isin(outlier_years)].copy()
    filtered_percentile = round(filtered_returns['Returns (%)'].quantile(percentile/100), 2) if not filtered_returns.empty else None
    filtered_skew = round(skew(filtered_returns['Returns (%)']), 2) if not filtered_returns.empty else None
    filtered_kurtosis = round(kurtosis(filtered_returns['Returns (%)']), 2) if not filtered_returns.empty else None

    summary_data = {
        'Metric': [f'{percentile}th Percentile (%)', 'Skewness', 'Kurtosis'],
        'All Data': [all_data_percentile, all_data_skew, all_data_kurtosis],
        'Excluding Outlier Years': [filtered_percentile, filtered_skew, filtered_kurtosis]
    }
    summary_df = pd.DataFrame(summary_data)

    returns_df['Date'] = returns_df['Date'].dt.strftime('%Y-%m-%d')

    st.write(f"### {ticker} Returns ({interval_days}-Day Interval) Sample:")
    st.dataframe(returns_df[['Date', 'TA35 Close', 'Returns (%)']].head())

    st.write(f"### Yearly Statistics (Skew, Kurtosis, {percentile}th Percentile, Extremes):")
    st.dataframe(stats)

    st.write(f"### Summary Table (Outliers Excluded: {', '.join(map(str, outlier_years))}):")
    st.dataframe(summary_df)

    st.write("### Returns Confidence Intervals:")
    st.dataframe(confidence_df)

    returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}day.csv", index=False)
    stats.to_csv(f"{ticker.lower()}_yearly_stats_{interval_days}day.csv", index=False)
    summary_df.to_csv(f"{ticker.lower()}_summary_{interval_days}day.csv", index=False)
    confidence_df.to_csv(f"{ticker.lower()}_confidence_intervals_{interval_days}day.csv", index=False)

    st.write("Files saved:")
    st.write(f"- {ticker.lower()}_returns_{interval_days}day.csv")
    st.write(f"- {ticker.lower()}_yearly_stats_{interval_days}day.csv")
    st.write(f"- {ticker.lower()}_summary_{interval_days}day.csv")
    st.write(f"- {ticker.lower()}_confidence_intervals_{interval_days}day.csv")

def price_forecast_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, interval_days: int, percentile: float, manual_price: float):
    ticker_data = yf.Ticker(ticker)
    hist = ticker_data.history(start=start_date, end=end_date, interval="1d")
    if hist.empty:
        st.error(f"No data for {ticker}.")
        return

    live_price = round(ticker_data.history(period="1d")["Close"].iloc[-1], 2)
    hist.reset_index(inplace=True)
    hist['Date'] = pd.to_datetime(hist['Date']).dt.normalize()
    df = hist[['Date', 'Close']].copy()
    df['Close'] = df['Close'].round(2)

    returns_df = df.copy()
    returns_df['Return (%)'] = returns_df['Close'].pct_change(periods=interval_days) * 100
    returns_df = returns_df.dropna().reset_index(drop=True)
    returns_df['Return (%)'] = returns_df['Return (%)'].round(2)

    returns_df['Year'] = returns_df['Date'].dt.year
    grouped = returns_df.groupby('Year')
    stats = []
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
    filtered_df = returns_df[~returns_df['Date'].dt.year.isin([best_year, worst_year])]

    all_data_pct = round(returns_df['Return (%)'].quantile(percentile / 100), 2)
    filtered_pct = round(filtered_df['Return (%)'].quantile(percentile / 100), 2) if not filtered_df.empty else None

    confidence_levels = [90, 95, 99]
    ci_results = []
    for cl in confidence_levels:
        low = (100 - cl) / 2
        high = 100 - low
        low_return = round(returns_df['Return (%)'].quantile(low / 100), 2)
        high_return = round(returns_df['Return (%)'].quantile(high / 100), 2)
        ci_results.append((cl, low_return, high_return))

    now = datetime.now().strftime('%H:%M on %d/%m/%Y')
    st.write(f"### Forecast generated at: {now} GMT")
    st.write(f"**Ticker:** {ticker}")
    st.write(f"**Live Price (Last Close):** {live_price}")
    st.write(f"**Manual Price Used:** {manual_price}")
    st.write(f"**Data Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.write(f"**Interval:** {interval_days}-day returns")
    st.write(f"**Percentile:** {percentile}th")

    forecast_all = round(manual_price * (1 + all_data_pct / 100), 2)
    st.write(f"**{percentile}th Percentile (All Years):** {forecast_all} (Change: {all_data_pct}%)")

    if filtered_pct is not None:
        forecast_filtered = round(manual_price * (1 + filtered_pct / 100), 2)
        st.write(f"**{percentile}th Percentile (Excl. Outliers):** {forecast_filtered} (Change: {filtered_pct}%)")
    else:
        st.write("Could not calculate filtered forecast (insufficient data).")

    st.write("### Confidence Interval Forecasts (based on Manual Price):")
    for cl, low_r, high_r in ci_results:
        low_price = round(manual_price * (1 + low_r / 100), 2)
        high_price = round(manual_price * (1 + high_r / 100), 2)
        st.write(f"- {cl}% CI: {low_price} to {high_price} ({low_r}% to {high_r}%)")

    returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}d.csv", index=False)
    yearly_stats.to_csv(f"{ticker.lower()}_stats_{interval_days}d.csv", index=False)

    st.write("Files saved:")
    st.write(f"- {ticker.lower()}_returns_{interval_days}d.csv")
    st.write(f"- {ticker.lower()}_stats_{interval_days}d.csv")

def streak_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, yearly: bool, max_length: str):
    df = get_price_data(ticker, start_date, end_date)
    df['Change'] = df['Close'].diff()
    df['Direction'] = df['Change'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else None)
    df = df[['Date', 'Direction']]

    def calculate_streaks(df):
        streaks = []
        current_dir = None
        current_streak = 0
        for direction in df['Direction']:
            if direction not in ['up', 'down']:
                if current_dir and current_streak >= 1:
                    streaks.append((current_dir, current_streak))
                current_dir = None
                current_streak = 0
                continue
            if direction == current_dir:
                current_streak += 1
            else:
                if current_dir and current_streak >= 1:
                    streaks.append((current_dir, current_streak))
                current_dir = direction
                current_streak = 1
        if current_dir and current_streak >= 1:
            streaks.append((current_dir, current_streak))
        return streaks

    def build_table(streaks, max_len):
        up_counts = Counter(l for d, l in streaks if d == 'up')
        down_counts = Counter(l for d, l in streaks if d == 'down')
        if max_len == 'all':
            max_len = max(max(up_counts.keys(), default=1), max(down_counts.keys(), default=1))
        else:
            max_len = int(max_len)
        data = {
            'Streak Length (Days)': [],
            'Up Streaks': [],
            'Down Streaks': []
        }
        for i in range(1, max_len + 1):
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
        max_len = max_length if max_length == 'all' else int(max_length)
        for year, group in df.groupby('Year'):
            streaks = calculate_streaks(group)
            table = build_table(streaks, max_len)
            if not table.empty:
                table['Year'] = year
                tables.append(table)
        if not tables:
            st.error("No streaks found.")
            return
        result = pd.concat(tables).set_index(['Year', 'Streak Length (Days)']).reset_index()
        stats = result[['Year', 'Streak Length (Days)', 'Up Streaks', 'Down Streaks']]
    else:
        streaks = calculate_streaks(df)
        stats = build_table(streaks, max_length)

    if stats.empty:
        st.warning("No streaks found for the selected parameters.")
    else:
        st.write("### Streak Counts:")
        st.dataframe(stats)

    stats.to_csv(f"{ticker.lower()}_streaks{'_yearly' if yearly else ''}.csv", index=False)
    st.write(f"Saved to: `{ticker.lower()}_streaks{'_yearly' if yearly else ''}.csv`")

# --- Streamlit Interface ---

st.title("Stock Data Analysis Tool")

st.sidebar.header("Input Parameters")

ticker = st.sidebar.text_input("Ticker Symbol", value="TA35.TA")

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = col2.date_input("End Date", value=datetime.now())

option = st.sidebar.selectbox("Select Analysis Type", options=[
    "Price History Analysis",
    "Returns Analysis",
    "Price Forecast Analysis",
    "Streak Analysis"
])

if option == "Price History Analysis":
    days_back = st.sidebar.number_input("Days Back (for High/Low)", min_value=1, max_value=365, value=30)
    if st.sidebar.button("Run Analysis"):
        price_history_analysis(ticker, start_date, end_date, days_back)

elif option == "Returns Analysis":
    interval_days = st.sidebar.number_input("Returns Interval (days)", min_value=1, max_value=365, value=5)
    percentile = st.sidebar.slider("Percentile", 0.0, 100.0, 5.0)
    if st.sidebar.button("Run Analysis"):
        returns_analysis
