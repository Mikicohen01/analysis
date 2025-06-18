# @title Default title text
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import skew, kurtosis
import ipywidgets as widgets
from IPython.display import display, clear_output

# Common data fetching
def get_price_data(ticker: str, start_date: datetime.date, end_date: datetime.date, interval: str = "1d") -> pd.DataFrame:
    """Fetch daily closing prices from yfinance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data for {ticker} in period.")
        df = df[['Close']].dropna().reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        return df
    except Exception as e:
        raise ValueError(f"Data fetch failed: {str(e)}")

# Price History Analysis
def price_history_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, days_back: int):
    """Fetch price history, last price, high/low in last N days, save to CSV."""
    try:
        ta = yf.Ticker(ticker)
        df = ta.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            raise ValueError(f"No data found for {ticker}.")

        live_price = round(ta.history(period="1d")['Close'].iloc[-1], 2)
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        last_date = df['Date'].max()
        days_ago_date = last_date - timedelta(days=days_back)
        last_days_df = df[df['Date'] >= days_ago_date]

        if last_days_df.empty:
            raise ValueError(f"No data in last {days_back} days.")

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

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Printed on: {now}")
        print(f"Last Price for {ticker}: {live_price}")
        print(f"Highest Price in last {days_back} days: {highest_price} on {highest_date}")
        print(f"Lowest Price in last {days_back} days: {lowest_price} on {lowest_date}")
        print(f"Saved {len(df_filtered)} rows to: {filename}")
        print("\nLast 20 Days:")
        print(df_filtered.tail(20).to_string(index=False))

    except Exception as e:
        print(f"Error: {str(e)}")

# Returns Analysis
def returns_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, interval_days: int, percentile: float):
    """Calculate returns, yearly stats, confidence intervals, summary table."""
    try:
        ta35 = yf.Ticker(ticker)
        daily_prices = ta35.history(start=start_date, end=end_date, interval="1d")
        if daily_prices.empty:
            raise ValueError(f"No data for {ticker}.")

        daily_prices.reset_index(inplace=True)
        daily_prices['Date'] = pd.to_datetime(daily_prices['Date']).dt.normalize()
        ta35_data = daily_prices[['Date', 'Close']].copy()
        ta35_data.columns = ['Date', 'TA35 Close']
        ta35_data['TA35 Close'] = ta35_data['TA35 Close'].round(1)
        ta35_data['Date'] = pd.to_datetime(ta35_data['Date'])

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

        print(f"\n{ticker} Returns ({interval_days}-Day Interval) Sample:")
        print(returns_df[['Date', 'TA35 Close', 'Returns (%)']].head().to_string(index=False))
        print(f"\nYearly Statistics (Skew, Kurtosis, {percentile}th Percentile, Extremes):")
        print(stats.to_string(index=False))
        print(f"\nSummary Table (Outliers Excluded: {', '.join(map(str, outlier_years))}):")
        print(summary_df.to_string(index=False))
        print("\nReturns Confidence Intervals:")
        print(confidence_df.to_string(index=False))

        returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}day.csv", index=False)
        stats.to_csv(f"{ticker.lower()}_yearly_stats_{interval_days}day.csv", index=False)
        summary_df.to_csv(f"{ticker.lower()}_summary_{interval_days}day.csv", index=False)
        confidence_df.to_csv(f"{ticker.lower()}_confidence_intervals_{interval_days}day.csv", index=False)

        print("\nFiles saved:")
        print(f" - {ticker.lower()}_returns_{interval_days}day.csv")
        print(f" - {ticker.lower()}_yearly_stats_{interval_days}day.csv")
        print(f" - {ticker.lower()}_summary_{interval_days}day.csv")
        print(f" - {ticker.lower()}_confidence_intervals_{interval_days}day.csv")

    except Exception as e:
        print(f"Error: {str(e)}")

# Price Forecast Analysis
def price_forecast_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, interval_days: int, percentile: float, manual_price: float):
    """Forecast price based on returns, manual price, and percentiles."""
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(start=start_date, end=end_date, interval="1d")
        if hist.empty:
            raise ValueError(f"No data for {ticker}.")

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
        print(f"\nForecast generated at: {now} GMT")
        print(f"Ticker: {ticker}")
        print(f"Live Price (Last Close): {live_price}")
        print(f"Manual Price Used: {manual_price}")
        print(f"Data Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Interval: {interval_days}-day returns")
        print(f"Percentile: {percentile}th")

        forecast_all = round(manual_price * (1 + all_data_pct / 100), 2)
        print(f"\n{percentile}th Percentile (All Years): {forecast_all} (Change: {all_data_pct}%)")

        if filtered_pct is not None:
            forecast_filtered = round(manual_price * (1 + filtered_pct / 100), 2)
            print(f"{percentile}th Percentile (Excl. Outliers): {forecast_filtered} (Change: {filtered_pct}%)")
        else:
            print("Could not calculate filtered forecast (insufficient data).")

        print("\nConfidence Interval Forecasts (based on Manual Price):")
        for cl, low_r, high_r in ci_results:
            low_price = round(manual_price * (1 + low_r / 100), 2)
            high_price = round(manual_price * (1 + high_r / 100), 2)
            print(f" - {cl}% CI: {low_price} to {high_price} ({low_r}% to {high_r}%)")

        returns_df.to_csv(f"{ticker.lower()}_returns_{interval_days}d.csv", index=False)
        yearly_stats.to_csv(f"{ticker.lower()}_stats_{interval_days}d.csv", index=False)

        print("\nFiles saved:")
        print(f" - {ticker.lower()}_returns_{interval_days}d.csv")
        print(f" - {ticker.lower()}_stats_{interval_days}d.csv")

    except Exception as e:
        print(f"Error: {str(e)}")

# Streak Analysis
def streak_analysis(ticker: str, start_date: datetime.date, end_date: datetime.date, yearly: bool, max_length: str):
    """Calculate streak counts, excluding zero rows."""
    try:
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
                raise ValueError("No streaks found.")
            result = pd.concat(tables).set_index(['Year', 'Streak Length (Days)']).reset_index()
            stats = result[['Year', 'Streak Length (Days)', 'Up Streaks', 'Down Streaks']]
        else:
            streaks = calculate_streaks(df)
            stats = build_table(streaks, max_length)

        if stats.empty:
            print("\nNo streaks found for the selected parameters.")
        else:
            print("\nStreak Counts:")
            print(stats.to_string(index=False))

        stats.to_csv(f"{ticker.lower()}_streaks{'_yearly' if yearly else ''}.csv", index=False)
        print(f"\nSaved to: {ticker.lower()}_streaks{'_yearly' if yearly else ''}.csv")

    except Exception as e:
        print(f"Error: {str(e)}")

# Widgets
topic_dropdown = widgets.Dropdown(options=['Select Topic', 'Price History', 'Returns Analysis', 'Price Forecast', 'Streak Analysis'], description='Analysis Topic:', value='Select Topic')
ticker_widget = widgets.Text(value='TA35.TA', description='Ticker:')
start_widget = widgets.DatePicker(description='Start Date:', value=datetime(2014, 1, 1))
end_widget = widgets.DatePicker(description='End Date:', value=datetime.today())
days_back_widget = widgets.BoundedIntText(value=20, min=1, max=365, description='Days Back:')
interval_widget = widgets.IntText(value=1, description='Interval (days):')
percentile_widget = widgets.FloatText(value=5, description='Percentile (0-100):')
manual_price_widget = widgets.FloatText(value=2000.0, description='Manual Price:')
dist_widget = widgets.Dropdown(options=['Yearly', 'All Years'], value='Yearly', description='Distribution:')
length_widget = widgets.Text(value='all', description='Max Streak Length:', placeholder='Enter number or "all"')
run_button = widgets.Button(description='Run Analysis', button_style='success')
output = widgets.Output()

# Widget groups
common_widgets = [ticker_widget, widgets.HBox([start_widget, end_widget])]
price_history_widgets = common_widgets + [days_back_widget, run_button]
returns_widgets = common_widgets + [interval_widget, percentile_widget, run_button]
forecast_widgets = common_widgets + [interval_widget, percentile_widget, manual_price_widget, run_button]
streak_widgets = common_widgets + [dist_widget, length_widget, run_button]
dynamic_container = widgets.VBox([])

def update_widgets(change):
    """Update visible widgets based on topic selection."""
    with output:
        clear_output()
        topic = change['new']
        if topic == 'Select Topic':
            dynamic_container.children = []
        elif topic == 'Price History':
            dynamic_container.children = price_history_widgets
        elif topic == 'Returns Analysis':
            dynamic_container.children = returns_widgets
        elif topic == 'Price Forecast':
            dynamic_container.children = forecast_widgets
        elif topic == 'Streak Analysis':
            dynamic_container.children = streak_widgets

topic_dropdown.observe(update_widgets, names='value')

def run_analysis(b):
    """Execute selected analysis."""
    with output:
        clear_output()
        try:
            topic = topic_dropdown.value
            if topic == 'Select Topic':
                print("Please select an analysis topic.")
                return

            ticker = ticker_widget.value.strip().upper()
            start_date = start_widget.value
            end_date = end_widget.value

            if isinstance(start_date, datetime):
                start_date = start_date.date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()

            if start_date is None or end_date is None:
                print("Select valid dates.")
                return
            if start_date > end_date:
                print("Start date must be before end date.")
                return
            if end_date > datetime.today().date():
                print("End date cannot be future.")
                return

            if topic == 'Price History':
                days_back = days_back_widget.value
                if days_back < 1:
                    print("Days Back must be at least 1.")
                    return
                price_history_analysis(ticker, start_date, end_date, days_back)

            elif topic == 'Returns Analysis':
                interval_days = interval_widget.value
                percentile = percentile_widget.value
                if not (0 <= percentile <= 100):
                    print("Percentile must be between 0 and 100.")
                    return
                if interval_days < 1:
                    print("Interval must be at least 1 day.")
                    return
                returns_analysis(ticker, start_date, end_date, interval_days, percentile)

            elif topic == 'Price Forecast':
                interval_days = interval_widget.value
                percentile = percentile_widget.value
                manual_price = manual_price_widget.value
                if not (0 <= percentile <= 100):
                    print("Percentile must be between 0 and 100.")
                    return
                if interval_days < 1:
                    print("Interval must be at least 1 day.")
                    return
                if manual_price <= 0:
                    print("Manual price must be positive.")
                    return
                price_forecast_analysis(ticker, start_date, end_date, interval_days, percentile, manual_price)

            elif topic == 'Streak Analysis':
                yearly = dist_widget.value == 'Yearly'
                max_length = length_widget.value.strip().lower()
                if max_length != 'all' and (not max_length.isdigit() or int(max_length) < 1):
                    print("Max Streak Length must be a positive number or 'all'.")
                    return
                streak_analysis(ticker, start_date, end_date, yearly, max_length)

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Suggestions:")
            print("- Check ticker: https://finance.yahoo.com/")
            print("- Try other tickers.")
            print("- Check internet.")
            print("- Update yfinance: pip install --upgrade yfinance")

run_button.on_click(run_analysis)

# Main UI
main_container = widgets.VBox([topic_dropdown, dynamic_container, output])
display(main_container)

