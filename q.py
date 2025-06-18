# q.ipynb as a script-compatible file (can also be saved as .ipynb)

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import skew, kurtosis
import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------- Helper Function -----------
def get_price_data(ticker: str, start_date: datetime.date, end_date: datetime.date, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} in period.")
    df = df[['Close']].dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    return df

# ----------- Price History -----------
def price_history_analysis(ticker, start_date, end_date, days_back):
    ta = yf.Ticker(ticker)
    df = ta.history(start=start_date, end=end_date, interval="1d")
    if df.empty:
        raise ValueError("No data.")

    live_price = round(ta.history(period="1d")['Close'].iloc[-1], 2)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    last_date = df['Date'].max()
    days_ago_date = last_date - timedelta(days=days_back)
    last_days_df = df[df['Date'] >= days_ago_date]

    highest_price = round(last_days_df['High'].max(), 2)
    highest_date = last_days_df.loc[last_days_df['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')
    lowest_price = round(last_days_df['Low'].min(), 2)
    lowest_date = last_days_df.loc[last_days_df['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')

    df_filtered = df[['Date', 'Close']].copy()
    df_filtered['Date'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
    df_filtered['Close'] = df_filtered['Close'].round(2)

    filename = f"{ticker.replace('.', '_')}_daily.csv"
    df_filtered.to_csv(filename, index=False)

    print(f"Last Price: {live_price}\nHigh: {highest_price} on {highest_date}\nLow: {lowest_price} on {lowest_date}")
    print(df_filtered.tail(20))

# ----------- Returns Analysis -----------
def returns_analysis(ticker, start_date, end_date, interval_days, percentile):
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No data")

    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df[['Date', 'Close']].copy()
    df['Returns (%)'] = df['Close'].pct_change(periods=interval_days) * 100
    df = df.dropna().reset_index(drop=True)
    df['Year'] = df['Date'].dt.year

    stats = df.groupby('Year')['Returns (%)'].agg([
        ('Observations', 'count'),
        ('Skew', lambda x: round(skew(x), 2)),
        ('Kurtosis', lambda x: round(kurtosis(x), 2)),
        (f'{percentile}th Percentile', lambda x: round(x.quantile(percentile / 100), 2))
    ])
    print(stats)

# ----------- Widgets Setup -----------
topic_dropdown = widgets.Dropdown(options=['Select Topic', 'Price History', 'Returns Analysis'], description='Topic:')
ticker_widget = widgets.Text(value='TA35.TA', description='Ticker:')
start_widget = widgets.DatePicker(description='Start:', value=datetime(2014, 1, 1))
end_widget = widgets.DatePicker(description='End:', value=datetime.today())
days_back_widget = widgets.BoundedIntText(value=20, min=1, max=365, description='Days Back:')
interval_widget = widgets.IntText(value=1, description='Interval:')
percentile_widget = widgets.FloatText(value=5, description='Percentile:')
run_button = widgets.Button(description='Run', button_style='success')
output = widgets.Output()

# ----------- Dynamic UI -----------
container = widgets.VBox([])

def update_widgets(change):
    with output:
        clear_output()
        if change['new'] == 'Price History':
            container.children = [ticker_widget, start_widget, end_widget, days_back_widget, run_button]
        elif change['new'] == 'Returns Analysis':
            container.children = [ticker_widget, start_widget, end_widget, interval_widget, percentile_widget, run_button]
        else:
            container.children = []

topic_dropdown.observe(update_widgets, names='value')

def run_analysis(b):
    with output:
        clear_output()
        try:
            topic = topic_dropdown.value
            ticker = ticker_widget.value.strip()
            start_date = start_widget.value
            end_date = end_widget.value

            if topic == 'Price History':
                days_back = days_back_widget.value
                price_history_analysis(ticker, start_date, end_date, days_back)
            elif topic == 'Returns Analysis':
                interval_days = interval_widget.value
                percentile = percentile_widget.value
                returns_analysis(ticker, start_date, end_date, interval_days, percentile)
        except Exception as e:
            print(f"Error: {e}")

run_button.on_click(run_analysis)

display(widgets.VBox([topic_dropdown, container, output]))
