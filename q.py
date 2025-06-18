import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings

# Widgets
ticker_input = widgets.Text(value='TA35.TA', description='Ticker:')
start_picker = widgets.DatePicker(value=datetime(2014, 1, 1), description='Start:')
end_picker = widgets.DatePicker(value=datetime.today(), description='End:')
days_back_input = widgets.BoundedIntText(value=20, min=1,max=365, description='Days Back:')
fetch_button = widgets.Button(description='Fetch Data')
output = widgets.Output()

# Fetch function
def fetch_data(b):
    with output:
        clear_output()

        ticker = ticker_input.value.strip()
        start = start_picker.value
        end = end_picker.value
        days_back = days_back_input.value

        if not ticker:
            print("Please enter a valid ticker.")
            return
        if start is None or end is None:
            print("Please select valid start and end dates.")
            return
        if start > end:
            print("Start date must be before end date.")
            return
        if days_back < 1:
            print("Please enter a valid number of days (>=1).")
            return

        try:
            # Get historical data
            ta = yf.Ticker(ticker)
            df = ta.history(start=start, end=end, interval="1d")

            if df.empty:
                print(f"No data found for {ticker}.")
                return

            # Get current price
            live_price = ta.history(period="1d")['Close'].iloc[-1]
            live_price = round(live_price, 2)

            # Prepare DataFrame
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])

            # Calculate the date to look back from the last date in df
            last_date = df['Date'].max()
            days_ago_date = last_date - timedelta(days=days_back)

            # Filter last 'days_back' days
            last_days_df = df[df['Date'] >= days_ago_date]

            if last_days_df.empty:
                print(f"No data found in the last {days_back} days range.")
                return

            # Calculate highest and lowest prices in the last 'days_back' days
            highest_price = round(last_days_df['High'].max(), 2)
            highest_date = last_days_df.loc[last_days_df['High'].idxmax(), 'Date'].strftime('%Y-%m-%d')
            lowest_price = round(last_days_df['Low'].min(), 2)
            lowest_date = last_days_df.loc[last_days_df['Low'].idxmin(), 'Date'].strftime('%Y-%m-%d')

            # Round close prices and prepare for saving
            df['Close'] = df['Close'].round(2)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df_filtered = df[['Date', 'Close']]
            df_filtered.columns = ['Date', f'{ticker} Close']

            filename = f"{ticker.replace('.', '_')}_daily.csv"
            df_filtered.to_csv(filename, index=False)

            # Display output
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"📅 Printed on: {now}")
            print(f"📈 Last Price for {ticker}: {live_price}")
            print(f"⬆️ Highest Price in last {days_back} days: {highest_price} on {highest_date}")
            print(f"⬇️ Lowest  Price in last {days_back} days: {lowest_price} on {lowest_date}")
            print(f"✅ Saved {len(df_filtered)} rows to file: {filename}")
            print(df_filtered.tail(20))

        except Exception as e:
            print(f"Error:\n{str(e)}")

# Connect button
fetch_button.on_click(fetch_data)

# Show UI
display(widgets.VBox([
    ticker_input,
    start_picker,
    end_picker,
    days_back_input,
    fetch_button,
    output
]))
