import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.title("Stock Price History Analysis")

# Input widgets
ticker = st.text_input("Ticker Symbol", value="TA35.TA")
start_date = st.date_input("Start Date", value=datetime(2014, 1, 1))
end_date = st.date_input("End Date", value=datetime.now())
days_back = st.number_input("Days Back", min_value=1, value=20)

if st.button("Run Analysis"):
    try:
        # Fetch data
        ta = yf.Ticker(ticker)
        df = ta.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            st.error(f"No data found for {ticker}.")
            return

        # Calculate metrics
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

        # Display results
        st.write(f"**Printed on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Last Price for {ticker}:** {live_price}")
        st.write(f"**Highest Price in last {days_back} days:** {highest_price} on {highest_date}")
        st.write(f"**Lowest Price in last {days_back} days:** {lowest_price} on {lowest_date}")

        # Save to CSV
        df_filtered = df[['Date', 'Close']].copy()
        df_filtered['Date'] = df_filtered['Date'].dt.strftime('%Y-%m-%d')
        df_filtered.columns = ['Date', f'{ticker} Close']
        filename = f"{ticker.replace('.', '_')}_daily.csv"
        df_filtered.to_csv(filename, index=False)
        st.write(f"**Saved {len(df_filtered)} rows to:** {filename}")
        st.write("**Last 20 Days:**")
        st.dataframe(df_filtered.tail(20))

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Suggestions: Check ticker, internet, or update yfinance with `pip install --upgrade yfinance`.")

