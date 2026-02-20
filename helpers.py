import pandas as pd
import logging
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


def get_app_features(ticker:str):
    "This function builds the features directly from the API results (YahooFinance)"

    logging.info("Downloading stock market data for %s", ticker)

    yf_data = (
        yf.download(
            ticker, 
            period = "1mo", 
            group_by="ticker")
            .stack(level=0, future_stack=True)
            .reset_index()
            )
    

    yf_data.columns.name = None
    yf_data.columns = yf_data.columns.str.lower()

    df = yf_data.drop_duplicates()

    # Always work with values from the day before: when we have market closed values
    yesterday = datetime.now().date() - timedelta(days=1)
    df = df[df['date'].dt.date <= yesterday]


    # set up features
    df['today'] = df.groupby('ticker')['close'].pct_change(1) * 100


    for i in range(1,6):
        df[f'lag_{i}'] = df.groupby('ticker')['today'].shift(i)


    df['volume_billions'] = df['volume'] / 1_000_000_000

    df['ma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=10).mean())
    df['dist_ma10'] = (df['close'] - df['ma_10']) / df['ma_10'].shift(1)

    df['target'] = (df['today'] > 0).astype(int)

    df = df.dropna()

    features = ['volume_billions', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'dist_ma10']

    #get only latest value
    latest = df.dropna().iloc[-1:][features]
    input_values = latest.values.astype(np.float32)

    #bring last close value in the predict output
    latest_row = df.dropna(subset=features).iloc[-1:]

    last_close = float(latest_row['close'].values[0])


    logging.info("Features builder concluded")

    return input_values, last_close
