import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_asset_data(symbols, start_date='2023-01-01', end_date=None, interval='1m'):
    """
    Fetch OHLC data from Yahoo Finance.
    Returns DataFrame with timestamps and adjusted close prices.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    data = yf.download(symbols, start=start_date, end=end_date, interval=interval, group_by='ticker')
    
    # Extract adjusted close for each symbol
    if len(symbols) == 1:
        prices = data['Adj Close'].to_frame(name=symbols[0])
    else:
        prices = data['Adj Close']
    
    prices = prices.dropna()
    return prices

def load_from_csv(filepath):
    """
    Load pre‑saved CSV with timestamp and asset columns.
    Assumes first column is datetime, rest are asset prices.
    """
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    return df

def compute_returns(prices):
    """
    Compute log returns.
    """
    returns = np.log(prices).diff().dropna()
    return returns
