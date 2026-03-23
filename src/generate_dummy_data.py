import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_assets(num_samples=1000, output_path='../data/synthetic_assets.csv'):
    """
    Generates synthetic asset prices where one asset (SPY) leads the others.
    This creates a perfect test environment for the MUSIC algorithm.
    """
    np.random.seed(42)
    
    # Base signal (the latent 'market news')
    base_returns = np.random.normal(0, 0.001, num_samples)
    
    # Define the lag (in samples/minutes). 
    # SPY will react instantly (delay 0). The others will lag by 3 samples.
    lag_samples = 3
    
    spy_returns = base_returns
    other_returns = np.roll(base_returns, lag_samples)
    
    # Add idiosyncratic noise to simulate real market conditions
    noise_level = 0.0005
    spy = spy_returns + np.random.normal(0, noise_level, num_samples)
    qqq = other_returns + np.random.normal(0, noise_level, num_samples)
    aapl = other_returns + np.random.normal(0, noise_level, num_samples)
    msft = other_returns + np.random.normal(0, noise_level, num_samples)
    
    # Convert log returns back to simulated prices
    spy_price = 100 * np.exp(np.cumsum(spy))
    qqq_price = 200 * np.exp(np.cumsum(qqq))
    aapl_price = 150 * np.exp(np.cumsum(aapl))
    msft_price = 250 * np.exp(np.cumsum(msft))
    
    # Generate timestamps (1-minute intervals)
    start_time = datetime(2025, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
    
    # Compile DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'SPY': np.round(spy_price, 2),
        'QQQ': np.round(qqq_price, 2),
        'AAPL': np.round(aapl_price, 2),
        'MSFT': np.round(msft_price, 2)
    })
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_samples} synthetic 1m bars at {output_path}")
    print(f"Engineered Lead-Lag: QQQ, AAPL, and MSFT lag SPY by {lag_samples} minutes.")

if __name__ == "__main__":
    generate_synthetic_assets()
