"""
data/loaders.py

Purpose: Use Pandas_datareader, gs_quant for market/marco data ingestion
"""
import os
import pandas as pd
import yfinance as yf


def fetch_raw_data(ticker: str, start: str, end: str, directory: str) -> pd.DataFrame:
    """
    Fetch raw market data using yfinance.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
        directory (str): Directory path to save the raw data CSV.

    Returns:
        pd.DataFrame: DataFrame containing the fetched market data.
    """
    data = yf.download(ticker, start=start, end=end)
    
    if data is not None:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"raw_{ticker}.csv")
        data.to_csv(filepath)
        return data 
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the functions with sample data
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2024-01-05"
    directory = "data/raw"

    print("Testing fetch_raw_data_using_pdr...")
    try:
        data = fetch_raw_data(ticker, start, end, directory)
        print(f"Data fetched successfully: {data.shape}")
    except Exception as e:
        print(f"Error in pdr: {e}")

