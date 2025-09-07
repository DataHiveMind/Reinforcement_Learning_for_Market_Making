"""
data/preprocessing.py

Purpose: to removing impruites in data and adding techinal indicators for better ML predictions
"""

import pandas as pd


def preprocessing_data(data: pd.DataFrame)->pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        ticker (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data = data.ffill()
    data = data.dropna(axis=1)

    return data

def techinal_indicators(data : pd.DataFrame, long: int, short: int)->pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        long (int): _description_
        short (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Ensure Close is numeric
    data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
    # Returns
    data["Returns"] = data["Close"].pct_change(1, fill_method=None)

    # Simple Moving Averages
    data[f"SMA_{short}"] = data["Close"].rolling(window = short).mean()
    data[f"SMA_{long}"] = data["Close"].rolling(window = long).mean()

    # Exponatial Moving Averages
    data[f"EMA_{short}"] = data["Close"].ewm(span=short, adjust = True).mean()
    data[f"EMA_{long}"] = data["Close"].ewm(span = long, adjust = True).mean()

    # TODO:Add More Indicators


    return data


if __name__ == "__main__":
    data = pd.read_csv("/workspaces/Reinforcement_Learning_for_Market_Making/data/raw/raw_AAPL.csv")
    f_data = preprocessing_data(data)
    fin = techinal_indicators(data, 50, 25)
    fin.to_csv("data/processed/filtered_AAPL_data.csv")
    print(fin.head(5))
    
    
