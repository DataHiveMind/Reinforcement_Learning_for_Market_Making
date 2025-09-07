"""
calibration/vol_models.py

Purpose: Fits statistical Volatility models (e.g., GARCH. ARIMA) to find historical midprice returns
"""
import pandas as pd 
import numpy as np

import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

class GARCH_Model():
    def __init__(self) -> None:
        pass

class ARIMA_Model():
    def __init__(self) -> None:
        pass

class Volatility():
    def __init__(self) -> None:
        prices : pd.DataFrame

    def calculate_volatility(self):
        """Calculates historical volatility."""
        log_returns = np.diff(np.log(self.prices))
        volatility = np.std(log_returns)
        return volatility

    def calculate_rolling_volatility(self, window=30):
        """Calculates rolling volatility over a given window."""
        log_returns = np.diff(np.log(self.prices))
        rolling_vol = [np.std(log_returns[i:i+window]) for i in range(len(log_returns) - window + 1)]
        return rolling_vol

    def analyze_volatility(self):
        """Analyzes price volatility."""
        vol = calculate_volatility(self.prices)
        print(f"Historical Volatility: {vol}")
        rolling_vol = calculate_rolling_volatility(self.prices)
        print(f"Rolling Volatility (last window): {rolling_vol[-1] if rolling_vol else 'N/A'}")

if __name__ == "__main__":
    pass