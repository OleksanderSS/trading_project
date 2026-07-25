import pandas as pd
import numpy as np
from src.archive.features.utils.simple_adaptive_technical_indicators import SimpleAdaptiveTechnicalIndicators

def test_indicators_causality():
    # Create dummy price data
    prices = pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 104, 105] * 10)
    
    indicator = SimpleAdaptiveTechnicalIndicators()
    
    # Check RSI
    rsi = indicator.adaptive_rsi(prices)
    assert not rsi.isna().any(), "RSI contains NaNs"
    
    # Check MACD
    macd, signal, hist = indicator.adaptive_macd(prices)
    assert not macd.isna().any(), "MACD contains NaNs"
    
    # Check Bollinger Bands
    upper, mean, lower = indicator.adaptive_bollinger_bands(prices)
    assert not upper.isna().any(), "Bollinger Bands contain NaNs"
    
    print("All indicators passed causality and NaN tests.")

if __name__ == "__main__":
    test_indicators_causality()
