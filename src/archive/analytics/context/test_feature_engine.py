import pandas as pd

from src.analytics.context.market_context_analyzer import MarketContextAnalyzer


def test_market_context_analyzer():
    # Setup test data
    data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'volume': [1000] * 20,
        'rsi': [50] * 20
    })
    
    # Features to test
    features = [
        'volatility_5d', 
        'volatility_20d', 
        'volatility_ratio',
        'trend_5d',
        'trend_20d',
        'trend_alignment',
        'rsi_current',
        'volume_ratio',
        'price_to_ma20'
    ]
    
    analyzer = MarketContextAnalyzer(context_features=features)
    result = analyzer.analyze(data)
    
    vector = result["market_context_vector"]
    print(f"Computed context vector:\n{vector}")
    
    # Assertions
    assert not vector.isna().any(), "Context vector contains NaN values!"
    assert 'volatility_5d' in vector.index

if __name__ == "__main__":
    test_market_context_analyzer()
