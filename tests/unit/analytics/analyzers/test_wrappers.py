import pandas as pd
from src.analytics.analyzers.wrappers import DrawdownAnalyzer, FamaFrenchAnalyzer, VolatilityAnalyzer

def test_drawdown_analyzer():
    analyzer = DrawdownAnalyzer()
    df = pd.DataFrame({'close': [100, 95, 105], 'high': [100, 100, 105]})
    result = analyzer.analyze(df)
    assert "max_drawdown" in result
    assert "drawdown" in result
    assert "underwater_duration" in result
    assert result["max_drawdown"] <= 0

def test_volatility_analyzer():
    analyzer = VolatilityAnalyzer()
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    result = analyzer.analyze(returns)
    assert "rolling_volatility" in result
    assert "realized_volatility" in result
    assert result["latest_volatility"] >= 0
    assert isinstance(result["rolling_volatility"], pd.Series)

def test_fama_french_analyzer():
    # FamaFrenchFactors requires yfinance and internet connectivity
    # This might fail in CI/restricted environments.
    analyzer = FamaFrenchAnalyzer()
    # Mocking might be needed for a robust test
    # Just checking initialization for now
    assert analyzer is not None
