import pandas as pd

from src.analytics.analyzer_registry import get_analyzer
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("AnalyzerRegistryTest")

def test_analyzers():
    logger.info("Starting test of Analyzer Registry integration...")
    
    # 1. Test HedgeFundAnalyzer (Integrate the previously 'unused' analyze_manager_skill)
    try:
        analyzer = get_analyzer("hedge_fund")
        # Mock data
        returns = pd.Series([0.01, -0.005, 0.02, 0.01, 0.01])
        performance = {'sharpe_ratio': 1.5}
        factors = {'exposures': {'const': 0.05}, 'p_values': {'const': 0.01}}
        
        result = analyzer.analyze_manager_skill(returns, performance, factors)
        logger.info(f"HedgeFundAnalyzer (analyze_manager_skill) test result: {result}")
    except Exception as e:
        logger.error(f"HedgeFundAnalyzer test failed: {e}")

    # 2. Test VolatilityAnalyzer
    try:
        analyzer = get_analyzer("volatility")
        prices = pd.DataFrame({'AAPL': [150, 152, 151, 153]})
        result = analyzer.analyze({'prices': prices}, window=2)
        logger.info(f"VolatilityAnalyzer test result: {result}")
    except Exception as e:
        logger.error(f"VolatilityAnalyzer test failed: {e}")

    # 3. Test CausalAnalyzer
    try:
        analyzer = get_analyzer("causal_wrapper")
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2026-01-01', '2026-01-02']),
            'event_type': ['monetary_tightening', 'monetary_expansion'],
            'magnitude': [1.0, 1.0]
        })
        # Use a method that actually exists in CausalEngine
        engine = analyzer.engine
        result = engine.generate_causal_vectors(data, pd.date_range('2026-01-01', periods=5))
        logger.info("CausalAnalyzer (CausalEngine) test executed successfully.")
    except Exception as e:
        logger.error(f"CausalAnalyzer test failed: {e}")

if __name__ == "__main__":
    test_analyzers()
