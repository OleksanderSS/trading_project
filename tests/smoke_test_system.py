import pandas as pd

from src.analytics.arena.arena_battle import get_trading_arena
from src.analytics.context.market_context_analyzer import MarketContextAnalyzer
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.risk.elite_risk_metrics import EliteRiskMetrics

logger = ProjectLogger.get_logger("SystemHealthCheck")

def run_smoke_test():
    report = {"INTEGRATED": [], "ORPHANED": []}
    logger.info("Starting System Health & Integration Check...")

    # 1. Check the real analyzer registration path (analysis.yaml-driven,
    # not the archived static ANALYZER_REGISTRY dict)
    engine = UnifiedAnalyticsEngine(get_current_config())
    logger.info(f"Registry status: {len(engine.analyzers)} components registered.")
    for name in engine.analyzers:
        report["INTEGRATED"].append(f"Analyzer: {name}")

    # 2. Check Market Context Analyzer (Feature Engine)
    try:
        mca = MarketContextAnalyzer(context_features=['volatility_5d', 'trend_5d'])
        df = pd.DataFrame({'close': [100, 101, 102, 103, 104], 'volume': [100]*5})
        mca.analyze(df)
        report["INTEGRATED"].append("MarketContextAnalyzer")
    except Exception:
        report["ORPHANED"].append("MarketContextAnalyzer")

    # 3. Check Risk Metrics
    try:
        risk = EliteRiskMetrics()
        risk.calculate_volatility(pd.Series([0.01, -0.01, 0.02]))
        report["INTEGRATED"].append("EliteRiskMetrics")
    except Exception:
        report["ORPHANED"].append("EliteRiskMetrics")

    # 4. Check Arena
    try:
        arena = get_trading_arena()
        report["INTEGRATED"].append("TradingModelArena (Arena)")
    except Exception:
        report["ORPHANED"].append("TradingModelArena (Arena)")

    # 5. Output Connectivity Report
    logger.info("--- INTEGRATION CONNECTIVITY REPORT ---")
    for category, items in report.items():
        logger.info(f"{category}: {len(items)}")
        for item in items:
            logger.info(f"  - {item}")
            
    logger.info("--- Audit Complete ---")

if __name__ == "__main__":
    run_smoke_test()
