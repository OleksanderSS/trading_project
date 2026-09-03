import pandas as pd

from src.analytics.arena.arena_battle import get_trading_arena
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.risk.elite_risk_metrics import EliteRiskMetrics

logger = ProjectLogger.get_logger("SystemHealthCheck")

def run_smoke_test():
    # These were called INTEGRATED and ORPHANED. Only the first section
    # measures integration -- a component registered by UnifiedAnalyticsEngine
    # is one the pipeline will call. The sections below construct a class and
    # call a method: that shows the code imports and runs, which is not the
    # same claim.
    #
    # MarketContextAnalyzer is why the distinction matters. It sat in the list
    # below and was reported INTEGRATED on every run, while its only consumer
    # set `self.analyzer = ...` and never read the attribute again. It was
    # archived on 2026-08-13, and a check that says "integrated" for a class
    # nothing calls is worse than no check.
    report = {"REGISTERED": [], "IMPORTABLE": [], "BROKEN": []}
    logger.info("Starting System Health & Integration Check...")

    # 1. The real registration path (analysis.yaml-driven, not the archived
    # static ANALYZER_REGISTRY dict). This one does mean integrated.
    engine = UnifiedAnalyticsEngine(get_current_config())
    logger.info(f"Registry status: {len(engine.analyzers)} components registered.")
    for name in engine.analyzers:
        report["REGISTERED"].append(f"Analyzer: {name}")

    # 2. Check Risk Metrics
    try:
        risk = EliteRiskMetrics()
        risk.calculate_volatility(pd.Series([0.01, -0.01, 0.02]))
        report["IMPORTABLE"].append("EliteRiskMetrics")
    except Exception:
        report["BROKEN"].append("EliteRiskMetrics")

    # 3. Check Arena
    try:
        arena = get_trading_arena()
        report["IMPORTABLE"].append("TradingModelArena (Arena)")
    except Exception:
        report["BROKEN"].append("TradingModelArena (Arena)")

    # 4. Output Connectivity Report
    logger.info("--- REGISTRATION AND IMPORT REPORT ---")
    for category, items in report.items():
        logger.info(f"{category}: {len(items)}")
        for item in items:
            logger.info(f"  - {item}")
            
    logger.info("--- Audit Complete ---")

if __name__ == "__main__":
    run_smoke_test()
