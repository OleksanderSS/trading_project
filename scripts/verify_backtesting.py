
import numpy as np
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.main.modes.backtest import BacktestMode

logger = ProjectLogger.get_logger("BacktestValidation")

def test_backtest_simulation():
    logger.info("Starting Backtest Mode validation...")
    
    # 1. Initialize Mode
    try:
        config_manager = UnifiedConfigManager()
        mode = BacktestMode(config_manager)
        logger.info("✅ BacktestMode initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize BacktestMode: {e}")
        return

    # 2. Prepare Mock Data for Simulation
    # Creating simple aligned series
    dates = pd.date_range(start="2026-01-01", periods=100)
    prices = pd.Series(np.linspace(100, 110, 100), index=dates, name='close')
    signals = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates, name='signal')

    # 3. Test Simulation
    try:
        logger.info("Testing _run_portfolio_simulation...")
        results = mode._run_portfolio_simulation(prices, signals)
        logger.info("✅ Simulation executed successfully.")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"❌ Backtest simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest_simulation()
