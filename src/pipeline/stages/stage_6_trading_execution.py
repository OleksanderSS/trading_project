"""
Stage 6: Trading Execution - Facade for Modular Trading Stage.
Maintains backward compatibility with the original TradingExecutionStage.
"""

from .trading.orchestrator import TradingExecutionStage as ModularTradingExecutionStage


class TradingExecutionStage(ModularTradingExecutionStage):
    """
    Facade for TradingExecutionStage.
    Delegates to the modular components in the 'trading' subdirectory.
    """
    pass
