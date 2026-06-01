"""
Provides the interface for trade execution.

This module is the final step in the trading pipeline (The "Executioner").
It takes a fully formed trade order and executes it via a broker API
or a simulated paper trading account.
"""

from dataclasses import dataclass
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("Trader")

@dataclass
class TradeOrder:
    """A data structure to hold all details for a trade execution."""
    ticker: str
    quantity: int
    price: float
    action: str  # 'BUY' or 'SELL'
    reason: str = "Signal-driven trade"

class Trader:
    """
    The execution engine. Connects to a broker and places trades.
    For now, it simulates trades for paper trading.
    """

    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.logger = logger
        if self.paper_trading:
            self.logger.info("Trader initialized in PAPER TRADING mode.")
        else:
            self.logger.warning("Trader initialized in LIVE TRADING mode. Real orders will be placed.")
            # In a real scenario, you would initialize the broker connection here.
            # e.g., self.broker = Alpaca(...) 

    def execute_order(self, order: TradeOrder) -> bool:
        """
        Executes a given trade order.

        Args:
            order (TradeOrder): The details of the trade to execute.

        Returns:
            bool: True if the order was executed successfully, False otherwise.
        """
        self.logger.info(f"Received order: {order.action} {order.quantity} {order.ticker} @ {order.price:.2f}. Reason: {order.reason}")

        if self.paper_trading:
            # In paper trading, we just log the action as a success.
            # The PortfolioManager will be responsible for updating the virtual portfolio.
            self.logger.info(f"[PAPER] Executed order for {order.ticker}.")
            return True
        else:
            # In live trading, this is where the broker API call would happen.
            self.logger.critical("[LIVE] FAILED to execute order: Live trading is not implemented.")
            raise NotImplementedError("Live trading is intentionally disabled until a broker adapter is configured. Cannot execute real orders.")
