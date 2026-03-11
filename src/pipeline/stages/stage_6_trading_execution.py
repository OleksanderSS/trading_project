"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""

from typing import Dict, Any

from src.pipeline.stages.base_stage import BaseStage
from src.core.logging.logger import ProjectLogger

# Import all the refactored trading components
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.consensus_engine import ConsensusEngine
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.portfolio_manager import PortfolioManager
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator

class TradingExecutionStage(BaseStage):
    """
    A pipeline stage to execute the trading logic.
    """
    def __init__(self, config_manager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain, **kwargs)
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self._initialize_trading_stack()

    def _initialize_trading_stack(self):
        """
        Initializes the full trading stack, wiring all components together.
        """
        self.logger.info("Initializing the complete trading stack...")

        # 1. Initialize the state keeper: Virtual Portfolio
        self.portfolio = VirtualPortfolio(config_manager=self.config_manager)
        self.logger.info(f"Initialized VirtualPortfolio. Cash: {self.portfolio.get_cash():.2f}")

        # 2. Initialize the optional filter
        self.post_inference_filter = PostInferenceFilter()
        self.logger.info("Initialized PostInferenceFilter.")

        # 3. Initialize the decision maker: Consensus Engine
        self.consensus_engine = ConsensusEngine(config_manager=self.config_manager)
        self.logger.info("Initialized ConsensusEngine.")

        # 4. Initialize the risk officer: Portfolio Manager
        self.portfolio_manager = PortfolioManager(
            portfolio=self.portfolio, 
            config_manager=self.config_manager
        )
        self.logger.info("Initialized PortfolioManager.")

        # 5. Initialize the executor: Trader
        self.trader = Trader(portfolio=self.portfolio)
        self.logger.info("Initialized Trader.")

        # 6. Initialize the main conductor for the trading process
        self.trading_orchestrator = TradingOrchestrator(
            consensus_engine=self.consensus_engine,
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio,
            trader=self.trader,
            post_inference_filter=self.post_inference_filter
        )
        self.logger.info("Trading stack initialization complete.")

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        The entry point for the trading execution stage.

        Args:
            **kwargs: The data dictionary from the previous stage.
                                   Expected to contain 'predictions' and 'current_prices'.

        Returns:
            Dict[str, Any]: The data dictionary, potentially updated with trading results.
        """
        self.logger.info("Starting trading execution stage...")

        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')

        if not predictions:
            self.logger.warning("No 'predictions' found in the data. Skipping trading execution.")
            return {}
            
        if not current_prices:
            self.logger.warning("No 'current_prices' found in the data. Skipping trading execution.")
            return {}

        # Let the trading orchestrator handle the entire process
        self.trading_orchestrator.process_signals(
            raw_predictions=predictions,
            current_prices=current_prices
        )

        # The portfolio state is managed internally. We can add results to the data dict if needed.
        return {
            'trading_activity': self.portfolio.get_trade_history()[-5:],
            'portfolio_summary': self.portfolio.get_summary()
        }
