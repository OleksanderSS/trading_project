"""
Acts as an intelligent bridge between trading signals and the portfolio manager.
Integrates DEAN principles and consensus-based execution.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine
from src.models.dean.dean_bootstrap_system import DeanSystem
from src.models.model_selector.selector import ModelSelector
from src.scripts.optimization.portfolio.optimizer import PortfolioOptimizer

from .portfolio_manager import PortfolioManager

# Initialize logger for the module
logger = ProjectLogger.get_logger("SmartTrader")

class SmartTrader:
    """
    An advanced trader that validates signals using DEAN's Critic 
    and model consensus before execution.
    """

    def __init__(self,
                 portfolio_manager: PortfolioManager,
                 model_selector: ModelSelector,
                 dean_system: DeanSystem,
                 data_manager: DataManager,
                 optimizer: PortfolioOptimizer,
                 diary: ExperienceDiaryEngine,
                 paper_trading: bool = True):
        """
        Args:
            portfolio_manager: Manages positions and account balance.
            model_selector: Logic for Smart Selection and Consensus.
            dean_system: Interface for DEAN (Actor/Critic/Simulator).
            data_manager: Access to unified market data.
            optimizer: Advanced portfolio sizing and optimization.
            diary: Meta-learning diary for recording decisions.
            paper_trading: If True, trades are not sent to exchange.
        """
        self.portfolio_manager = portfolio_manager
        self.model_selector = model_selector
        self.dean_system = dean_system
        self.data_manager = data_manager
        self.optimizer = optimizer
        self.diary = diary
        self.paper_trading = paper_trading
        self.logger = logger
        self.global_kill_switch = False

    def execute_signal(self,
                       ticker: str,
                       price: float,
                       consensus_data: dict[str, Any],
                       context_fingerprint: str):
        """
        Validates and executes a trading signal based on Heavy/Light consensus and DEAN critique.
        """
        if self.global_kill_switch:
            self.logger.critical(f"TRADE BLOCKED: Global kill switch is ACTIVE for {ticker}.")
            return

        signal_type = consensus_data.get('signal') # 'BUY', 'SELL', or 'HOLD'
        consensus_status = consensus_data.get('status', 'NEUTRAL')
        confidence = consensus_data.get('confidence', 0.5)
        reasoning = f"Consensus: {consensus_status}, Fingerprint: {context_fingerprint}"

        if signal_type not in ['BUY', 'SELL']:
            return

        # 1. Market Context Check via DataManager
        latest_data = self.data_manager.get_latest_market_context(ticker, n_rows=100)

        # Interaction with DEAN for Final Approval
        # Assuming dean_system has a method to critique based on latest data
        try:
            _, critique = self.dean_system.bootstrap_action_critique({
                'ticker': ticker,
                'price': price,
                'signal': signal_type,
                'confidence': confidence,
                'latest_data': latest_data,
                'fingerprint': context_fingerprint
            })

            if critique.critique_score < 0:
                block_reason = f"[DEAN CRITIC] Blocked {signal_type}. Score: {critique.critique_score}"
                self.logger.warning(f"{block_reason} for {ticker}")
                self._log_to_diary(ticker, signal_type, reasoning, context_fingerprint, confidence, "BLOCKED", reason=block_reason)
                return
        except Exception as e:
            self.logger.error(f"Error during DEAN critique for {ticker}: {e}")
            # Fallback: if critic fails, we might want to be conservative
            return

        # 2. Dynamic Sizing via Kelly Criterion/Optimizer
        allocation_multiplier = 1.0
        if consensus_status == 'DIVERGENCE_WARNING':
            allocation_multiplier = 0.25
        elif consensus_status == 'WEAK_CONFIRMATION':
            allocation_multiplier = 0.5
        elif consensus_status != 'STRONGLY_CONFIRMED' and consensus_status != 'NORMAL':
            # Be cautious with unknown statuses
            allocation_multiplier = 0.0

        if allocation_multiplier <= 0:
            self.logger.info(f"Signal for {ticker} skipped due to status: {consensus_status}")
            return

        # 3. Dynamic Position Sizing including Slippage & Fees
        shares_to_trade = self._calculate_shares(ticker, price, confidence, allocation_multiplier)

        if shares_to_trade <= 0:
            self.logger.warning(f"Calculated 0 shares for {ticker}. Check balance/risk settings.")
            return

        # 4. Final Execution
        try:
            if not self.paper_trading:
                # Actual execution would go here
                self.portfolio_manager.buy_stock(ticker, shares_to_trade, price) if signal_type == 'BUY' else \
                self.portfolio_manager.sell_stock(ticker, shares_to_trade, price)

            self.logger.info(f"SUCCESS ({'PAPER' if self.paper_trading else 'LIVE'}): {signal_type} {shares_to_trade} {ticker} at {price}")

            self._log_to_diary(ticker, signal_type, reasoning, context_fingerprint, confidence, "EXECUTED", price=price)

        except Exception as e:
            self.logger.error(f"Execution Error for {ticker}: {e}", exc_info=True)

    def update_active_positions(self, current_prices: dict[str, float]):
        """
        Monitors active positions for exits based on stops or reverse signals.
        """
        # This assumes VirtualPortfolio logic or similar in portfolio_manager
        # Checking for SL/TP
        if hasattr(self.portfolio_manager, 'check_stop_loss_take_profit'):
            exit_signals = self.portfolio_manager.check_stop_loss_take_profit(current_prices)
            for signal in exit_signals:
                ticker = signal['ticker']
                self.logger.info(f"EXIT SIGNAL: {signal['type']} for {ticker} at {signal['price']}. Reason: {signal['reason']}")
                # Close entire position
                pos = self.portfolio_manager.positions.get(ticker, {})
                if pos:
                    self.portfolio_manager.sell_stock(ticker, pos['quantity'], signal['price'], reason=signal['reason'])

    def _calculate_shares(self, ticker: str, price: float, confidence: float, multiplier: float) -> int:
        """
        Calculates number of shares using risk parameters and accounting for costs.
        """
        try:
            balance = self.portfolio_manager.get_current_balance()

            # Use optimizer to get suggested fraction (e.g. via Kelly or Volatility parity)
            # Falling back to a safe default if specific method is not found
            suggested_fraction = 0.02 # 2% default risk
            if hasattr(self.optimizer, 'get_optimal_allocation'):
                suggested_fraction = self.optimizer.get_optimal_allocation(ticker)

            risk_per_trade = balance * suggested_fraction * confidence * multiplier

            # Account for estimated costs (fees + spread)
            estimated_costs_pct = 0.0015
            effective_price = price * (1 + estimated_costs_pct)

            if effective_price <= 0:
                return 0

            return int(risk_per_trade // effective_price)
        except Exception as e:
            self.logger.error(f"Error calculating shares for {ticker}: {e}")
            return 0

    def _log_to_diary(self, ticker, signal, reasoning, context_fingerprint, conf, status, price=None, reason=None):
        """Records the decision in the experience diary for meta-learning."""
        try:
            self.diary.record_decision_metadata({
                'timestamp': pd.Timestamp.now().isoformat(),
                'ticker': ticker,
                'signal': signal,
                'reasoning': f"{reasoning} | {reason if reason else ''}",
                'fingerprint': context_fingerprint,
                'confidence': conf,
                'status': status,
                'entry_price': price
            })
        except Exception as e:
            self.logger.error(f"Failed to record decision in diary: {e}")

    def set_kill_switch(self, state: bool):
        self.global_kill_switch = state
        self.logger.warning(f"GLOBAL KILL SWITCH set to: {state}")
