"""
Acts as the Risk Officer for the trading system.

This module is responsible for risk management, position sizing, and generating
trade orders based on signals from the Consensus Engine. It does not manage
portfolio state directly but queries a VirtualPortfolio instance.
"""

from typing import List, Dict, Optional
from src.core.logging.logger import ProjectLogger
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.trader import TradeOrder # Re-using the dataclass
# Assuming an optimizer exists and follows a similar pattern
# from src.optimization.portfolio_optimizer import PortfolioOptimizer

logger = ProjectLogger.get_logger("PortfolioManager")

class PortfolioManager:
    """
    Manages portfolio-level risk, position sizing, and order generation.
    """

    def __init__(self, 
                 virtual_portfolio: VirtualPortfolio, 
                 # optimizer: PortfolioOptimizer,
                 config: Optional[Dict] = None):
        """
        Args:
            virtual_portfolio: The stateful portfolio object.
            optimizer: The portfolio optimization engine.
            config: Risk management configuration.
        """
        self.portfolio = virtual_portfolio
        # self.optimizer = optimizer
        self.logger = logger
        
        # Load risk parameters from config or use defaults
        risk_config = config if config is not None else {}
        self.risk_per_trade_pct = risk_config.get('risk_per_trade_pct', 0.01) # 1% of total equity per trade
        self.max_position_size_pct = risk_config.get('max_position_size_pct', 0.10) # Max 10% of equity in one asset
        self.max_daily_drawdown_pct = risk_config.get('max_daily_drawdown_pct', 0.05) # 5% max daily loss
        
        self.kill_switch_active = False

    def is_trading_allowed(self, current_prices: Dict[str, float]) -> bool:
        """
        Primary gatekeeper. Checks if any risk rule prevents trading.
        """
        if self.kill_switch_active:
            self.logger.critical("Trading blocked: KILL SWITCH IS ACTIVE.")
            return False

        # Check for daily drawdown
        # This requires the portfolio to track daily starting equity
        # For now, we assume a method exists on the portfolio
        if hasattr(self.portfolio, 'get_daily_drawdown') and self.portfolio.get_daily_drawdown(current_prices) < -self.max_daily_drawdown_pct:
            self.logger.critical(f"Trading blocked: Max daily drawdown of {self.max_daily_drawdown_pct:.2%} exceeded.")
            self.kill_switch_active = True
            return False
        
        return True

    def generate_orders_from_signals(self, 
                                       signals: List[Dict[str, any]], 
                                       current_prices: Dict[str, float]) -> List[TradeOrder]:
        """
        Processes signals from the ConsensusEngine and generates executable TradeOrders.
        """
        if not self.is_trading_allowed(current_prices):
            return []

        orders = []
        for signal in signals:
            action = signal.get('final_signal')
            ticker = signal.get('ticker')
            confidence = signal.get('confidence', 0.5)
            price = current_prices.get(ticker)

            if not all([action, ticker, price]):
                self.logger.warning(f"Skipping invalid signal: {signal}")
                continue

            if action == 'BUY':
                shares_to_trade = self._calculate_position_size(ticker, price, confidence)
                if shares_to_trade > 0:
                    orders.append(TradeOrder(
                        ticker=ticker,
                        quantity=shares_to_trade,
                        price=price,
                        action='BUY',
                        reason=f"Consensus Signal (Conf: {confidence:.2f})"
                    ))
            
            elif action == 'SELL':
                # For now, we assume a SELL signal means closing the entire position.
                # More complex logic could sell a portion.
                position = self.portfolio.positions.get(ticker)
                if position and position['quantity'] > 0:
                    orders.append(TradeOrder(
                        ticker=ticker,
                        quantity=position['quantity'], # Sell all
                        price=price,
                        action='SELL',
                        reason="Consensus Signal (SELL)"
                    ))
        return orders

    def check_risk_exits(self, current_prices: Dict[str, float]) -> List[TradeOrder]:
        """
        Checks open positions for stop-loss or take-profit triggers and generates exit orders.
        """
        exit_orders = []
        if not self.portfolio.positions: 
            return []

        for ticker, position in self.portfolio.positions.items():
            current_price = current_prices.get(ticker)
            if not current_price: 
                continue

            # NOTE: This assumes the position dict contains SL/TP levels.
            # The VirtualPortfolio would need to be extended to hold this.
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')

            if stop_loss and current_price <= stop_loss:
                exit_orders.append(TradeOrder(
                    ticker=ticker, 
                    quantity=position['quantity'], 
                    price=current_price, 
                    action='SELL', 
                    reason="Stop-Loss Triggered"
                ))
            elif take_profit and current_price >= take_profit:
                 exit_orders.append(TradeOrder(
                    ticker=ticker, 
                    quantity=position['quantity'], 
                    price=current_price, 
                    action='SELL', 
                    reason="Take-Profit Triggered"
                ))

        return exit_orders

    def _calculate_position_size(self, ticker: str, price: float, confidence: float) -> int:
        """
        Calculates the number of shares to buy based on risk parameters and portfolio state.
        """
        total_equity = self.portfolio.get_total_value({ticker: price}) # Approximate value
        
        # 1. Determine max capital for this single trade
        capital_at_risk = total_equity * self.risk_per_trade_pct * confidence

        # Simple assumption: risk is the full amount. A better way involves stop-loss.
        # For now, shares are based on capital at risk.
        shares_from_risk = capital_at_risk / price

        # 2. Determine max capital based on position size limit
        max_position_value = total_equity * self.max_position_size_pct
        current_position_value = 0
        if ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[ticker]['quantity'] * price
        
        allowed_capital = max(0, max_position_value - current_position_value)
        shares_from_exposure = allowed_capital / price

        # 3. Determine max capital based on available cash
        available_cash = self.portfolio.current_balance
        shares_from_cash = available_cash / price

        # The final number of shares is the minimum of all constraints
        final_shares = min(shares_from_risk, shares_from_exposure, shares_from_cash)

        if final_shares <= 0:
            return 0
            
        # Return as an integer number of shares
        return int(final_shares)

