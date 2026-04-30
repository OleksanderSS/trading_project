# src/trading/portfolio_manager.py
"""
Acts as the Risk Officer for the trading system.

This module is responsible for risk management, position sizing, and generating
trade orders based on signals from the Consensus Engine. It does not manage
portfolio state directly but queries a VirtualPortfolio instance.
"""

from typing import List, Dict, Optional
import numpy as np
from src.core.logging.logger import ProjectLogger
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.trader import TradeOrder 
from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer
from src.algorithms.risk_parity_allocator import RiskParityAllocator

logger = ProjectLogger.get_logger("PortfolioManager")

class PortfolioManager:
    """
    Manages portfolio-level risk, position sizing, and order generation.
    """

    def __init__(self, 
                 virtual_portfolio: VirtualPortfolio, 
                 elite_risk_sizer=None, 
                 config: Optional[Dict] = None):
        """
        Args:
            virtual_portfolio: The stateful portfolio object.
            elite_risk_sizer: EliteRiskSizer for optimal position sizing (Kelly + correlation-aware).
            config: Risk management configuration.
        """
        self.portfolio = virtual_portfolio
        self.elite_risk_sizer = elite_risk_sizer 
        self.logger = logger
        
        # Load risk parameters from config or use defaults
        risk_config = config if config is not None else {}
        self.risk_per_trade_pct = risk_config.get('risk_per_trade_pct', 0.03) 
        self.max_position_size_pct = risk_config.get('max_position_size_pct', 0.10) 
        self.max_daily_drawdown_pct = risk_config.get('max_daily_drawdown_pct', 0.05) 
        
        # Initialize advanced algorithms
        self.position_sizer = AdaptivePositionSizer(config=risk_config.get('position_sizer', {}))
        self.risk_allocator = RiskParityAllocator(config=risk_config.get('risk_allocator', {}))
        
        self.kill_switch_active = False

    def is_trading_allowed(self, current_prices: Dict[str, float]) -> bool:
        """
        Primary gatekeeper. Checks if any risk rule prevents trading.
        """
        self.logger.debug(f"[PORTFOLIO] is_trading_allowed called with {len(current_prices)} prices")
        
        if self.kill_switch_active:
            self.logger.critical("Trading blocked: KILL SWITCH IS ACTIVE.")
            return False

        # Check for daily drawdown limit
        if hasattr(self.portfolio, 'get_daily_drawdown') and self.portfolio.get_daily_drawdown(current_prices) < -self.max_daily_drawdown_pct:
            self.logger.critical(f"Trading blocked: Max daily drawdown of {self.max_daily_drawdown_pct:.2%} exceeded.")
            self.kill_switch_active = True
            return False
        
        self.logger.debug("[PORTFOLIO] is_trading_allowed: TRUE")
        return True

    def generate_orders_from_signals(self, 
                                       signals: List[Dict[str, any]], 
                                       current_prices: Dict[str, float]) -> List[TradeOrder]:
        """
        Processes signals from the ConsensusEngine and generates executable TradeOrders.
        """
        self.logger.info(f"[PORTFOLIO] generate_orders_from_signals called with {len(signals)} signals")
        
        if not self.is_trading_allowed(current_prices):
            self.logger.warning("[PORTFOLIO] Trading is NOT allowed by risk protocol!")
            return []

        self.logger.info("[PORTFOLIO] Trading protocol cleared, processing signals...")
        
        orders = []
        for signal in signals:
            action = signal.get('final_signal')
            ticker = signal.get('ticker')
            confidence = signal.get('confidence', 0.5)
            price = current_prices.get(ticker)

            self.logger.debug(f"[PORTFOLIO] Analyzing signal: ticker={ticker}, action={action}, confidence={confidence}, price={price}")

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
                # Simplified SELL: close entire existing position
                position = self.portfolio.positions.get(ticker)
                if position and position['quantity'] > 0:
                    orders.append(TradeOrder(
                        ticker=ticker,
                        quantity=position['quantity'], 
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

            # Check hard risk exits
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
        Calculates optimal position size using available sizing algorithms (Adaptive, Elite, or Basic).
        """
        total_equity = self.portfolio.get_total_value({ticker: price})
        
        # 1. PRIMARY: AdaptivePositionSizer (Market-aware)
        try:
            market_regime = 'NORMAL'  # Logic integration pending
            portfolio_volatility = 0.15 
            active_positions = len(self.portfolio.positions)
            max_drawdown = 0.0 
            
            sizing_result = self.position_sizer.calculate_position_size(
                portfolio_value=total_equity,
                volatility=portfolio_volatility,
                confidence=confidence,
                max_drawdown=max_drawdown,
                active_positions=active_positions,
                market_regime=market_regime,
                current_price=price
            )
            
            capital_allocated = sizing_result['position_size']
            shares = int(capital_allocated / price)
            
            self.logger.info(f"💰 ADAPTIVE [{ticker}]: cap={capital_allocated:.2f}, shares={shares}, conf={confidence:.2f}")
            return max(0, shares)
            
        except Exception as e:
            self.logger.debug(f"AdaptivePositionSizer skipped: {e}. Trying Elite fallback.")
        
        # 2. SECONDARY: EliteRiskSizer (Correlation-aware Kelly)
        if self.elite_risk_sizer:
            try:
                portfolio_vol = 0.15 
                correlation_matrix = {} 
                
                position_pct, sizing_details = self.elite_risk_sizer.compute_optimal_position_size(
                    ticker=ticker,
                    confidence=confidence,
                    prediction=0.0,
                    total_capital=total_equity,
                    ticker_volatility=portfolio_vol * 1.2,
                    portfolio_volatility=portfolio_vol,
                    portfolio_positions=self.portfolio.positions,
                    correlation_matrix=correlation_matrix
                )
                
                capital_allocated = total_equity * position_pct
                shares = int(capital_allocated / price)
                
                self.logger.info(f"💰 ELITE [{ticker}]: pct={position_pct:.2%}, shares={shares}, Kelly={sizing_details.get('stages', {}).get('kelly_size', 0):.2%}")
                return max(0, shares)
            except Exception as e:
                self.logger.debug(f"EliteRiskSizer skipped: {e}. Using Basic fallback.")
        
        # 3. BASIC: Static percentage based on capital at risk
        # Risk is pre-calculated from signals (confidence filtering happens in Consensus)
        # We use a fixed risk_per_trade_pct as the baseline allocation.
        capital_at_risk = total_equity * self.risk_per_trade_pct
        shares_from_risk = capital_at_risk / price

        # Position concentration limit
        max_position_value = total_equity * self.max_position_size_pct
        current_position_value = 0
        if ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[ticker]['quantity'] * price
        
        allowed_capital = max(0, max_position_value - current_position_value)
        shares_from_exposure = allowed_capital / price

        # Margin/Cash limit
        shares_from_cash = self.portfolio.current_balance / price

        # Combine all constraints
        final_shares = int(min(shares_from_risk, shares_from_exposure, shares_from_cash))

        self.logger.debug(f"[POSITION_SIZE] ticker={ticker}, price={price:.2f}, total_equity={total_equity:.2f}")
        self.logger.info(f"💰 BASIC [{ticker}]: shares={final_shares} (final) | constraints: risk={int(shares_from_risk)}, exposure={int(shares_from_exposure)}, cash={int(shares_from_cash)}")

        return max(0, final_shares)
    
    def rebalance_portfolio(self, 
                          current_prices: Dict[str, float],
                          volatilities: Dict[str, float],
                          target_assets: List[str] = None) -> List[TradeOrder]:
        """
        Rebalance portfolio using Risk Parity allocation mechanism.
        """
        try:
            if not target_assets:
                target_assets = list(current_prices.keys())
            
            total_value = self.portfolio.get_total_value(current_prices)
            correlations = np.eye(len(target_assets))
            
            # Risk Parity Allocation logic
            allocation_result = self.risk_allocator.allocate(
                assets=target_assets,
                volatilities={asset: volatilities.get(asset, 0.15) for asset in target_assets},
                correlations=correlations,
                target_volatility=None 
            )
            
            orders = []
            for asset, target_weight in allocation_result['weights'].items():
                if asset not in current_prices:
                    continue
                    
                target_value = total_value * target_weight
                current_value = 0
                
                if asset in self.portfolio.positions:
                    current_quantity = self.portfolio.positions[asset]['quantity']
                    current_value = current_quantity * current_prices[asset]
                
                value_adjustment = target_value - current_value
                shares_adjustment = int(value_adjustment / current_prices[asset])
                
                if abs(shares_adjustment) > 0:
                    action = 'BUY' if shares_adjustment > 0 else 'SELL'
                    orders.append(TradeOrder(
                        ticker=asset,
                        quantity=abs(shares_adjustment),
                        price=current_prices[asset],
                        action=action,
                        reason="Risk Parity Rebalancing"
                    ))
            
            self.logger.info(f"📊 Portfolio rebalanced: {len(orders)} optimization orders generated.")
            return orders
            
        except Exception as e:
            self.logger.error(f"❌ Critical failure during portfolio rebalancing: {e}")
            return []
