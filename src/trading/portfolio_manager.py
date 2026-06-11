# src/trading/portfolio_manager.py
"""
Acts as the Risk Officer for the trading system.

This module is responsible for risk management, position sizing, and generating
trade orders based on signals from the Consensus Engine. It does not manage
portfolio state directly but queries a VirtualPortfolio instance.
"""

from typing import Any

import numpy as np

from src.algorithms.adaptive_position_sizer import (
    AdaptivePositionSizer,
    PositionSizingParams,
)
from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.trading.virtual_portfolio import VirtualPortfolio

from .trader import TradeOrder

logger = ProjectLogger.get_logger("PortfolioManager")

from src.risk.elite_risk_metrics import EliteRiskMetrics


class EqualVolatilityRiskAllocator:
    """Fallback Risk Parity Allocator using inverse volatility weights."""
    def allocate(self, assets: list[str], volatilities: dict[str, float], correlations: np.ndarray, params: dict) -> dict[str, Any]:
        inv_vols = {asset: 1.0 / max(vols, 0.01) for asset, vols in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol > 0:
            weights = {asset: inv_vol / total_inv_vol for asset, inv_vol in inv_vols.items()}
        else:
            weights = {asset: 1.0 / len(assets) for asset in assets} if assets else {}
        return {"weights": weights}


class PortfolioManager:
    """
    Unified Risk and Portfolio Management logic.
    Centralizes SL/TP, drawdown limits, and Kelly-optimal sizing.
    """
    def __init__(self,
                 virtual_portfolio: VirtualPortfolio,
                 elite_risk_sizer=None,
                 config: dict[str, Any] | None = None):
        self.portfolio = virtual_portfolio
        self.elite_risk_sizer = elite_risk_sizer
        self.logger = logger
        self.config_manager = get_current_config()
        self.risk_metrics = EliteRiskMetrics(config_manager=self.config_manager)

        # Risk parameters from unified config
        risk_cfg = self.config_manager.get_config('strategy.risk_management', {})
        self.max_position_size = risk_cfg.get('max_position_size', 0.1)
        self.max_position_size_pct = self.max_position_size  # alias for backward compatibility
        self.max_portfolio_var = risk_cfg.get('max_portfolio_var', 0.05)
        self.max_daily_drawdown = risk_cfg.get('max_daily_drawdown', 0.05)
        self.max_daily_drawdown_pct = self.max_daily_drawdown  # alias for drawdown checks
        self.risk_per_trade_pct = risk_cfg.get('risk_per_trade', 0.02)  # default 2% risk per trade

        # Position sizing engines
        self.position_sizer = AdaptivePositionSizer(config=risk_cfg.get("position_sizer", {}))
        self.risk_allocator = EqualVolatilityRiskAllocator()
        self.kill_switch_active = False

    def check_portfolio_risk(self, current_prices: dict[str, float]) -> bool:
        """Global portfolio risk check using EliteRiskMetrics."""
        if self.kill_switch_active: return False

        positions = {s: p['quantity'] for s, p in self.portfolio.positions.items()}
        total_val = self.portfolio.get_total_value(current_prices)

        # 1. Check VaR and concentration via Elite Metrics
        report = self.risk_metrics.get_risk_report(positions, current_prices, total_val)
        if report.get('risk_status') == 'high':
            self.logger.critical(f"Portfolio Risk HIGH: {report['portfolio_var_95_pct']:.2%}")
            return False

        # 2. Check drawdown limits
        summary = self.portfolio.get_portfolio_summary(current_prices)
        mdd = summary.get('metrics', {}).get('max_drawdown', 0)
        if abs(mdd) > self.max_daily_drawdown:
            self.logger.critical(f"Max Drawdown Limit Breached: {mdd:.2%}")
            return False

        return True

    def is_trading_allowed(self, current_prices: dict[str, float]) -> bool:
        """
        Primary gatekeeper. Checks if any risk rule prevents trading.
        """
        self.logger.debug(f"[PORTFOLIO] is_trading_allowed called with {len(current_prices)} prices")

        if self.kill_switch_active:
            self.logger.critical("Trading blocked: KILL SWITCH IS ACTIVE.")
            return False

        # Check for daily drawdown limit
        if (
            hasattr(self.portfolio, "get_daily_drawdown")
            and self.portfolio.get_daily_drawdown(current_prices) < -self.max_daily_drawdown_pct
        ):
            self.logger.critical(
                f"Trading blocked: Max daily drawdown of {self.max_daily_drawdown_pct:.2%} exceeded."
            )
            self.kill_switch_active = True
            return False

        self.logger.debug("[PORTFOLIO] is_trading_allowed: TRUE")
        return True

    def generate_orders_from_signals(self,
                                       signals: list[dict[str, Any]],
                                       current_prices: dict[str, float]) -> list[TradeOrder]:
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
            order = self._process_single_signal(signal, current_prices)
            if order:
                orders.append(order)
        return orders

    def _process_single_signal(self, signal: dict[str, Any], current_prices: dict[str, float]) -> TradeOrder | None:
        """Process an individual signal and return an order if valid."""
        action = signal.get('final_signal')
        ticker = str(signal.get('ticker', ''))
        if not ticker or ticker not in current_prices:
            return None

        price = float(current_prices[ticker])
        confidence = float(signal.get('confidence', 0.5))

        self.logger.debug(f"[PORTFOLIO] Analyzing signal: ticker={ticker}, action={action}, confidence={confidence}, price={price}")

        if action == 'BUY':
            return self._create_buy_order(ticker, price, confidence, signal.get('report'))
        if action == 'SELL':
            return self._create_sell_order(ticker, price)

        if action:
            self.logger.warning(f"Skipping unhandled action '{action}' for {ticker}")
        return None

    def _create_buy_order(self, ticker: str, price: float, confidence: float, report: Any | None) -> TradeOrder | None:
        """Create a BUY order with calculated position sizing."""
        regime = 'NORMAL'
        if report and hasattr(report, 'market_regime'):
            regime = str(report.market_regime)

        shares = self._calculate_position_size(ticker, price, confidence, regime=regime)
        if shares > 0:
            return TradeOrder(
                ticker=ticker,
                quantity=shares,
                price=price,
                action='BUY',
                reason=f"Consensus Signal (Conf: {confidence:.2f}, Regime: {regime})"
            )
        return None

    def _create_sell_order(self, ticker: str, price: float) -> TradeOrder | None:
        """Create a SELL order to close an existing position."""
        position = self.portfolio.positions.get(ticker)
        if position and position['quantity'] > 0:
            quantity_value = position['quantity']
            quantity_int = int(quantity_value) if isinstance(quantity_value, (int, float)) else 0
            return TradeOrder(
                ticker=ticker,
                quantity=quantity_int,
                price=price,
                action='SELL',
                reason="Consensus Signal (SELL)"
            )
        return None

    def check_risk_exits(self, current_prices: dict[str, float]) -> list[TradeOrder]:
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

    def _calculate_position_size(self, ticker: str, price: float, confidence: float, regime: str = 'NORMAL') -> int:
        """
        Calculates optimal position size using available sizing algorithms (Adaptive, Elite, or Basic).
        """
        total_equity = self.portfolio.get_total_value({ticker: price})

        # 1. PRIMARY: AdaptivePositionSizer (Market-aware)
        try:
            market_regime = regime.upper()
            portfolio_volatility = 0.15
            active_positions = len(self.portfolio.positions)
            max_drawdown = 0.0

            params = PositionSizingParams(
                portfolio_value=total_equity,
                volatility=portfolio_volatility,
                confidence=confidence,
                max_drawdown=max_drawdown,
                active_positions=active_positions,
                market_regime=market_regime,
                current_price=price
            )
            sizing_result = self.position_sizer.calculate_position_size(params)

            capital_allocated = sizing_result["position_size"]
            shares = int(capital_allocated / price)

            self.logger.info(
                f"💰 ADAPTIVE [{ticker}]: cap={capital_allocated:.2f}, shares={shares}, conf={confidence:.2f}"
            )
            return max(0, shares)

        except Exception as e:
            self.logger.debug(f"AdaptivePositionSizer skipped: {e}. Trying Elite fallback.")

        # 2. SECONDARY: EliteRiskSizer (Correlation-aware Kelly)
        if self.elite_risk_sizer:
            try:
                portfolio_vol = 0.15
                correlation_matrix: dict[str, Any] = {}

                position_pct, sizing_details = self.elite_risk_sizer.compute_optimal_position_size(
                    ticker=ticker,
                    confidence=confidence,
                    prediction=0.0,
                    total_capital=total_equity,
                    ticker_volatility=portfolio_vol * 1.2,
                    portfolio_volatility=portfolio_vol,
                    portfolio_positions=self.portfolio.positions,
                    correlation_matrix=correlation_matrix,
                )

                capital_allocated = total_equity * position_pct
                shares = int(capital_allocated / price)

                self.logger.info(
                    f"💰 ELITE [{ticker}]: pct={position_pct:.2%}, shares={shares}, Kelly={sizing_details.get('stages', {}).get('kelly_size', 0):.2%}"
                )
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

    def optimize_allocation(
        self,
        current_prices: dict[str, float],
        volatilities: dict[str, float],
        target_assets: list[str] | None = None,
    ) -> list[TradeOrder]:
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
                params={"target_volatility": None}
            )

            orders = []
            for asset, target_weight in allocation_result["weights"].items():
                if asset not in current_prices:
                    continue

                target_value = total_value * target_weight
                current_value = 0

                if asset in self.portfolio.positions:
                    current_quantity = self.portfolio.positions[asset]["quantity"]
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
