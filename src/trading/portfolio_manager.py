"""
Acts as the Risk Officer for the trading system.

This module is responsible for risk management, position sizing, and generating
trade orders based on signals from the Consensus Engine. It does not manage
portfolio state directly but queries a VirtualPortfolio instance.
"""
import logging
from typing import Any

import numpy as np

from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer, PositionSizingParams
from src.algorithms.risk_parity_allocator import RiskParityAllocator
from src.core.logging.logger import ProjectLogger
from src.trading.virtual_portfolio import VirtualPortfolio

from .trader import TradeOrder

logger = ProjectLogger.get_logger('PortfolioManager')


class PortfolioManager:
    """
    Manages portfolio-level risk, position sizing, and order generation.
    """

    def __init__(self, virtual_portfolio: VirtualPortfolio,
        elite_risk_sizer=None, config: (dict[str, Any] | None)=None):
        """
        Args:
            virtual_portfolio: The stateful portfolio object.
            elite_risk_sizer: EliteRiskSizer for optimal position sizing (Kelly + correlation-aware).
            config: Risk management configuration.
        """
        self.portfolio = virtual_portfolio
        self.elite_risk_sizer = elite_risk_sizer
        self.logger = logger
        risk_config = config if config is not None else {}
        self.risk_per_trade_pct = risk_config.get('risk_per_trade_pct', 0.03)
        self.max_position_size_pct = risk_config.get('max_position_size_pct',
            0.1)
        self.max_daily_drawdown_pct = risk_config.get('max_daily_drawdown_pct',
            0.05)
        self.position_sizer = AdaptivePositionSizer(config=risk_config.get(
            'position_sizer', {}))
        self.risk_allocator = RiskParityAllocator(config=risk_config.get(
            'risk_allocator', {}))
        self.kill_switch_active = False

    def is_trading_allowed(self, current_prices: dict[str, float]) ->bool:
        """
        Primary gatekeeper. Checks if any risk rule prevents trading.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'[PORTFOLIO] is_trading_allowed called with {len(current_prices)} prices'
                )
        if self.kill_switch_active:
            self.logger.critical('Trading blocked: KILL SWITCH IS ACTIVE.')
            return False
        if hasattr(self.portfolio, 'get_daily_drawdown'
            ) and self.portfolio.get_daily_drawdown(current_prices
            ) < -self.max_daily_drawdown_pct:
            self.logger.critical(
                f'Trading blocked: Max daily drawdown of {self.max_daily_drawdown_pct:.2%} exceeded.'
                )
            self.kill_switch_active = True
            return False
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('[PORTFOLIO] is_trading_allowed: TRUE')
        return True

    def generate_orders_from_signals(self, signals: list[dict[str, Any]],
        current_prices: dict[str, float]) ->list[TradeOrder]:
        """
        Processes signals from the ConsensusEngine and generates executable TradeOrders.
        """
        self.logger.info(
            f'[PORTFOLIO] generate_orders_from_signals called with {len(signals)} signals'
            )
        if not self.is_trading_allowed(current_prices):
            self.logger.warning(
                '[PORTFOLIO] Trading is NOT allowed by risk protocol!')
            return []
        self.logger.info(
            '[PORTFOLIO] Trading protocol cleared, processing signals...')
        orders = []
        for signal in signals:
            order = self._process_single_signal(signal, current_prices)
            if order:
                orders.append(order)
        return orders

    def _process_single_signal(self, signal: dict[str, Any], current_prices:
        dict[str, float]) ->(TradeOrder | None):
        """Process an individual signal and return an order if valid."""
        action = signal.get('final_signal')
        ticker = str(signal.get('ticker', ''))
        if not ticker or ticker not in current_prices:
            return None
        price = float(current_prices[ticker])
        confidence = float(signal.get('confidence', 0.5))
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'[PORTFOLIO] Analyzing signal: ticker={ticker}, action={action}, confidence={confidence}, price={price}'
                )
        if action == 'BUY':
            return self._create_buy_order(ticker, price, confidence, signal
                .get('report'))
        if action == 'SELL':
            return self._create_sell_order(ticker, price)
        if action:
            self.logger.warning(
                f"Skipping unhandled action '{action}' for {ticker}")
        return None

    def _create_buy_order(self, ticker: str, price: float, confidence:
        float, report: (Any | None)) ->(TradeOrder | None):
        """Create a BUY order with calculated position sizing."""
        regime = 'NORMAL'
        if report and hasattr(report, 'market_regime'):
            regime = str(report.market_regime)
        shares = self._calculate_position_size(ticker, price, confidence,
            regime=regime)
        if shares > 0:
            return TradeOrder(ticker=ticker, quantity=shares, price=price,
                action='BUY', reason=
                f'Consensus Signal (Conf: {confidence:.2f}, Regime: {regime})')
        return None

    def _create_sell_order(self, ticker: str, price: float) ->(TradeOrder |
        None):
        """Create a SELL order to close an existing position."""
        position = self.portfolio.positions.get(ticker)
        if position and position['quantity'] > 0:
            quantity_value = position['quantity']
            quantity_int = int(quantity_value) if isinstance(quantity_value,
                (int, float)) else 0
            return TradeOrder(ticker=ticker, quantity=quantity_int, price=
                price, action='SELL', reason='Consensus Signal (SELL)')
        return None

    def check_risk_exits(self, current_prices: dict[str, float]) ->list[
        TradeOrder]:
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
            stop_loss = position.get('stop_loss')
            take_profit = position.get('take_profit')
            if stop_loss and current_price <= stop_loss:
                exit_orders.append(TradeOrder(ticker=ticker, quantity=
                    position['quantity'], price=current_price, action=
                    'SELL', reason='Stop-Loss Triggered'))
            elif take_profit and current_price >= take_profit:
                exit_orders.append(TradeOrder(ticker=ticker, quantity=
                    position['quantity'], price=current_price, action=
                    'SELL', reason='Take-Profit Triggered'))
        return exit_orders

    def _calculate_position_size(self, ticker: str, price: float,
        confidence: float, regime: str='NORMAL') ->int:
        """
        Calculates optimal position size using available sizing algorithms (Adaptive, Elite, or Basic).
        """
        total_equity = self.portfolio.get_total_value({ticker: price})
        try:
            market_regime = regime.upper()
            portfolio_volatility = 0.15
            active_positions = len(self.portfolio.positions)
            max_drawdown = 0.0
            params = PositionSizingParams(portfolio_value=total_equity,
                volatility=portfolio_volatility, confidence=confidence,
                max_drawdown=max_drawdown, active_positions=
                active_positions, market_regime=market_regime,
                current_price=price)
            sizing_result = self.position_sizer.calculate_position_size(params)
            capital_allocated = sizing_result['position_size']
            shares = int(capital_allocated / price) if price > 0 else 0
            self.logger.info(
                f'💰 ADAPTIVE [{ticker}]: cap={capital_allocated:.2f}, shares={shares}, conf={confidence:.2f}'
                )
            return max(0, shares)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f'AdaptivePositionSizer skipped: {e}. Trying Elite fallback.')
        if self.elite_risk_sizer:
            try:
                portfolio_vol = 0.15
                correlation_matrix: dict[str, Any] = {}
                position_pct, sizing_details = (self.elite_risk_sizer.
                    compute_optimal_position_size(ticker=ticker, confidence
                    =confidence, prediction=0.0, total_capital=total_equity,
                    ticker_volatility=portfolio_vol * 1.2,
                    portfolio_volatility=portfolio_vol, portfolio_positions
                    =self.portfolio.positions, correlation_matrix=
                    correlation_matrix, current_price=price))
                capital_allocated = total_equity * position_pct
                shares = int(capital_allocated / price) if price > 0 else 0
                self.logger.info(
                    f"💰 ELITE [{ticker}]: pct={position_pct:.2%}, shares={shares}, Kelly={sizing_details.get('stages', {}).get('kelly_size', 0):.2%}"
                    )
                return max(0, shares)
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'EliteRiskSizer skipped: {e}. Using Basic fallback.')
        capital_at_risk = total_equity * self.risk_per_trade_pct
        shares_from_risk = capital_at_risk / price
        max_position_value = total_equity * self.max_position_size_pct
        current_position_value = 0
        if ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[ticker][
                'quantity'] * price
        allowed_capital = max(0, max_position_value - current_position_value)
        shares_from_exposure = allowed_capital / price
        shares_from_cash = self.portfolio.current_balance / price
        final_shares = int(min(shares_from_risk, shares_from_exposure,
            shares_from_cash))
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'[POSITION_SIZE] ticker={ticker}, price={price:.2f}, total_equity={total_equity:.2f}'
                )
        self.logger.info(
            f'💰 BASIC [{ticker}]: shares={final_shares} (final) | constraints: risk={int(shares_from_risk)}, exposure={int(shares_from_exposure)}, cash={int(shares_from_cash)}'
            )
        return max(0, final_shares)

    def optimize_allocation(self, current_prices: dict[str, float],
        volatilities: dict[str, float], target_assets: (list[str] | None)=None
        ) ->list[TradeOrder]:
        """
        Rebalance portfolio using Risk Parity allocation mechanism.
        """
        try:
            if not target_assets:
                target_assets = list(current_prices.keys())
            total_value = self.portfolio.get_total_value(current_prices)
            correlations = np.eye(len(target_assets))
            allocation_result = self.risk_allocator.allocate(assets=
                target_assets, volatilities={asset: volatilities.get(asset,
                0.15) for asset in target_assets}, correlations=
                correlations, params={'target_volatility': None})
            orders = []
            for asset, target_weight in allocation_result['weights'].items():
                if asset not in current_prices:
                    continue
                target_value = total_value * target_weight
                current_value = 0
                if asset in self.portfolio.positions:
                    current_quantity = self.portfolio.positions[asset][
                        'quantity']
                    current_value = current_quantity * current_prices[asset]
                value_adjustment = target_value - current_value
                shares_adjustment = int(value_adjustment / current_prices[
                    asset])
                if abs(shares_adjustment) > 0:
                    action = 'BUY' if shares_adjustment > 0 else 'SELL'
                    orders.append(TradeOrder(ticker=asset, quantity=abs(
                        shares_adjustment), price=current_prices[asset],
                        action=action, reason='Risk Parity Rebalancing'))
            self.logger.info(
                f'📊 Portfolio rebalanced: {len(orders)} optimization orders generated.'
                )
            return orders
        except Exception as e:
            self.logger.error(
                f'❌ Critical failure during portfolio rebalancing: {e}')
            raise RuntimeError("Critical failure during portfolio rebalancing") from e
