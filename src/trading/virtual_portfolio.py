"""
Virtual Portfolio - Virtual account for paper trading with real prices.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.advanced.advanced_engine import TransactionCostModel
from src.config.unified_config_manager import get_current_config
from src.core.error_handling.error_handler import get_error_handler
from src.core.logging.logger import ProjectLogger
from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator

logger = ProjectLogger.get_logger('VirtualPortfolio')
error_handler = get_error_handler()


class VirtualPortfolio:
    """
    Virtual portfolio for paper trading.
    Uses real prices but virtual money. Integrated with project-wide config and metrics.
    """

    def __init__(self, initial_balance: float=10000.0, portfolio_name: str=
        'default'):
        self.config_manager = get_current_config()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio_name = portfolio_name
        self.positions: dict[str, Any] = {}
        self.transactions: list[dict[str, Any]] = []
        self.performance_history: list[dict[str, Any]] = []
        self.metrics_calculator = PortfolioMetricsCalculator()
        risk_config = self.config_manager.get('strategy.risk_management', {})
        # THE CONFIGURED NAME, NOT A NEIGHBOURING ONE. Until 2026-09-06 this
        # asked for `max_position_size` while the config declares
        # `max_position_size_pct`, so the configured value never once reached
        # here and the limit in force was the default written on this line.
        # Both happen to be 0.10, which is exactly why nobody noticed
        # (REGISTER #288). The old name is still accepted so an existing
        # deployment that set it does not silently lose its setting.
        self.max_position_size = float(
            risk_config.get('max_position_size_pct',
                            risk_config.get('max_position_size', 0.1)))
        self.max_total_risk = float(
            risk_config.get('max_total_risk_pct',
                            risk_config.get('max_total_risk', 0.3)))
        self.stop_loss_pct = risk_config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = risk_config.get('take_profit_pct', 0.1)
        cost_config = self.config_manager.get(
            'backtesting.transaction_costs', {})
        self.transaction_cost_model = TransactionCostModel(cost_config)
        portfolio_dir = Path('data/portfolios')
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = (portfolio_dir /
            f'{portfolio_name}_portfolio.json')
        self.load_portfolio()
        logger.info(
            f"Virtual portfolio '{portfolio_name}' initialized with ${initial_balance:,.2f}"
            )

    def load_portfolio(self):
        """Loads portfolio state from disk."""
        try:
            if not self.portfolio_file.exists():
                self.save_portfolio()
                return
            with open(self.portfolio_file, encoding='utf-8') as f:
                data = json.load(f)
            self._load_portfolio_data(data)
            logger.info(f'Portfolio loaded from {self.portfolio_file}')
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"⚠️ Portfolio file {self.portfolio_file} is corrupted. Resetting. Error: {e}")
            self.reset_portfolio()
        except (TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            error_handler.handle_error(e, 'Loading Virtual Portfolio')
            raise

    def _load_portfolio_data(self, data: dict[str, Any]):
        """Load portfolio data from loaded JSON."""
        self.current_balance = data.get('current_balance', self.initial_balance
            )
        self.positions = data.get('positions', {})
        self.transactions = data.get('transactions', [])
        self.performance_history = data.get('performance_history', [])
        self._convert_position_timestamps()
        self._convert_transaction_timestamps()

    def _convert_position_timestamps(self):
        """Convert position timestamps to datetime objects."""
        for pos in self.positions.values():
            if 'entry_time' in pos and isinstance(pos['entry_time'], str):
                pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])

    def _convert_transaction_timestamps(self):
        """Convert transaction timestamps to datetime objects."""
        for tx in self.transactions:
            if 'timestamp' in tx and isinstance(tx['timestamp'], str):
                tx['timestamp'] = datetime.fromisoformat(tx['timestamp'])

    def save_portfolio(self):
        """Saves portfolio state to disk."""
        try:
            data = self._prepare_portfolio_data()
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Portfolio saved to {self.portfolio_file}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Виникла помилка: {e}')
            error_handler.handle_error(e, 'Saving Virtual Portfolio')
            raise

    def _prepare_portfolio_data(self) ->dict[str, Any]:
        """Prepare portfolio data for JSON serialization."""
        return {'portfolio_name': self.portfolio_name, 'initial_balance':
            self.initial_balance, 'current_balance': self.current_balance,
            'positions': self._serialize_positions(), 'transactions': self.
            _serialize_transactions(), 'performance_history': self.
            performance_history, 'last_updated': datetime.now().isoformat()}

    def _serialize_positions(self) ->dict[str, Any]:
        """Serialize positions for JSON storage."""
        serialized = {}
        for ticker, pos in self.positions.items():
            pos_copy = pos.copy()
            if isinstance(pos_copy.get('entry_time'), datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
            serialized[ticker] = pos_copy
        return serialized

    def _serialize_transactions(self) ->list[dict[str, Any]]:
        """Serialize transactions for JSON storage."""
        serialized = []
        for tx in self.transactions:
            tx_copy = tx.copy()
            if isinstance(tx_copy.get('timestamp'), datetime):
                tx_copy['timestamp'] = tx_copy['timestamp'].isoformat()
            serialized.append(tx_copy)
        return serialized

    def get_total_value(self, current_prices: dict[str, float]) ->float:
        """Calculates total portfolio value (cash + mark-to-market positions)."""
        total_value = self.current_balance
        for ticker, position in self.positions.items():
            price = current_prices.get(ticker)
            if price:
                total_value += position['quantity'] * price
        return total_value

    def get_daily_drawdown(
        self,
        current_prices: dict[str, float],
        as_of: datetime | None = None,
    ) -> float:
        """Loss since the start of the trading day named by ``as_of``.

        The day boundary came from the wall clock while ``update_performance``
        stamped its records with the wall clock too. Inside one backtest or one
        automated session every record therefore carried today's date, nothing
        satisfied ``record_date < today``, and the anchor fell through to
        ``performance_history[0]`` -- the run's opening equity. What the
        function returned was the loss since the run began, under a name that
        says daily, to a caller that latches a kill switch on it.

        Callers now pass the bar's own time. Live callers pass nothing and get
        the clock, exactly as before.

        The no-prior-day fallback now anchors to the day's own first valuation
        rather than to the first one ever taken. Where every record shares a
        date those are the same record, so this changes no existing behaviour;
        it stops the fallback from meaning "since inception" once bar times
        make the two differ.
        """
        current_value = self.get_total_value(current_prices)
        if current_value == 0:
            return 0.0

        today_date = (as_of or datetime.now()).date()
        start_value = None

        # Last valuation from any earlier day: the day's opening equity.
        for record in reversed(self.performance_history):
            record_date = datetime.fromisoformat(record['timestamp']).date()
            if record_date < today_date:
                start_value = record['total_value']
                break

        # No earlier day on record: anchor to today's first valuation, not to
        # the first one ever taken.
        if start_value is None:
            for record in self.performance_history:
                if datetime.fromisoformat(record['timestamp']).date() == today_date:
                    start_value = record['total_value']
                    break

        if start_value is None:
            start_value = self.initial_balance

        if start_value == 0:
            return 0.0

        return (current_value - start_value) / start_value

    def _held_value(self, current_prices: dict[str, float] | None = None
                    ) -> dict[str, float]:
        """What each position is worth, marked to market where a price is given.

        Falls back to `avg_price` -- the cost basis, which already includes the
        costs paid -- for any name without a quoted price. Stated rather than
        silent because the two answers differ after a move, and a limit checked
        against the wrong one binds at the wrong place.
        """
        prices = current_prices or {}
        return {ticker: position['quantity'] * float(
                    prices.get(ticker, position['avg_price']))
                for ticker, position in self.positions.items()}

    def _refuse_if_over_limit(self, ticker: str, trade_value: float,
                              current_prices: dict[str, float] | None = None
                              ) -> dict[str, Any] | None:
        """The declared risk limits, enforced. Returns a refusal or None.

        WHY THIS EXISTS. `strategy.risk_management` declared
        `max_position_size_pct: 0.10` and `max_drawdown_pct: 0.15` and the rest,
        and until 2026-09-06 `buy_stock` rejected an order for exactly one
        reason: not enough cash. The config described a risk system that did not
        exist, and read as configured (REGISTER #101, #288).

        The numbers are NOT invented here. They are the ones already written in
        the config -- 10% of the portfolio in one name, 30% invested in total --
        so this changes what the code DOES without changing what the owner
        decided. `max_total_risk_pct` is declared at 0.30 because that was
        already this class's hardcoded default: the line makes visible what was
        in force, it does not choose anew.

        Checked at cost basis unless prices are supplied. That is the
        conservative side for a buy: a position whose price has risen is worth
        MORE than its basis, so cost basis can only under-state exposure, and
        the caller who wants the tighter answer passes prices.
        """
        held = self._held_value(current_prices)
        portfolio_value = self.current_balance + sum(held.values())
        if portfolio_value <= 0:
            return None

        after_position = held.get(ticker, 0.0) + trade_value
        if after_position / portfolio_value > self.max_position_size:
            return {'success': False, 'error':
                    f'Position limit: {ticker} would be '
                    f'{after_position / portfolio_value:.1%} of the portfolio, '
                    f'over the configured {self.max_position_size:.1%} '
                    f'(strategy.risk_management.max_position_size_pct)'}

        after_total = sum(held.values()) + trade_value
        if after_total / portfolio_value > self.max_total_risk:
            return {'success': False, 'error':
                    f'Total risk limit: {after_total / portfolio_value:.1%} of '
                    f'the portfolio would be invested, over the configured '
                    f'{self.max_total_risk:.1%} '
                    f'(strategy.risk_management.max_total_risk_pct)'}
        return None

    def buy_stock(self, order_params: dict[str, Any]) ->dict[str, Any]:
        """Executes a virtual buy order with transaction costs."""
        try:
            ticker = order_params['ticker']
            quantity = order_params['quantity']
            price = order_params['price']
            confidence = order_params.get('confidence', 0.8)
            trade_value = quantity * price
            daily_volume = order_params.get('daily_volume', 1000000)
            volatility = order_params.get('volatility', 0.02)
            order_size_pct = (quantity * price / daily_volume if
                daily_volume > 0 else 0.01)
            cost_breakdown = (self.transaction_cost_model.
                calculate_execution_costs(trade_value=trade_value,
                daily_volume=daily_volume, volatility=volatility,
                order_size_pct=order_size_pct))
            total_cost = trade_value + cost_breakdown['total']
            if total_cost > self.current_balance:
                return {'success': False, 'error':
                    'Insufficient funds including transaction costs'}
            # The funds check stays first so no existing refusal changes its
            # wording; the limits only ADD refusals that never happened before.
            refusal = self._refuse_if_over_limit(
                ticker, trade_value, order_params.get('current_prices'))
            if refusal is not None:
                logger.info(
                    f"REFUSED {quantity} {ticker} at ${price:.2f}: "
                    f"{refusal['error']}")
                return refusal
            transaction = self._create_buy_transaction(order_params,
                trade_value, cost_breakdown)
            self._process_buy_order(ticker, quantity, price, trade_value,
                total_cost, confidence)
            self.transactions.append(transaction)
            self.save_portfolio()
            logger.info(
                f"BOUGHT {quantity} {ticker} at ${price:.2f} (Costs: ${cost_breakdown['total']:.2f})"
                )
            return {'success': True, 'transaction': transaction}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            error_handler.handle_error(e,
                f"Buy Stock {order_params.get('ticker', 'unknown')}")
            return {'success': False, 'error': str(e)}

    def _create_buy_transaction(self, order_params: dict[str, Any],
        trade_value: float, cost_breakdown: dict[str, Any]) ->dict[str, Any]:
        """Create buy transaction record with cost breakdown."""
        return {'timestamp': datetime.now(), 'type': 'BUY', 'ticker':
            order_params['ticker'], 'quantity': order_params['quantity'],
            'price': order_params['price'], 'trade_value': trade_value,
            'transaction_costs': cost_breakdown, 'total_cost': trade_value +
            cost_breakdown['total'], 'reason': order_params.get('reason',
            ''), 'confidence': order_params.get('confidence', 0.8)}

    def _process_buy_order(self, ticker: str, quantity: int, price: float,
        trade_value: float, total_cost: float, confidence: float):
        """Process buy order and update positions."""
        self.current_balance -= total_cost
        if ticker in self.positions:
            old_qty = self.positions[ticker]['quantity']
            old_avg = self.positions[ticker]['avg_price']
            new_qty = old_qty + quantity
            self.positions[ticker]['avg_price'] = (old_qty * old_avg +
                total_cost) / new_qty
            self.positions[ticker]['quantity'] = new_qty
        else:
            self.positions[ticker] = {'quantity': quantity, 'avg_price':
                total_cost / quantity, 'entry_time': datetime.now(),
                'confidence': confidence}

    def sell_stock(self, ticker: str, quantity: int, price: float, reason:
        str='', daily_volume: float=1000000, volatility: float=0.02) ->dict[
        str, Any]:
        """Executes a virtual sell order with transaction costs."""
        try:
            if ticker not in self.positions or self.positions[ticker][
                'quantity'] < quantity:
                return {'success': False, 'error': 'Insufficient position'}
            pos = self.positions[ticker]
            trade_value = quantity * price
            cost_basis = quantity * pos['avg_price']
            order_size_pct = (quantity * price / daily_volume if
                daily_volume > 0 else 0.01)
            cost_breakdown = (self.transaction_cost_model.
                calculate_execution_costs(trade_value=trade_value,
                daily_volume=daily_volume, volatility=volatility,
                order_size_pct=order_size_pct))
            net_revenue = trade_value - cost_breakdown['total']
            pnl = net_revenue - cost_basis
            transaction = {'timestamp': datetime.now(), 'type': 'SELL',
                'ticker': ticker, 'quantity': quantity, 'price': price,
                'trade_value': trade_value, 'transaction_costs':
                cost_breakdown, 'net_revenue': net_revenue, 'pnl': pnl,
                'pnl_pct': pnl / cost_basis * 100 if cost_basis != 0 else 0,
                'reason': reason}
            self.current_balance += net_revenue
            if quantity == pos['quantity']:
                del self.positions[ticker]
            else:
                self.positions[ticker]['quantity'] -= quantity
            self.transactions.append(transaction)
            self.save_portfolio()
            logger.info(
                f"SOLD {quantity} {ticker} at ${price:.2f} (Net: ${net_revenue:.2f}, Costs: ${cost_breakdown['total']:.2f}, PnL: ${pnl:.2f})"
                )
            return {'success': True, 'transaction': transaction}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            error_handler.handle_error(e, f'Sell Stock {ticker}')
            return {'success': False, 'error': str(e)}

    def get_portfolio_summary(self, current_prices: dict[str, float]) ->dict[
        str, Any]:
        """
        Generates a comprehensive portfolio report, including risk metrics
        from PortfolioMetricsCalculator.
        """
        total_value = self.get_total_value(current_prices)
        perf_df = pd.DataFrame(self.performance_history)
        metrics = {}
        if not perf_df.empty and 'total_value' in perf_df.columns:
            equity_curve = perf_df.set_index(pd.to_datetime(perf_df[
                'timestamp']))['total_value']
            equity_curve[datetime.now()] = total_value
            metrics = self.metrics_calculator.calculate(equity_curve)
        positions_report = []
        for ticker, pos in self.positions.items():
            curr_price = current_prices.get(ticker, pos['avg_price'])
            val = pos['quantity'] * curr_price
            pnl = val - pos['quantity'] * pos['avg_price']
            positions_report.append({'ticker': ticker, 'qty': pos[
                'quantity'], 'avg_price': pos['avg_price'], 'market_price':
                curr_price, 'value': val, 'pnl': pnl, 'pnl_pct': pnl / (pos
                ['quantity'] * pos['avg_price']) * 100})
        return {'portfolio_name': self.portfolio_name, 'balance': self.
            current_balance, 'total_value': total_value, 'positions':
            positions_report, 'metrics': metrics, 'timestamp': datetime.now
            ().isoformat()}

    def update_performance(
        self,
        current_prices: dict[str, float],
        as_of: datetime | None = None,
    ):
        """Records current portfolio valuation into history.

        ``as_of`` is the time the valuation belongs to -- the bar's time in a
        backtest. Stamping the wall clock instead is what collapsed every
        record in a run onto a single date and made daily drawdown mean
        something else entirely.
        """
        total_val = self.get_total_value(current_prices)
        record = {'timestamp': (as_of or datetime.now()).isoformat(),
            'total_value': total_val, 'cash': self.current_balance,
            'positions_count': len(self.positions)}
        self.performance_history.append(record)
        if len(self.performance_history) > 5000:
            self.performance_history = self.performance_history[-5000:]
        self.save_portfolio()

    def reset_portfolio(self):
        """Wipes the portfolio state."""
        self.current_balance = self.initial_balance
        self.positions = {}
        self.transactions = []
        self.performance_history = []
        self.save_portfolio()
        logger.info(f"Portfolio '{self.portfolio_name}' has been reset.")
