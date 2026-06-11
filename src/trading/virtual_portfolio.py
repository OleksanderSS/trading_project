#!/usr/bin/env python3
"""
Virtual Portfolio - Virtual account for paper trading with real prices.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.advanced.advanced_engine import TransactionCostModel
from src.config.unified_config_manager import get_current_config
from src.core.error_handling.error_handler import get_error_handler
from src.core.logging.logger import ProjectLogger
from src.metrics.financial.portfolio_metrics import PortfolioMetricsCalculator

logger = ProjectLogger.get_logger("VirtualPortfolio")
error_handler = get_error_handler()


class VirtualPortfolio:
    """
    Virtual portfolio for paper trading.
    Uses real prices but virtual money. Integrated with project-wide config and metrics.
    """

    def __init__(self, initial_balance: float = 10000.0, portfolio_name: str = "default"):
        self.config_manager = get_current_config()
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio_name = portfolio_name

        # Positions: {ticker: {'quantity': int, 'avg_price': float, 'entry_time': datetime}}
        self.positions: dict[str, Any] = {}

        # Transaction history
        self.transactions: list[dict[str, Any]] = []

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.metrics_calculator = PortfolioMetricsCalculator()

        # Initialize transaction cost model
        cost_config = self.config_manager.get_config('backtest.transaction_costs', {})
        self.transaction_cost_model = TransactionCostModel(cost_config)

        # Portfolio persistence
        portfolio_dir = Path("data/portfolios")
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = portfolio_dir / f"{portfolio_name}_portfolio.json"

        self.load_portfolio()
        logger.info(f"Virtual portfolio '{portfolio_name}' initialized with ${initial_balance:,.2f}")

    def load_portfolio(self):
        """Loads portfolio state from disk."""
        try:
            if not self.portfolio_file.exists():
                self.save_portfolio()
                return

            with open(self.portfolio_file, encoding='utf-8') as f:
                data = json.load(f)

            self._load_portfolio_data(data)
            logger.info(f"Portfolio loaded from {self.portfolio_file}")

        except Exception as e:
            error_handler.handle_error(e, "Loading Virtual Portfolio")

    def _load_portfolio_data(self, data: dict[str, Any]):
        """Load portfolio data from loaded JSON."""
        self.current_balance = data.get('current_balance', self.initial_balance)
        self.positions = data.get('positions', {})
        self.transactions = data.get('transactions', [])
        self.performance_history = data.get('performance_history', [])

        self._convert_position_timestamps()
        self._convert_transaction_timestamps()

    def _convert_position_timestamps(self):
        """Convert position timestamps to datetime objects."""
        for pos in self.positions.values():
            if 'entry_time' in pos:
                pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])

    def _convert_transaction_timestamps(self):
        """Convert transaction timestamps to datetime objects."""
        for tx in self.transactions:
            if 'timestamp' in tx:
                tx['timestamp'] = datetime.fromisoformat(tx['timestamp'])

    def save_portfolio(self):
        """Saves portfolio state to disk."""
        try:
            data = self._prepare_portfolio_data()

            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Portfolio saved to {self.portfolio_file}")

        except Exception as e:
            error_handler.handle_error(e, "Saving Virtual Portfolio")

    def _prepare_portfolio_data(self) -> dict[str, Any]:
        """Prepare portfolio data for JSON serialization."""
        return {
            'portfolio_name': self.portfolio_name,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'positions': self._serialize_positions(),
            'transactions': self._serialize_transactions(),
            'performance_history': self.performance_history,
            'last_updated': datetime.now().isoformat()
        }

    def _serialize_positions(self) -> dict[str, Any]:
        """Serialize positions for JSON storage."""
        serialized = {}
        for ticker, pos in self.positions.items():
            pos_copy = pos.copy()
            if isinstance(pos_copy.get('entry_time'), datetime):
                pos_copy['entry_time'] = pos_copy['entry_time'].isoformat()
            serialized[ticker] = pos_copy
        return serialized

    def _serialize_transactions(self) -> list[dict[str, Any]]:
        """Serialize transactions for JSON storage."""
        serialized = []
        for tx in self.transactions:
            tx_copy = tx.copy()
            if isinstance(tx_copy.get('timestamp'), datetime):
                tx_copy['timestamp'] = tx_copy['timestamp'].isoformat()
            serialized.append(tx_copy)
        return serialized

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Calculates total portfolio value (cash + mark-to-market positions)."""
        total_value = self.current_balance
        for ticker, position in self.positions.items():
            price = current_prices.get(ticker)
            if price:
                total_value += position['quantity'] * price
        return total_value

    def buy_stock(self, order_params: dict[str, Any]) -> dict[str, Any]:
        """Executes a virtual buy order with transaction costs."""
        try:
            ticker = order_params['ticker']
            quantity = order_params['quantity']
            price = order_params['price']
            confidence = order_params.get('confidence', 0.8)

            # Calculate transaction costs
            trade_value = quantity * price
            daily_volume = order_params.get('daily_volume', 1000000)  # Default volume
            volatility = order_params.get('volatility', 0.02)  # Default volatility
            order_size_pct = (quantity * price) / daily_volume if daily_volume > 0 else 0.01

            cost_breakdown = self.transaction_cost_model.calculate_execution_costs(
                trade_value=trade_value,
                daily_volume=daily_volume,
                volatility=volatility,
                order_size_pct=order_size_pct
            )

            # Total cost including transaction costs
            total_cost = trade_value + cost_breakdown['total']

            if total_cost > self.current_balance:
                return {'success': False, 'error': 'Insufficient funds including transaction costs'}

            transaction = self._create_buy_transaction(order_params, trade_value, cost_breakdown)
            self._process_buy_order(ticker, quantity, price, trade_value, total_cost, confidence)

            self.transactions.append(transaction)
            self.save_portfolio()
            logger.info(f"BOUGHT {quantity} {ticker} at ${price:.2f} (Costs: ${cost_breakdown['total']:.2f})")
            return {'success': True, 'transaction': transaction}

        except Exception as e:
            error_handler.handle_error(e, f"Buy Stock {order_params.get('ticker', 'unknown')}")
            return {'success': False, 'error': str(e)}

    def _create_buy_transaction(self, order_params: dict[str, Any], trade_value: float, cost_breakdown: dict[str, Any]) -> dict[str, Any]:
        """Create buy transaction record with cost breakdown."""
        return {
            'timestamp': datetime.now(),
            'type': 'BUY',
            'ticker': order_params['ticker'],
            'quantity': order_params['quantity'],
            'price': order_params['price'],
            'trade_value': trade_value,
            'transaction_costs': cost_breakdown,
            'total_cost': trade_value + cost_breakdown['total'],
            'reason': order_params.get('reason', ''),
            'confidence': order_params.get('confidence', 0.8)
        }

    def _process_buy_order(self, ticker: str, quantity: int, price: float, trade_value: float, total_cost: float, confidence: float):
        """Process buy order and update positions."""
        self.current_balance -= total_cost

        if ticker in self.positions:
            old_qty = self.positions[ticker]['quantity']
            old_avg = self.positions[ticker]['avg_price']
            new_qty = old_qty + quantity
            # Include transaction costs in average price calculation
            self.positions[ticker]['avg_price'] = ((old_qty * old_avg) + total_cost) / new_qty
            self.positions[ticker]['quantity'] = new_qty
        else:
            self.positions[ticker] = {
                'quantity': quantity,
                'avg_price': total_cost / quantity,  # Include costs in avg price
                'entry_time': datetime.now(),
                'confidence': confidence
            }

    def sell_stock(self, ticker: str, quantity: int, price: float, reason: str = "",
                daily_volume: float = 1000000, volatility: float = 0.02) -> dict[str, Any]:
        """Executes a virtual sell order with transaction costs."""
        try:
            if ticker not in self.positions or self.positions[ticker]['quantity'] < quantity:
                return {'success': False, 'error': 'Insufficient position'}

            pos = self.positions[ticker]
            trade_value = quantity * price
            cost_basis = quantity * pos['avg_price']

            # Calculate transaction costs for selling
            order_size_pct = (quantity * price) / daily_volume if daily_volume > 0 else 0.01

            cost_breakdown = self.transaction_cost_model.calculate_execution_costs(
                trade_value=trade_value,
                daily_volume=daily_volume,
                volatility=volatility,
                order_size_pct=order_size_pct
            )

            # Net revenue after transaction costs
            net_revenue = trade_value - cost_breakdown['total']
            pnl = net_revenue - cost_basis

            transaction = {
                'timestamp': datetime.now(),
                'type': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'transaction_costs': cost_breakdown,
                'net_revenue': net_revenue,
                'pnl': pnl,
                'pnl_pct': (pnl / cost_basis) * 100 if cost_basis != 0 else 0,
                'reason': reason
            }

            self.current_balance += net_revenue
            if quantity == pos['quantity']:
                del self.positions[ticker]
            else:
                self.positions[ticker]['quantity'] -= quantity

            self.transactions.append(transaction)
            self.save_portfolio()
            logger.info(f"SOLD {quantity} {ticker} at ${price:.2f} (Net: ${net_revenue:.2f}, Costs: ${cost_breakdown['total']:.2f}, PnL: ${pnl:.2f})")
            return {'success': True, 'transaction': transaction}

        except Exception as e:
            error_handler.handle_error(e, f"Sell Stock {ticker}")
            return {'success': False, 'error': str(e)}

    def get_portfolio_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """
        Generates a comprehensive portfolio report, including risk metrics
        from PortfolioMetricsCalculator.
        """
        total_value = self.get_total_value(current_prices)

        # Prepare data for metrics calculation
        perf_df = pd.DataFrame(self.performance_history)
        metrics = {}
        if not perf_df.empty and 'total_value' in perf_df.columns:
            # Generate metrics if we have history
            equity_curve = perf_df.set_index(pd.to_datetime(perf_df['timestamp']))['total_value']
            # Append current value to series for up-to-date calc using robust pandas concatenation
            current_value_series = pd.Series({pd.Timestamp.now(): total_value})
            equity_curve = pd.concat([equity_curve, current_value_series])
            metrics = self.metrics_calculator.calculate(equity_curve)

        positions_report = []
        for ticker, pos in self.positions.items():
            curr_price = current_prices.get(ticker, pos['avg_price'])
            val = pos['quantity'] * curr_price
            pnl = val - (pos['quantity'] * pos['avg_price'])
            positions_report.append({
                'ticker': ticker,
                'qty': pos['quantity'],
                'avg_price': pos['avg_price'],
                'market_price': curr_price,
                'value': val,
                'pnl': pnl,
                'pnl_pct': (pnl / (pos['quantity'] * pos['avg_price'])) * 100
            })

        return {
            'portfolio_name': self.portfolio_name,
            'balance': self.current_balance,
            'total_value': total_value,
            'positions': positions_report,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

    def update_performance(self, current_prices: dict[str, float]):
        """Records current portfolio valuation into history."""
        total_val = self.get_total_value(current_prices)
        record = {
            'timestamp': datetime.now().isoformat(),
            'total_value': total_val,
            'cash': self.current_balance,
            'positions_count': len(self.positions)
        }
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
