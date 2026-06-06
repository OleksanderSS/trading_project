"""
Elite Risk Sizing Engine
- Kelly Criterion для оптимальної фракції
- Correlation-aware diversification factor
- Dynamic adjustment за волатильністю
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class EliteRiskSizer:
    """
    Calculates optimal position size using:
    1. Kelly Criterion (core formula)
    2. Correlation adjustment (decrease if correlated with portfolio)
    3. Volatility scaling (more volatile = smaller positions)
    """

    def __init__(self, logger=None, kelly_fraction=0.25):
        """
        Args:
            kelly_fraction: Conservative fraction (0.25 = 1/4 Kelly), do not use full Kelly
        """
        self.logger = logger or ProjectLogger.get_logger(__name__)
        self.kelly_fraction = kelly_fraction
        self.historical_returns = {}
        self.portfolio_correlation_matrix = None

    def update_returns_data(self, ticker: str, returns: pd.Series):
        """Update historical returns data"""
        self.historical_returns[ticker] = returns

    def calculate_optimal_position_size(self, ticker: str, entry_price:
        float, win_rate: float, avg_win_loss_ratio: float,
        current_positions: dict[str, dict], total_equity: float,
        position_value_limit: float, portfolio_volatility: float,
        cash_available: float) ->int:
        """
        Calculate optimal position size

        Args:
            ticker: Asset to buy
            entry_price: Entry price
            win_rate: % winning trades (0-1)
            avg_win_loss_ratio: Average win / average loss
            current_positions: {ticker: {quantity, entry_price, ...}}
            total_equity: Current portfolio equity
            position_value_limit: Max % of portfolio in a single position (e.g., 0.10)
            portfolio_volatility: Current portfolio volatility (annualized)
            cash_available: Cash available

        Returns:
            Optimal share quantity
        """
        if win_rate <= 0 or avg_win_loss_ratio <= 0:
            self.logger.warning(f'Invalid win rate or ratio for {ticker}')
            return 0
        kelly_f = (win_rate * avg_win_loss_ratio - (1 - win_rate)
            ) / avg_win_loss_ratio
        kelly_f = max(0, min(kelly_f, 1.0))
        fractional_kelly = kelly_f * self.kelly_fraction
        capital_at_risk = total_equity * fractional_kelly
        self.logger.info(
            f'[KELLY] {ticker}: win_rate={win_rate:.1%}, ratio={avg_win_loss_ratio:.2f}'
            )
        self.logger.info(
            f'  Kelly f*={kelly_f:.3f}, fractional={fractional_kelly:.3f}')
        self.logger.info(f'  capital_at_risk=${capital_at_risk:.2f}')
        correlation_factor = self._compute_correlation_factor(ticker,
            current_positions)
        capital_at_risk *= correlation_factor
        self.logger.info(
            f'  correlation_factor={correlation_factor:.2f} (adjusted: ${capital_at_risk:.2f})'
            )
        ticker_volatility = self._estimate_ticker_volatility(ticker)
        vol_factor = portfolio_volatility / max(ticker_volatility, 0.01)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        capital_at_risk *= vol_factor
        self.logger.info(
            f'  ticker_volatility={ticker_volatility:.2%}, portfolio_vol={portfolio_volatility:.2%}'
            )
        self.logger.info(
            f'  vol_factor={vol_factor:.2f} (adjusted: ${capital_at_risk:.2f})'
            )
        max_position_value = total_equity * position_value_limit
        capital_at_risk = min(capital_at_risk, max_position_value)
        existing_position = current_positions.get(ticker, {})
        existing_value = existing_position.get('quantity', 0) * entry_price
        additional_capital = max(0, capital_at_risk - existing_value)
        additional_capital = min(additional_capital, cash_available)
        self.logger.info(
            f'  constraints: max_pos=${max_position_value:.2f}, existing=${existing_value:.2f}, cash=${cash_available:.2f}'
            )
        self.logger.info(
            f'  final_capital_at_risk=${capital_at_risk:.2f}, additional=${additional_capital:.2f}'
            )
        if entry_price <= 0:
            return 0
        shares = int(additional_capital / entry_price)
        return max(0, shares)

    def _compute_correlation_factor(self, ticker: str, current_positions: dict
        ) ->float:
        """
        Compute factor based on correlation with portfolio

        If new ticker highly correlates with current positions:
        - factor = 0.5 (half size)

        If independent:
        - factor = 1.0 (full size)

        Якщо neg correlated:
        - factor = 1.2 (even more due to diversification)
        """
        if not current_positions or ticker not in self.historical_returns:
            return 1.0
        try:
            new_ticker_returns = self.historical_returns[ticker]
            correlations = []
            for pos_ticker in current_positions.keys():
                if pos_ticker in self.historical_returns:
                    pos_returns = self.historical_returns[pos_ticker]
                    common_dates = new_ticker_returns.index.intersection(
                        pos_returns.index)
                    if len(common_dates) > 20:
                        corr = new_ticker_returns[common_dates].corr(
                            pos_returns[common_dates])
                        correlations.append(corr)
            if not correlations:
                return 1.0
            avg_correlation = np.mean(correlations)
            factor = 1.0 - avg_correlation * 0.3
            factor = np.clip(factor, 0.3, 1.3)
            self.logger.info(
                f'[CORRELATION] {ticker} vs portfolio: avg_corr={avg_correlation:.2f}, factor={factor:.2f}'
                )
            return float(factor)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Correlation calculation failed: {e}')
            return 1.0

    def _estimate_ticker_volatility(self, ticker: str) ->float:
        """
        Estimate annualized volatility for ticker
        """
        if ticker not in self.historical_returns:
            return 0.2
        try:
            returns = self.historical_returns[ticker]
            if len(returns) < 10:
                return 0.2
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            return float(annual_vol)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Volatility estimation failed: {e}')
            return 0.2

    def rebalance_portfolio(self, current_positions: dict[str, dict],
        target_allocations: dict[str, float], current_prices: dict[str,
        float], total_equity: float) ->dict[str, int]:
        """
        Calculate trades to reach target allocations

        Args:
            current_positions: {ticker: {quantity, entry_price}}
            target_allocations: {ticker: 0.10}  (10% each)
            current_prices: {ticker: price}
            total_equity: Current portfolio value

        Returns:
            {ticker: quantity_to_trade}  (+ = buy, - = sell)
        """
        rebalance_trades = {}
        for ticker, target_pct in target_allocations.items():
            target_value = total_equity * target_pct
            current_position = current_positions.get(ticker, {})
            current_quantity = current_position.get('quantity', 0)
            current_price = current_prices.get(ticker, 0)
            if current_price <= 0:
                continue
            current_value = current_quantity * current_price
            diff_value = target_value - current_value
            diff_shares = int(diff_value / current_price)
            if diff_shares != 0:
                rebalance_trades[ticker] = diff_shares
                action = 'BUY' if diff_shares > 0 else 'SELL'
                self.logger.info(
                    f'[REBALANCE] {ticker}: {action} {abs(diff_shares)} shares (current: ${current_value:.2f}, target: ${target_value:.2f})'
                    )
        return rebalance_trades

    def compute_optimal_position_size(self, ticker: str, confidence: float,
        prediction: float, total_capital: float, ticker_volatility: float,
        portfolio_volatility: float, portfolio_positions: dict[str, Any],
        correlation_matrix: dict[str, Any] | None=None,
        current_price: float | None=None) ->tuple[float, dict[str, Any]]:
        """
        Elite sizing interface expected by PortfolioManager.

        Args:
            prediction: Expected return (%)
            total_capital: Current portfolio value

        Returns:
            (position_fraction, metadata)
        """
        win_rate = 0.55 if confidence > 0.6 else 0.51
        win_loss_ratio = 1.8 if abs(prediction) > 0.02 else 1.5
        entry_price = current_price if current_price and current_price > 0 else 100.0
        shares = self.calculate_optimal_position_size(ticker=ticker,
            entry_price=entry_price, win_rate=win_rate, avg_win_loss_ratio=
            win_loss_ratio, current_positions=portfolio_positions,
            total_equity=total_capital, position_value_limit=0.15,
            portfolio_volatility=portfolio_volatility, cash_available=
            total_capital)
        position_fraction = (shares * entry_price / total_capital if
            total_capital > 0 else 0)
        metadata = {'stages': {'kelly_size': position_fraction, 'vol_adj': 1.0}
            }
        return position_fraction, metadata
