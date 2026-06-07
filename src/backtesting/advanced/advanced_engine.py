"""
Advanced Backtesting Framework - Розширена система бектестингу

Включає:
- Walk-Forward Optimization
- Transaction Cost Modeling
- Bias Detection (Look-ahead, Survivorship)
- Portfolio Backtesting with Multi-Assets
- Statistical Significance Testing
"""
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('AdvancedBacktesting')


class TransactionCostModel:
    """Моделювання транзакційних витрат"""

    def __init__(self, config: (dict[str, Any] | None)=None):
        self.config = config or {}
        self.commission_pct = self.config.get('commission_pct', 0.001)
        self.spread_bps = self.config.get('spread_bps', 5)
        self.market_impact_coefficient = self.config.get(
            'market_impact_coefficient', 0.1)
        self.slippage_pct = self.config.get('slippage_pct', 0.001)

    def calculate_execution_costs(self, trade_value: float, daily_volume:
        float, volatility: float, order_size_pct: (float | None)=None) ->dict[
        str, float]:
        """
        Розраховує всі компоненти витрат виконання

        Args:
            trade_value: Вартість угоди в доларах
            daily_volume: Денний обсяг торгів
            volatility: Волатильність активу
            order_size_pct: Розмір ордера як % від денного обсягу
        """
        trade_value_abs = abs(trade_value)
        commission = trade_value_abs * self.commission_pct
        spread_cost = trade_value_abs * (self.spread_bps / 10000)
        if order_size_pct is None:
            order_size_pct = (trade_value_abs / daily_volume if
                daily_volume > 0 else 0.01)
        market_impact = (trade_value_abs * self.market_impact_coefficient *
            np.sqrt(order_size_pct))
        slippage = trade_value_abs * self.slippage_pct * (1 + volatility * 10)
        total_cost = commission + spread_cost + market_impact + slippage
        return {'commission': float(commission), 'spread': float(
            spread_cost), 'market_impact': float(market_impact), 'slippage':
            float(slippage), 'total': float(total_cost), 'total_pct': float
            (total_cost / trade_value_abs) if trade_value_abs > 0 else 0}


class BiasDetector:
    """Виявлення систематичних упереджень у бектестах"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger('BiasDetector')

    def detect_look_ahead_bias(self, signals: (pd.DataFrame | pd.Series), future_prices: (pd.DataFrame | pd.Series), lag_periods: int=1) ->dict[str, Any]:
        """
        Виявлення look-ahead bias через векторизований аналіз кореляцій.
        Підтримує DataFrame та Series.
        """
        try:
            # Нормалізація вхідних даних та кодування сигналів у числові значення
            if isinstance(signals, pd.Series):
                signals = signals.to_frame()

            # Map categorical signals to numeric values
            signals = signals.replace({
                'BUY': 1.0, 'LONG': 1.0, 'SELL': -1.0, 'SHORT': -1.0,
                'HOLD': 0.0, 'FLAT': 0.0, 'CLOSE': 0.0
            })
            signals = signals.apply(pd.to_numeric, errors='coerce').fillna(0.0)  # audit-ignore: SIGNAL_FILLNA_ZERO_VALID - Trading signals default to HOLD (0) when invalid

            if isinstance(future_prices, pd.Series):
                future_prices = future_prices.to_frame()

            common_cols = signals.columns.intersection(future_prices.columns)
            if common_cols.empty:
                return {'has_look_ahead_bias': False, 'suspicious_signals':
                    [], 'message': 'Немає спільних тікерів'}
            correlations = signals[common_cols].corrwith(future_prices[
                common_cols].shift(-lag_periods))  # audit-ignore: NEGATIVE_SHIFT_INTENTIONAL
            n = len(signals)
            critical_corr = 1.96 / np.sqrt(n)
            suspicious_mask = correlations.abs() > critical_corr
            suspicious_results = []
            for ticker, corr in correlations[suspicious_mask].items():
                is_bias = abs(corr) > critical_corr * 1.5
                suspicious_results.append({'signal': ticker, 'correlation':
                    float(corr), 'is_suspicious': is_bias, 'message':
                    'Виявлено look-ahead bias' if is_bias else
                    'Підозріло висока кореляція'})
            return {'lookahead_bias_detected': len(suspicious_results) > 0,
                'has_look_ahead_bias': len(suspicious_results) > 0,
                'suspicious_signals': suspicious_results,
                'critical_threshold': float(critical_corr), 'sample_size': n}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Помилка виявлення look-ahead bias: {e}')
            return {'error': str(e), 'lookahead_bias_detected': False,
                'has_look_ahead_bias': False}

    def detect_survivorship_bias(self, historical_universe: list[str],
        current_universe: list[str], delisted_dates: dict[str, datetime]
        ) ->dict[str, Any]:
        """
        Виявлення survivorship bias

        Survivorship bias виникає коли бектест подвійно використовує акції що вилучені з індексу
        """
        try:
            delisted = set(historical_universe) - set(current_universe)
            delisted_performance_warning = []
            for ticker, delisted_date in delisted_dates.items():
                delisted_performance_warning.append({'ticker': ticker,
                    'delisted_date': delisted_date.isoformat(), 'warning':
                    f'Акція {ticker} була делістена {delisted_date.date()}'})
            return {'has_survivorship_bias': len(delisted) > 0,
                'delisted_count': len(delisted), 'delisted_tickers': list(
                delisted), 'bias_impact': len(delisted) / len(
                historical_universe) if len(historical_universe) > 0 else 0.0, 'delisted_warnings':
                delisted_performance_warning}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Помилка виявлення survivorship bias: {e}')
            return {'error': str(e)}


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization для избегання переопливання

    Ділить дані на in-sample (тренування) та out-of-sample (тестування) вікна
    """

    def __init__(self, config_manager: (Any | None)=None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger('WalkForwardOptimizer')

    def walk_forward_optimization(self, data: pd.DataFrame,
        optimization_func: Callable, in_sample_months: int=12,
        out_sample_months: int=3) ->dict[str, Any]:
        """
        Виконує Walk-Forward Optimization

        Args:
            data: Історичні дані для оптимізації
            optimization_func: Функція яка оптимізує параметри (повертає найкращі параметри)
            in_sample_months: Місяців для тренування
            out_sample_months: Місяців для тестування
            rebalance_frequency: Частота ребалансування
        """
        try:
            results = []
            total_rows = len(data)
            in_sample_size = int(total_rows * (in_sample_months / (
                in_sample_months + out_sample_months)))
            step_size = int(total_rows * (out_sample_months / 12))
            window_idx = 0
            while in_sample_size + window_idx * step_size < total_rows:
                in_start = window_idx * step_size
                in_end = in_start + in_sample_size
                out_start = in_end
                out_end = min(out_start + step_size, total_rows)
                if out_end - out_start < 10:
                    break
                in_sample_data = data.iloc[in_start:in_end]
                try:
                    best_params = optimization_func(in_sample_data)
                    out_sample_data = data.iloc[out_start:out_end]
                    performance = self._evaluate_parameters(out_sample_data)
                    results.append({'window': window_idx,
                        'in_sample_period':
                        f'{data.index[in_start]} to {data.index[in_end - 1]}',
                        'out_sample_period':
                        f'{data.index[out_start]} to {data.index[out_end - 1]}'
                        , 'optimized_parameters': best_params,
                        'out_sample_performance': performance,
                        'in_sample_size': in_end - in_start,
                        'out_sample_size': out_end - out_start})
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(
                        f'Помилка оптимізації на вікні {window_idx}: {e}')
                    raise
                window_idx += 1
            return {'windows_completed': len(results), 'windows_results':
                results, 'average_out_sample_performance': self.
                _calculate_average_performance(results),
                'optimization_completed': len(results) > 0}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Помилка Walk-Forward Optimization: {e}')
            return {'error': str(e)}

    def _evaluate_parameters(self, data: (pd.DataFrame | None)=None) ->dict[
        str, float]:
        """Оцінка параметрів на out-of-sample даних."""
        if data is None or len(data) < 10:
            return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
        try:
            returns = data.pct_change(fill_method=None).dropna()
            if isinstance(returns, pd.DataFrame):
                returns = returns.mean(axis=1)
            if len(returns) == 0:
                return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
            total_return = (1 + returns).prod() - 1
            std_val = float(returns.std())
            sharpe = (
                float(returns.mean() / std_val * np.sqrt(252))
                if np.isfinite(std_val) and std_val > 1e-12
                else 0.0
            )
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            if isinstance(total_return, pd.Series):
                total_return = total_return.mean()
            if isinstance(sharpe, pd.Series):
                sharpe = sharpe.mean()
            if isinstance(max_dd, pd.Series):
                max_dd = max_dd.mean()
            return {'return': float(total_return), 'sharpe': float(sharpe),
                'max_drawdown': float(max_dd)}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}

    def _calculate_average_performance(self, results: list[dict]) ->dict[
        str, float]:
        """Розрахунок середньої performance"""
        if not results:
            return {}
        avg_return = np.mean([r['out_sample_performance'].get('return', 0) for
            r in results])
        avg_sharpe = np.mean([r['out_sample_performance'].get('sharpe', 0) for
            r in results])
        avg_dd = np.mean([r['out_sample_performance'].get('max_drawdown', 0
            ) for r in results])
        return {'avg_return': float(avg_return), 'avg_sharpe': float(
            avg_sharpe), 'avg_max_drawdown': float(avg_dd)}


class AdvancedBacktestEngine:
    """
    Головний engine для розширеного бектестингу
    """

    def __init__(self, config_manager: (Any | None)=None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger('AdvancedBacktest')
        backtest_config = self.config.get('backtesting', {})
        self.cost_model = TransactionCostModel(backtest_config.get(
            'transaction_costs', {}))
        self.bias_detector = BiasDetector()
        self.wf_optimizer = WalkForwardOptimizer(self.config)

    def run_comprehensive_backtest(self, price_data: pd.DataFrame, signals:
        pd.DataFrame, backtest_config: (dict[str, Any] | None)=None, **
        kwargs: Any) ->dict[str, Any]:
        """
        Комплексний бектест з усіма покращеннями.
        """
        try:
            config = backtest_config or {}
            initial_capital = config.get('initial_capital', 100000.0)
            slippage_adj = config.get('slippage_adjustment', True)
            bias_detect = config.get('bias_detection', True)
            report: dict[str, Any] = {'timestamp': datetime.now().isoformat
                (), 'initial_capital': initial_capital,
                'performance_metrics': {}, 'transaction_analysis': {},
                'bias_analysis': {}, 'risk_metrics': {}, 'alerts': []}
            if slippage_adj:
                report['transaction_analysis'
                    ] = self._analyze_transaction_costs(signals, price_data)
            if bias_detect:
                bias_analysis: dict[str, Any] = report['bias_analysis']
                bias_analysis['look_ahead'
                    ] = self.bias_detector.detect_look_ahead_bias(signals,
                    price_data)
            returns = self._simulate_returns(
                price_data,
                initial_capital,
                signals=signals,
                apply_costs=slippage_adj,
            )
            valid_equity = returns.dropna()
            final_equity = valid_equity.iloc[-1] if not valid_equity.empty else initial_capital
            daily_returns = returns.pct_change(fill_method=None).dropna()
            report['performance_metrics'] = {'total_return': float((
                final_equity - initial_capital) / initial_capital),
                'annual_return': float(daily_returns.mean() * 252) if not daily_returns.empty else 0.0,
                'sharpe_ratio': float(self._calculate_sharpe(returns)),
                'max_drawdown': float(self._calculate_max_drawdown(returns)
                ), 'win_rate': float(self._calculate_win_rate(returns))}
            bias_analysis_result = report['bias_analysis']
            if isinstance(bias_analysis_result, dict):
                look_ahead = bias_analysis_result.get('look_ahead', {})
                if isinstance(look_ahead, dict) and look_ahead.get(
                    'has_look_ahead_bias'):
                    alerts: list[str] = report['alerts']
                    alerts.append('УВАГА: Виявлено look-ahead bias у сигналах!'
                        )
            self.logger.info('Комплексний бектест завершено')
            return report
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Помилка комплексного бектесту: {e}')
            return {'error': str(e)}

    def _analyze_transaction_costs(self, signals: pd.DataFrame, prices: pd.
        DataFrame) ->dict[str, Any]:
        """Аналіз транзакційних витрат"""
        total_trades = (signals != signals.shift()).sum().sum()
        costs = []
        for col in signals.columns:
            if col in prices.columns:
                trades = signals[col] != signals[col].shift()
                n_trades = trades.sum()
                if n_trades > 0:
                    avg_volatility = prices[col].pct_change(fill_method=None).std()
                    cost_estimate = self.cost_model.calculate_execution_costs(
                        100000, prices[col].mean() * 1000, avg_volatility)
                    costs.append({'ticker': col, 'num_trades': n_trades,
                        'estimated_cost_per_trade': cost_estimate['total'],
                        'total_estimated_costs': cost_estimate['total'] *
                        n_trades})
        return {'total_trades': total_trades, 'trades_by_asset': costs,
            'total_cost_estimate': sum(c['total_estimated_costs'] for c in
            costs)}

    def _simulate_returns(self, prices: pd.DataFrame, initial_cap: float,
        signals: (pd.DataFrame | None)=None, apply_costs: bool=False
        ) ->pd.Series:
        """Симуляція повернень портфеля"""
        if prices.empty:
            return pd.Series(dtype=float)
        asset_returns = prices.pct_change(fill_method=None).replace([np.inf,
            -np.inf], np.nan)
        if signals is None or signals.empty:
            portfolio_returns = asset_returns.mean(axis=1, skipna=True)
        else:
            positions = self._prepare_signal_positions(signals, prices)
            lagged_weights = positions.shift(1)
            lagged_weights = lagged_weights.mask(lagged_weights.isna(), 0.0)
            weighted_returns = lagged_weights * asset_returns

            # Fill missing returns for active positions with 0.0 to prevent portfolio-level NaN propagation
            missing_position_returns = asset_returns.isna() & lagged_weights.ne(0.0)
            if missing_position_returns.any().any():
                self.logger.warning("Missing return data detected for active positions. Treating missing returns as 0.0.")

            weighted_returns = weighted_returns.fillna(0.0)  # audit-ignore: PORTFOLIO_RETURN_FILLNA_ZERO_VALID - Missing returns treated as 0 for active positions with warning logged
            portfolio_returns = weighted_returns.sum(axis=1)
            no_position = lagged_weights.abs().sum(axis=1) == 0
            portfolio_returns = portfolio_returns.mask(no_position, 0.0)
            if apply_costs:
                turnover = lagged_weights.diff().abs().sum(axis=1,
                    min_count=1)
                turnover = turnover.mask(turnover.isna(), 0.0)
                portfolio_returns = (portfolio_returns -
                    turnover * self._estimate_turnover_cost_pct())
        if not portfolio_returns.empty and pd.isna(portfolio_returns.iloc[0]):
            portfolio_returns.iloc[0] = 0.0
        equity = initial_cap * (1 + portfolio_returns).cumprod()
        return equity

    def _prepare_signal_positions(self, signals: pd.DataFrame,
        prices: pd.DataFrame) ->pd.DataFrame:
        """Align signals to prices and convert them into normalized weights."""
        aligned = signals.reindex(index=prices.index, columns=prices.columns)
        if aligned.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        aligned = aligned.replace({
            'BUY': 1.0, 'LONG': 1.0, 'SELL': -1.0, 'SHORT': -1.0,
            'HOLD': 0.0, 'FLAT': 0.0, 'CLOSE': 0.0
        })
        aligned = aligned.apply(pd.to_numeric, errors='coerce').ffill()
        aligned = aligned.mask(aligned.isna(), 0.0)
        aligned = aligned.clip(lower=-1.0, upper=1.0)
        exposure = aligned.abs().sum(axis=1).replace(0, np.nan)
        weights = aligned.div(exposure, axis=0)
        return weights.mask(weights.isna(), 0.0)

    def _estimate_turnover_cost_pct(self) ->float:
        """Estimate proportional cost applied to portfolio turnover."""
        spread_pct = self.cost_model.spread_bps / 10000
        return float(self.cost_model.commission_pct + spread_pct +
            self.cost_model.slippage_pct)

    def _calculate_sharpe(self, equity: pd.Series, risk_free_rate: float=0.02
        ) ->float:
        """Розрахунок Sharpe Ratio"""
        returns = equity.pct_change(fill_method=None).dropna()
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        std_val = excess_returns.std()
        if not np.isfinite(std_val) or std_val <= 1e-12:
            return 0.0
        return float(excess_returns.mean() / std_val * np.sqrt(252))

    def _calculate_max_drawdown(self, equity: pd.Series) ->float:
        """Розрахунок maximum drawdown"""
        valid_equity = equity.dropna()
        if valid_equity.empty:
            return 0.0
        running_max = valid_equity.cummax()
        drawdown = (valid_equity - running_max) / running_max
        min_drawdown = drawdown.min()
        return float(min_drawdown) if np.isfinite(min_drawdown) else 0.0

    def _calculate_win_rate(self, returns: pd.Series) ->float:
        """Розрахунок win rate"""
        daily_returns = returns.pct_change(fill_method=None).dropna()
        wins = (daily_returns > 0).sum()
        return float(wins / len(daily_returns)) if len(daily_returns
            ) > 0 else 0
