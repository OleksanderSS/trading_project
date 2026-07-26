import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.archive.algorithms.bias_detector import BiasDetector
from src.archive.algorithms.transaction_cost_model import TransactionCostModel
from src.archive.algorithms.walk_forward_optimizer import WalkForwardOptimizer
from src.algorithms.metrics_mixin import PerformanceMetricsMixin
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('AdvancedBacktesting')


class AdvancedBacktestEngine(PerformanceMetricsMixin):
    """Розширена система бектестингу з урахуванням витрат та перевіркою на витоки"""

    def __init__(self, config_manager: Any=None):
        self.config_manager = config_manager
        self.logger = logger
        self.cost_model = TransactionCostModel()
        self.bias_detector = BiasDetector()
        self.wf_optimizer = WalkForwardOptimizer(self.config_manager)

    def run_comprehensive_backtest(self, price_data: pd.DataFrame, signals:
        pd.DataFrame, initial_capital: float=100000.0) ->dict[str, Any]:
        """Запускає повний цикл бектестингу з урахуванням усіх факторів."""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'Starting comprehensive backtest on {len(price_data)} periods'
                )
        bias_results = self.bias_detector.detect_look_ahead_bias(signals,
            price_data)
        if bias_results.get('lookahead_bias_detected'):
            self.logger.warning('🚨 CRITICAL: Look-ahead bias detected!')
        raw_equity = self._simulate_returns(price_data, initial_capital,
            signals)
        costs = self._analyze_transaction_costs(signals, price_data)
        cumulative_costs = costs.cumsum().reindex(raw_equity.index,
            method='ffill')
        cumulative_costs = cumulative_costs.mask(cumulative_costs.isna(), 0.0)
        net_equity = raw_equity - cumulative_costs
        return {'equity_curve': net_equity, 'sharpe_ratio': self.
            _calculate_sharpe(net_equity), 'max_drawdown': self.
            _calculate_max_drawdown(net_equity), 'win_rate': self.
            _calculate_win_rate(net_equity.pct_change(fill_method=None)), 'bias_results':
            bias_results}

    def _analyze_transaction_costs(self, signals: pd.DataFrame, prices: pd.
        DataFrame) ->pd.Series:
        """Розраховує витрати для кожної угоди"""
        costs = pd.Series(0.0, index=signals.index)
        for t in signals.index:
            if abs(signals.loc[t]).sum() > 0:
                trade_value = signals.loc[t].sum() * prices.loc[t].mean()
                costs.loc[t] = self.cost_model.calculate_execution_costs(
                    trade_value)
        return costs

    def _simulate_returns(self, prices: pd.DataFrame, initial_cap: float,
        signals: pd.DataFrame) ->pd.Series:
        """Симуляція капіталу"""
        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf],
            np.nan)
        lagged_signals = signals.shift(1).reindex(returns.index)
        lagged_signals = lagged_signals.mask(lagged_signals.isna(), 0.0)
        weighted_returns = lagged_signals * returns
        portfolio_returns = weighted_returns.sum(axis=1, min_count=1)
        no_position = lagged_signals.abs().sum(axis=1) == 0
        portfolio_returns = portfolio_returns.mask(no_position, 0.0)
        missing_position_returns = returns.isna() & lagged_signals.ne(0.0)
        portfolio_returns = portfolio_returns.mask(missing_position_returns.any(axis=1))
        if not portfolio_returns.empty and pd.isna(portfolio_returns.iloc[0]):
            portfolio_returns.iloc[0] = 0.0
        equity = initial_cap * (1 + portfolio_returns).cumprod()
        return equity

    def _calculate_sharpe(self, equity: pd.Series, risk_free_rate: float=0.02
        ) ->float:
        returns = equity.pct_change(fill_method=None).fillna(0)
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        std_val = excess_returns.std()
        
        # Безпечний розрахунок
        if not np.isfinite(std_val) or std_val <= 1e-12:
            return 0.0
        
        sharpe = (np.sqrt(252) * excess_returns.mean()) / std_val
        return float(sharpe) if np.isfinite(sharpe) else 0.0

    def _calculate_max_drawdown(self, equity: pd.Series) ->float:
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    def _calculate_win_rate(self, returns: pd.Series) ->float:
        pos_returns = returns[returns > 0]
        return len(pos_returns) / len(returns) if len(returns) > 0 else 0.0

    def optimize_parameters(self, data: pd.DataFrame, param_space: dict[str,
        Any], optimization_metric: str='sharpe', n_splits: int=5) ->dict[
        str, Any]:
        """Оптимізація параметрів з використанням walk-forward analysis."""
        try:
            self.logger.info('Початок оптимізації параметрів')
            wf_results = self.wf_optimizer.run_walk_forward(data=data,
                param_space=param_space, metric=optimization_metric,
                n_splits=n_splits)
            if not wf_results.get('success'):
                return {'success': False, 'error': wf_results.get('error',
                    'Walk-forward optimization failed'), 'best_params': {},
                    'out_sample_performance': {}}
            best_params = wf_results.get('best_params', {})
            out_sample_perf = self._evaluate_parameters(wf_results.get(
                'out_of_sample_data'))
            avg_perf = self._calculate_average_performance(wf_results.get(
                'fold_results', []))
            optimization_report = {'success': True, 'best_params':
                best_params, 'optimization_metric': optimization_metric,
                'n_splits': n_splits, 'in_sample_performance': wf_results.
                get('best_performance', {}), 'out_sample_performance':
                out_sample_perf, 'average_performance': avg_perf,
                'stability_score': self._calculate_stability_score(
                wf_results.get('fold_results', [])), 'timestamp': datetime.
                now().isoformat()}
            self.logger.info(
                f"Оптимізацію завершено: {optimization_metric}={out_sample_perf.get('sharpe', 0):.2f}"
                )
            return optimization_report
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise DataProcessingError('Parameter optimization failed') from e

    def _calculate_stability_score(self, fold_results: list[dict]) ->float:
        """Розрахунок стабільності результатів across folds"""
        if not fold_results or len(fold_results) < 2:
            return 0.0
        try:
            sharpe_values = []
            for result in fold_results:
                perf = result.get('out_sample_performance', {})
                if isinstance(perf, dict):
                    sharpe_values.append(perf.get('sharpe', 0))
            if len(sharpe_values) < 2:
                return 0.0
            finite_sharpes = [float(value) for value in sharpe_values if
                np.isfinite(value)]
            if len(finite_sharpes) < 2:
                return 0.0
            std_sharpe = np.std(finite_sharpes)
            mean_sharpe = np.mean(finite_sharpes)
            if abs(mean_sharpe) > 1e-12:
                cv = abs(std_sharpe / mean_sharpe)
                return float(max(0, 1 - cv))
            return 0.0
        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
            self.logger.error(
                f'Помилка розрахунку стабільності: {e}',
                exc_info=True)
            return 0.0

    def _evaluate_parameters(self, data: pd.DataFrame | None) ->dict[str,
        float]:
        """Оцінка параметрів на OOS даних"""
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
            return {'return': float(total_return), 'sharpe': float(sharpe),
                'max_drawdown': float(max_dd)}
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self.logger.error(f'Помилка при оцінці параметрів: {e}',
                exc_info=True)
            return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}

    def _calculate_average_performance(self, results: list[dict]) ->dict[
        str, float]:
        """Розрахунок середніх метрик по фолдам"""
        if not results:
            return {'mean_sharpe': 0.0, 'mean_return': 0.0}
        sharpes = []
        returns = []
        for r in results:
            perf = r.get('out_sample_performance', {})
            if isinstance(perf, dict):
                sharpes.append(perf.get('sharpe', 0))
                returns.append(perf.get('return', 0))
        return {'mean_sharpe': float(np.mean(sharpes)) if sharpes else 0.0,
            'mean_return': float(np.mean(returns)) if returns else 0.0}


