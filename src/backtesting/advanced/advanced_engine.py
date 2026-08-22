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

from src.analytics.detectors.bias_detector import BiasDetector
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


# BiasDetector is imported from src.analytics.detectors.bias_detector to avoid duplication


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization для избегання переопливання

    Ділить дані на in-sample (тренування) та out-of-sample (тестування) вікна
    """

    def __init__(self, config_manager: (Any | None)=None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger('WalkForwardOptimizer')

    @staticmethod
    def _span_months(data: pd.DataFrame) -> float:
        idx = data.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
            return float('nan')
        return (idx[-1] - idx[0]).days / 30.44

    @staticmethod
    def _walk_forward_windows(data: pd.DataFrame, in_sample_months: int,
                              out_sample_months: int,
                              embargo_bars: int = 0) -> list[tuple[int, int, int, int]]:
        """Row boundaries for each fold, measured in CALENDAR months.

        Returns [(in_start, in_end, out_start, out_end), ...] as positional
        slices, so the caller keeps using .iloc and nothing else changes.

        A non-datetime index cannot be cut into months, and guessing a
        bars-per-month figure is how the previous version produced windows that
        silently changed size with the amount of history. It returns no windows
        instead, and the caller reports why.

        `embargo_bars` is the gap between the two windows, and it defaults to
        zero only so existing callers keep their behaviour until they say what
        their horizon is.

        Why it has to exist: the out-of-sample window began at `in_end`, the
        very row training ended on. A label attached to the last training bar
        is computed from prices that fall INSIDE the validation window, so an
        optimiser tuned on these folds is rewarded for exploiting the overlap.
        With a 5-bar target the last five training labels are contaminated;
        with `target_hourly_volume_spike_1h`, twenty-three of them.

        This is the second walk-forward implementation in the repository. The
        one in `src/pipeline/stages/modeling` purges, and raises its gap to the
        target horizon automatically. This one did not, which is the shape this
        codebase keeps repeating: a fix landing in one of two copies.

        Pass `embargo_bars` >= the target's lookahead. `_get_target_horizon_rows`
        in walk_forward_validation computes it from a target name.
        """
        idx = data.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
            return []
        if not idx.is_monotonic_increasing:
            idx = idx.sort_values()

        windows: list[tuple[int, int, int, int]] = []
        in_start_time = idx[0]
        last = idx[-1]
        while True:
            in_end_time = in_start_time + pd.DateOffset(months=in_sample_months)
            out_end_time = in_end_time + pd.DateOffset(months=out_sample_months)
            if in_end_time > last:
                break
            in_start = int(idx.searchsorted(in_start_time, side='left'))
            in_end = int(idx.searchsorted(in_end_time, side='left'))
            out_start = in_end + max(0, int(embargo_bars))
            out_end = int(idx.searchsorted(min(out_end_time, last), side='right'))
            if in_end <= in_start or out_end <= out_start:
                break
            windows.append((in_start, in_end, out_start, out_end))
            if out_end_time >= last:
                break
            # Step forward by the OUT-OF-SAMPLE length: each bar is tested
            # exactly once, which is what makes the folds independent.
            in_start_time = in_start_time + pd.DateOffset(months=out_sample_months)
        return windows

    def walk_forward_optimization(self, data: pd.DataFrame,
        optimization_func: Callable, in_sample_months: int=12,
        out_sample_months: int=3, embargo_bars: int=0) ->dict[str, Any]:
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

            # Window sizes come from the DATA'S OWN CALENDAR, not from a
            # fraction of however much history happened to be passed in.
            #
            # They used to be:
            #     in_sample_size = total_rows * (in / (in + out))
            #     step_size      = total_rows * (out / 12)
            #
            # The first is 80% of the input for the default 12/3 split whatever
            # "12 months" is supposed to mean, and the second divides by 12 on
            # the assumption that `total_rows` is always exactly one year. Hand
            # this 30 years of daily bars and a 12/3 config and it trains on
            # TWENTY-FOUR YEARS and steps forward by SEVEN AND A HALF, which is
            # not a walk-forward at all -- and the window sizes change whenever
            # the caller passes a different amount of history, so no two runs
            # are comparable.
            #
            # Slicing by calendar offset is also timeframe-independent: the
            # same call is correct for daily, hourly and 15-minute bars, which
            # a bar count can never be.
            # The gap between training and validation. Zero is the old
            # behaviour and it leaks: the label on the last training bar is
            # computed from prices inside the validation window. Callers with a
            # forward-looking target must pass their horizon.
            if not embargo_bars:
                logger.warning(
                    "Walk-forward optimisation running with NO embargo gap. "
                    "Validation begins on the bar training ended on, so labels "
                    "at the boundary are computed from prices inside the "
                    "validation window. Pass embargo_bars >= the target's "
                    "lookahead horizon."
                )
            windows = self._walk_forward_windows(
                data, in_sample_months, out_sample_months, embargo_bars
            )
            if not windows:
                return {
                    'status': 'insufficient_history',
                    'reason': (
                        f'{total_rows} rows spanning '
                        f'{self._span_months(data):.1f} months cannot hold one '
                        f'{in_sample_months}+{out_sample_months} month window'
                    ),
                    'windows': 0,
                }
            for window_idx, (in_start, in_end, out_start, out_end) in enumerate(windows):
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
            from src.algorithms.metrics_mixin import _infer_periods_per_year
            ppy = _infer_periods_per_year(returns)
            sharpe = (
                float(returns.mean() / std_val * np.sqrt(ppy))
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

from src.algorithms.metrics_mixin import PerformanceMetricsMixin


class AdvancedBacktestEngine(PerformanceMetricsMixin):
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
            returns_series = self._simulate_returns(
                price_data,
                initial_capital,
                signals=signals,
                apply_costs=slippage_adj,
            )
            # Calculate returns for metrics
            daily_returns = returns_series.pct_change(fill_method=None).dropna()

            valid_equity = returns_series.dropna()
            final_equity = valid_equity.iloc[-1] if not valid_equity.empty else initial_capital

            from src.algorithms.metrics_mixin import _infer_periods_per_year as _ppy
            _ppy_val = _ppy(daily_returns) if not daily_returns.empty else 252
            from src.metrics.financial.financial_metrics_library import get_risk_free_rate
            _rf = get_risk_free_rate()
            report['performance_metrics'] = {'total_return': float((
                final_equity - initial_capital) / initial_capital),
                'annual_return': float(daily_returns.mean() * _ppy_val) if not daily_returns.empty else 0.0,
                'sharpe_ratio': float(self._calculate_sharpe(daily_returns, risk_free_rate=_rf)),
                'max_drawdown': float(self._calculate_max_drawdown(returns_series)
                # _calculate_win_rate takes an EQUITY CURVE and differences it
                # itself. It was handed `daily_returns`, which is already
                # returns_series.pct_change() -- so the published win rate was
                # the share of bars on which the RETURN rose, not the share of
                # bars that made money. Those coincide only by accident.
                ), 'win_rate': float(self._calculate_win_rate(returns_series)),
                # The Sharpe above is meaningless next to another Sharpe unless
                # the convention travels with it.
                'risk_free_rate_used': float(_rf),
                'periods_per_year_used': int(_ppy_val)}
            # Expose the real simulated equity curve so downstream
            # consumers (BacktestAnalyzer._normalize_backtest_results)
            # don't have to fabricate a fake straight-line approximation
            # between initial/final capital - that discards the actual
            # daily-return path this method already computed, silently
            # zeroing out real drawdown/volatility in any derived metrics.
            report['portfolio_history'] = pd.DataFrame(
                {'total_value': returns_series}
            )
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
        signals: (pd.DataFrame | None)=None, apply_costs: bool=False,
        fillna_policy: str='zero') ->pd.Series:
        """
        Симуляція повернень портфеля.
        
        Args:
            prices: Price data for assets
            initial_cap: Initial capital
            signals: Trading signals (optional)
            apply_costs: Whether to apply transaction costs
            fillna_policy: Policy for handling missing returns for active positions:
                - 'zero': Fill with 0.0 (default, understates risk)
                - 'ffill': Forward fill from last known value
                - 'drop': Drop periods with missing returns
                - 'warn_only': Keep NaN and warn (most conservative)
        """
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

            # Handle missing returns for active positions based on policy
            missing_position_returns = asset_returns.isna() & lagged_weights.ne(0.0)
            if missing_position_returns.any().any():
                self.logger.warning(f"Missing return data detected for active positions. Using fillna_policy: '{fillna_policy}'")

            if fillna_policy == 'zero':
                weighted_returns = weighted_returns.fillna(0.0)  # audit-ignore: PORTFOLIO_RETURN_FILLNA_ZERO - User-configurable policy
            elif fillna_policy == 'ffill':
                weighted_returns = weighted_returns.ffill()
            elif fillna_policy == 'drop':
                # Drop rows with missing returns for active positions
                weighted_returns = weighted_returns.dropna()
                lagged_weights = lagged_weights.reindex(weighted_returns.index)
            elif fillna_policy == 'warn_only':
                # Keep NaN, will propagate to portfolio returns
                pass
            else:
                self.logger.warning(f"Unknown fillna_policy '{fillna_policy}', defaulting to 'zero'")
                weighted_returns = weighted_returns.fillna(0.0)

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

    def _calculate_win_rate(self, equity_curve: pd.Series) ->float:
        """Розрахунок win rate (частка днів з позитивними поверненнями)"""
        daily_returns = equity_curve.pct_change(fill_method=None).dropna()
        wins = (daily_returns > 0).sum()
        return float(wins / len(daily_returns)) if len(daily_returns
            ) > 0 else 0
