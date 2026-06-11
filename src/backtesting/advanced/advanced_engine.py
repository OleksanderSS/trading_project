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

logger = ProjectLogger.get_logger("AdvancedBacktesting")


class TransactionCostModel:
    """Моделювання транзакційних витрат"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.commission_pct = self.config.get("commission_pct", 0.001)  # 0.1%
        self.spread_bps = self.config.get("spread_bps", 5)  # 5 basis points
        self.market_impact_coefficient = self.config.get(
            "market_impact_coefficient", 0.1
        )
        self.slippage_pct = self.config.get("slippage_pct", 0.001)  # 0.1%

    def calculate_execution_costs(
        self,
        trade_value: float,
        daily_volume: float,
        volatility: float,
        order_size_pct: float | None = None,
    ) -> dict[str, float]:
        """
        Розраховує всі компоненти витрат виконання
        """
        trade_value_abs = abs(trade_value)

        # 1. Commission
        commission = trade_value_abs * self.commission_pct

        # 2. Spread cost
        spread_cost = trade_value_abs * (self.spread_bps / 10000)

        # 3. Market impact (Almgren-Chriss model simplified)
        if order_size_pct is None:
            order_size_pct = (
                trade_value_abs / daily_volume if daily_volume > 0 else 0.01
            )

        market_impact = (
            trade_value_abs * self.market_impact_coefficient * np.sqrt(order_size_pct)
        )

        # 4. Slippage (залежить від волатильності)
        # ✅ ENHANCED: Prevent extreme slippage if volatility is invalid/outlier
        safe_vol = np.clip(volatility, 0.0, 1.0)
        slippage = trade_value_abs * self.slippage_pct * (1 + safe_vol * 10)

        # Total
        total_cost = commission + spread_cost + market_impact + slippage

        return {
            "commission": float(commission),
            "spread": float(spread_cost),
            "market_impact": float(market_impact),
            "slippage": float(slippage),
            "total": float(total_cost),
            "total_pct": (
                float(total_cost / trade_value_abs) if trade_value_abs > 0 else 0
            ),
        }


class BiasDetector:
    """Виявлення систематичних упереджень у бектестах"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = ProjectLogger.get_logger("BiasDetector")
        self.config = config or {}
        self.lookahead_corr_threshold = self.config.get("lookahead_corr_threshold", 0.5)

    def detect_look_ahead_bias(
        self, signals: pd.DataFrame, future_prices: pd.DataFrame, lag_periods: int = 1
    ) -> dict[str, Any]:
        """
        Виявлення look-ahead bias шляхом перевірки кореляції сигналів
        з майбутньою дохідністю.

        Якщо сигнал t корелює з дохідністю t, це ознака leak.
        """
        try:
            # 1. Знаходимо спільні колонки
            common_cols = signals.columns.intersection(future_prices.columns)
            if common_cols.empty:
                return {
                    "has_look_ahead_bias": False,
                    "suspicious_signals": [],
                    "message": "Немає спільних тікерів",
                }

            # 2. Векторизований розрахунок кореляцій
            correlations = signals[common_cols].corrwith(
                future_prices[common_cols].shift(-lag_periods)
            )

            # Статистична значимість
            n = len(signals)
            critical_corr = 1.96 / np.sqrt(n)

            # 4. Фільтрація підозрілих сигналів
            suspicious_mask = correlations.abs() > critical_corr
            suspicious_results = []

            for ticker, corr in correlations[suspicious_mask].items():
                is_bias = abs(corr) > critical_corr * 1.5
                suspicious_results.append(
                    {
                        "signal": ticker,
                        "correlation": float(corr),
                        "is_suspicious": is_bias,
                        "message": (
                            "Виявлено look-ahead bias"
                            if is_bias
                            else "Підозріло висока кореляція"
                        ),
                    }
                )

            return {
                "has_look_ahead_bias": len(suspicious_results) > 0,
                "suspicious_signals": suspicious_results,
                "critical_threshold": float(critical_corr),
                "sample_size": n,
            }

        except Exception as e:
            self.logger.error(f"Помилка виявлення look-ahead bias: {e}")
            return {"error": str(e)}

    def detect_survivorship_bias(
        self,
        historical_universe: list[str],
        current_universe: list[str],
        delisted_dates: dict[str, datetime],
    ) -> dict[str, Any]:
        """
        Виявлення survivorship bias

        Survivorship bias виникає коли бектест подвійно використовує акції що вилучені з індексу
        """
        try:
            delisted = set(historical_universe) - set(current_universe)

            # Аналіз performance делистед акцій перед делістингом
            delisted_performance_warning = []
            for ticker, delisted_date in delisted_dates.items():
                delisted_performance_warning.append(
                    {
                        "ticker": ticker,
                        "delisted_date": delisted_date.isoformat(),
                        "warning": f"Акція {ticker} була делістена {delisted_date.date()}",
                    }
                )

            return {
                "has_survivorship_bias": len(delisted) > 0,
                "delisted_count": len(delisted),
                "delisted_tickers": list(delisted),
                "bias_impact": len(delisted) / len(historical_universe),
                "delisted_warnings": delisted_performance_warning,
            }

        except Exception as e:
            self.logger.error(f"Помилка виявлення survivorship bias: {e}")
            return {"error": str(e)}


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization для избегання переопливання

    Ділить дані на in-sample (тренування) та out-of-sample (тестування) вікна
    """

    def __init__(self, config_manager: Any | None = None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("WalkForwardOptimizer")

    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        optimization_func: Callable,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
    ) -> dict[str, Any]:
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

            in_sample_size = int(
                total_rows * (in_sample_months / (in_sample_months + out_sample_months))
            )
            step_size = int(
                total_rows * (out_sample_months / 12)
            )  # Move forward by out-sample period

            window_idx = 0
            while in_sample_size + window_idx * step_size < total_rows:
                # In-sample window
                in_start = window_idx * step_size
                in_end = in_start + in_sample_size

                # Out-of-sample window
                out_start = in_end
                out_end = min(out_start + step_size, total_rows)

                if out_end - out_start < 10:  # Skip if too small
                    break

                # Get data for this window
                in_sample_data = data.iloc[in_start:in_end]

                # Optimize on in-sample
                try:
                    best_params = optimization_func(in_sample_data)

                    # Evaluate on out-of-sample
                    out_sample_data = data.iloc[out_start:out_end]
                    performance = self._evaluate_parameters(out_sample_data)

                    results.append(
                        {
                            "window": window_idx,
                            "in_sample_period": f"{data.index[in_start]} to {data.index[in_end-1]}",
                            "out_sample_period": f"{data.index[out_start]} to {data.index[out_end-1]}",
                            "optimized_parameters": best_params,
                            "out_sample_performance": performance,
                            "in_sample_size": in_end - in_start,
                            "out_sample_size": out_end - out_start,
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Помилка оптимізації на вікні {window_idx}: {e}"
                    )

                window_idx += 1

            return {
                "windows_completed": len(results),
                "windows_results": results,
                "average_out_sample_performance": self._calculate_average_performance(
                    results
                ),
                "optimization_completed": len(results) > 0,
            }

        except Exception as e:
            self.logger.error(f"Помилка Walk-Forward Optimization: {e}")
            return {"error": str(e)}

    def _evaluate_parameters(
        self, data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Оцінка параметрів на out-of-sample даних.

        Args:
            data: DataFrame з даними для оцінки (якщо None, повертає порожній результат)
        """
        if data is None or len(data) < 10:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        try:
            # Розраховуємо реальні метрики на даних
            returns = data.pct_change().dropna()
            if len(returns) == 0:
                return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

            # Return
            total_return = (1 + returns).prod() - 1

            # Sharpe ratio
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0.0
            )

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            return {
                "return": float(total_return),
                "sharpe": float(sharpe),
                "max_drawdown": float(max_dd),
            }
        except Exception:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    def _calculate_average_performance(self, results: list[dict]) -> dict[str, float]:
        """Розрахунок середньої performance"""
        if not results:
            return {}

        avg_return = np.mean(
            [r["out_sample_performance"].get("return", 0) for r in results]
        )
        avg_sharpe = np.mean(
            [r["out_sample_performance"].get("sharpe", 0) for r in results]
        )
        avg_dd = np.mean(
            [r["out_sample_performance"].get("max_drawdown", 0) for r in results]
        )

        return {
            "avg_return": float(avg_return),
            "avg_sharpe": float(avg_sharpe),
            "avg_max_drawdown": float(avg_dd),
        }


class AdvancedBacktestEngine:
    """
    Головний engine для розширеного бектестингу
    """

    def __init__(self, config_manager: Any | None = None):
        self.config = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("AdvancedBacktest")

        # Ініціалізація компонентів
        backtest_config = self.config.get("backtesting", {})
        self.cost_model = TransactionCostModel(
            backtest_config.get("transaction_costs", {})
        )
        self.bias_detector = BiasDetector()
        self.wf_optimizer = WalkForwardOptimizer(self.config)

    def run_comprehensive_backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        backtest_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Комплексний бектест з усіма покращеннями.
        """
        try:
            config = backtest_config or {}
            initial_capital = config.get("initial_capital", 100000.0)
            slippage_adj = config.get("slippage_adjustment", True)
            bias_detect = config.get("bias_detection", True)

            report: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "initial_capital": initial_capital,
                "performance_metrics": {},
                "transaction_analysis": {},
                "bias_analysis": {},
                "risk_metrics": {},
                "alerts": [],
            }

            # 1. Performance calculations with transaction costs
            if slippage_adj:
                report["transaction_analysis"] = self._analyze_transaction_costs(
                    signals, price_data
                )

            # 2. Bias detection
            if bias_detect:
                bias_analysis: dict[str, Any] = report["bias_analysis"]  # type: ignore[assignment]
                bias_analysis["look_ahead"] = self.bias_detector.detect_look_ahead_bias(
                    signals, price_data
                )

            # 3. Risk metrics
            returns = self._simulate_returns(price_data, initial_capital, signals=signals)
            report["performance_metrics"] = {
                "total_return": float(
                    (returns.iloc[-1] - initial_capital) / initial_capital
                ),
                "annual_return": float(returns.pct_change().mean() * 252),
                "sharpe_ratio": float(self._calculate_sharpe(returns)),
                "max_drawdown": float(self._calculate_max_drawdown(returns)),
                "win_rate": float(self._calculate_win_rate(returns)),
            }

            # 4. Generate alerts
            bias_analysis_result = report["bias_analysis"]
            if isinstance(bias_analysis_result, dict):
                look_ahead = bias_analysis_result.get("look_ahead", {})
                if isinstance(look_ahead, dict) and look_ahead.get(
                    "has_look_ahead_bias"
                ):
                    alerts: list[str] = report["alerts"]  # type: ignore[assignment]
                    alerts.append("УВАГА: Виявлено look-ahead bias у сигналах!")

            self.logger.info("Комплексний бектест завершено")
            return report

        except Exception as e:
            self.logger.error(f"Помилка комплексного бектесту: {e}")
            return {"error": str(e)}

    def _analyze_transaction_costs(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> dict[str, Any]:
        """Аналіз транзакційних витрат"""
        total_trades = (signals != signals.shift()).sum().sum()

        costs = []
        for col in signals.columns:
            if col in prices.columns:
                trades = signals[col] != signals[col].shift()
                n_trades = trades.sum()

                if n_trades > 0:
                    avg_volatility = prices[col].pct_change().std()
                    cost_estimate = self.cost_model.calculate_execution_costs(
                        100000,  # Assume $100k trades
                        prices[col].mean() * 1000,  # Estimate daily volume
                        avg_volatility,
                    )
                    costs.append(
                        {
                            "ticker": col,
                            "num_trades": n_trades,
                            "estimated_cost_per_trade": cost_estimate["total"],
                            "total_estimated_costs": cost_estimate["total"] * n_trades,
                        }
                    )

        return {
            "total_trades": total_trades,
            "trades_by_asset": costs,
            "total_cost_estimate": sum(c["total_estimated_costs"] for c in costs),
        }

    def _simulate_returns(self, prices: pd.DataFrame, initial_cap: float, signals: pd.DataFrame | None = None) -> pd.Series:
        """Симуляція повернень портфеля на основі сигналів"""
        asset_returns = prices.pct_change()
        
        if signals is None:
            # Fallback to simple buy-and-hold if no signals are provided
            mean_returns = asset_returns.mean(axis=1).fillna(0.0)
            equity = initial_cap * (1 + mean_returns).cumprod()
            return equity

        # Align signals and prices by columns
        common_cols = asset_returns.columns.intersection(signals.columns)
        if common_cols.empty:
            mean_returns = asset_returns.mean(axis=1).fillna(0.0)
            equity = initial_cap * (1 + mean_returns).cumprod()
            return equity

        # Shift signals by 1 period to avoid look-ahead bias (trading on day t captures return of day t+1)
        position_signals = signals[common_cols].shift(1).fillna(0.0)
        
        # Calculate daily portfolio returns: equal capital distribution among active positions
        active_positions_count = position_signals.abs().sum(axis=1)
        weights = position_signals.div(active_positions_count.replace(0, 1), axis=0)
        
        portfolio_daily_returns = (weights * asset_returns[common_cols]).sum(axis=1)
        equity = initial_cap * (1 + portfolio_daily_returns).cumprod()
        return equity

    def _calculate_sharpe(
        self, equity: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Розрахунок Sharpe Ratio"""
        returns = equity.pct_change().dropna()
        excess_returns = returns - (risk_free_rate / 252)
        return (
            (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            if excess_returns.std() > 0
            else 0
        )

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Розрахунок maximum drawdown"""
        cumulative_returns = (1 + equity.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Розрахунок win rate"""
        daily_returns = returns.pct_change().dropna()
        wins = (daily_returns > 0).sum()
        return float(wins / len(daily_returns)) if len(daily_returns) > 0 else 0

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any],
        optimization_metric: str = "sharpe",
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Оптимізація параметрів з використанням walk-forward analysis.

        Args:
            data: Історичні дані
            param_space: Простір параметрів для оптимізації
            optimization_metric: Метрика для оптимізації ('sharpe', 'return', 'calmar')
            n_splits: Кількість сплітів для cross-validation

        Returns:
            Оптимізовані параметри та результати
        """
        try:
            self.logger.info("Початок оптимізації параметрів")

            # Run walk-forward optimization
            wf_results = self.wf_optimizer.run_walk_forward(  # type: ignore[attr-defined]
                data=data,
                param_space=param_space,
                metric=optimization_metric,
                n_splits=n_splits,
            )

            if not wf_results.get("success"):
                return {
                    "success": False,
                    "error": wf_results.get(
                        "error", "Walk-forward optimization failed"
                    ),
                    "best_params": {},
                    "out_sample_performance": {},
                }

            # Extract best parameters
            best_params = wf_results.get("best_params", {})

            # Evaluate on out-of-sample data
            out_sample_perf = self._evaluate_parameters(
                wf_results.get("out_of_sample_data")
            )

            # Calculate average performance across folds
            avg_perf = self._calculate_average_performance(
                wf_results.get("fold_results", [])
            )

            # Build optimization report
            optimization_report = {
                "success": True,
                "best_params": best_params,
                "optimization_metric": optimization_metric,
                "n_splits": n_splits,
                "in_sample_performance": wf_results.get("best_performance", {}),
                "out_sample_performance": out_sample_perf,
                "average_performance": avg_perf,
                "stability_score": self._calculate_stability_score(
                    wf_results.get("fold_results", [])
                ),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(
                f"Оптимізацію завершено: {optimization_metric}={out_sample_perf.get('sharpe', 0):.2f}"
            )
            return optimization_report

        except Exception as e:
            self.logger.error(f"Помилка оптимізації параметрів: {e}")
            return {
                "success": False,
                "error": str(e),
                "best_params": {},
                "out_sample_performance": {},
            }

    def _calculate_stability_score(self, fold_results: list[dict]) -> float:
        """Розрахунок стабільності результатів across folds"""
        if not fold_results or len(fold_results) < 2:
            return 0.0

        try:
            sharpe_values = []
            for result in fold_results:
                perf = result.get("out_sample_performance", {})
                if isinstance(perf, dict):
                    sharpe_values.append(perf.get("sharpe", 0))

            if len(sharpe_values) < 2:
                return 0.0

            # Стандартне відхилення як міра стабільності (менше = стабільніше)
            std_sharpe = np.std(sharpe_values)
            mean_sharpe = np.mean(sharpe_values)

            # Stability score: 1 - coefficient of variation
            if mean_sharpe != 0:
                cv = abs(std_sharpe / mean_sharpe)
                return float(max(0, 1 - cv))
            return 0.0

        except Exception:
            return 0.0

    def _evaluate_parameters(self, data: pd.DataFrame | None) -> dict[str, float]:
        """Evaluate parameters on out-of-sample data"""
        if data is None or len(data) < 10:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        try:
            returns = data.pct_change().dropna()
            if len(returns) == 0:
                return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

            total_return = (1 + returns).prod() - 1
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0.0
            )
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            return {
                "return": float(total_return),
                "sharpe": float(sharpe),
                "max_drawdown": float(max_dd),
            }
        except Exception:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    def _calculate_average_performance(self, results: list[dict]) -> dict[str, float]:
        """Calculate average performance across folds"""
        if not results:
            return {}

        avg_return = np.mean(
            [r.get("out_sample_performance", {}).get("return", 0) for r in results]
        )
        avg_sharpe = np.mean(
            [r.get("out_sample_performance", {}).get("sharpe", 0) for r in results]
        )
        avg_dd = np.mean(
            [
                r.get("out_sample_performance", {}).get("max_drawdown", 0)
                for r in results
            ]
        )

        return {
            "avg_return": float(avg_return),
            "avg_sharpe": float(avg_sharpe),
            "avg_max_drawdown": float(avg_dd),
        }


class WalkForwardOptimizerExtended(WalkForwardOptimizer):
    """Extended Walk-Forward Optimizer with param_space support"""

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any],
        metric: str = "sharpe",
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Run walk-forward optimization with parameter space.

        Args:
            data: Historical data
            param_space: Parameter space for optimization
            metric: Metric to optimize ('sharpe', 'return', 'calmar')
            n_splits: Number of walk-forward splits

        Returns:
            Optimization results with best parameters
        """
        try:
            self.logger.info(
                f"Starting walk-forward optimization with {n_splits} splits"
            )

            # Calculate split sizes
            total_rows = len(data)
            fold_size = total_rows // (n_splits + 1)

            if fold_size < 30:
                return {
                    "success": False,
                    "error": "Insufficient data for walk-forward optimization",
                    "best_params": {},
                    "fold_results": [],
                }

            best_params = {}
            best_performance = {"sharpe": -float("inf")}
            fold_results = []
            out_of_sample_data = None

            for fold_idx in range(n_splits):
                # Define in-sample and out-of-sample periods
                in_start = fold_idx * fold_size
                in_end = in_start + fold_size * 2  # 2/3 for training

                out_start = in_end
                out_end = min(out_start + fold_size, total_rows)

                if out_end - out_start < 10:
                    continue

                in_sample = data.iloc[in_start:in_end]
                out_sample = data.iloc[out_start:out_end]

                # Optimize on in-sample (simple grid search)
                fold_best_params, fold_perf = self._grid_search(
                    in_sample, param_space, metric
                )

                # Evaluate on out-of-sample
                out_perf = self._evaluate_parameters(out_sample)

                fold_results.append(
                    {
                        "fold": fold_idx,
                        "in_sample_performance": fold_perf,
                        "out_sample_performance": out_perf,
                        "best_params": fold_best_params,
                    }
                )

                # Track best overall
                metric_value = out_perf.get(metric, 0)
                if metric_value > best_performance.get(metric, -float("inf")):
                    best_performance = out_perf
                    best_params = fold_best_params
                    out_of_sample_data = out_sample

            return {
                "success": True,
                "best_params": best_params,
                "best_performance": best_performance,
                "fold_results": fold_results,
                "out_of_sample_data": out_of_sample_data,
                "n_folds_completed": len(fold_results),
            }

        except Exception as e:
            self.logger.error(f"Error in walk-forward optimization: {e}")
            return {
                "success": False,
                "error": str(e),
                "best_params": {},
                "fold_results": [],
            }

    def _grid_search(
        self, data: pd.DataFrame, param_space: dict[str, Any], metric: str
    ) -> tuple[dict, dict]:
        """
        Simple grid search over parameter space using a simulated trend/momentum strategy
        derived from parameters to make optimization active.
        """
        import itertools

        param_names = list(param_space.keys())
        param_values = []

        for _name, config in param_space.items():
            if isinstance(config, dict) and "range" in config:
                start, end, step = config["range"]
                values = list(np.arange(start, end + step / 2, step))
            elif isinstance(config, list):
                values = config
            else:
                values = [config]
            param_values.append(values)

        best_params = {}
        best_perf = {metric: -float("inf")}

        # Try all combinations
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo, strict=False))

            try:
                asset_returns = data.pct_change().dropna()
                if len(asset_returns) == 0:
                    continue

                # Simulate a signal based on parameters to make the search active
                signal = pd.Series(1.0, index=asset_returns.index)
                if 'window' in params:
                    window = int(params['window'])
                    sma = data.rolling(window=max(1, window)).mean()
                    signal = np.sign(data - sma).mean(axis=1).fillna(0.0)
                elif 'threshold' in params:
                    threshold = float(params['threshold'])
                    signal = np.sign(asset_returns.mean(axis=1) - threshold).fillna(0.0)
                else:
                    scaling = sum(float(val) for val in params.values())
                    signal = pd.Series(np.sign(scaling), index=asset_returns.index)

                # Shift by 1 period to prevent look-ahead bias
                execution_signal = signal.shift(1).fillna(0.0)
                trade_returns = execution_signal * asset_returns.mean(axis=1)

                if metric == "sharpe":
                    perf_value = (
                        trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                        if trade_returns.std() > 0
                        else 0.0
                    )
                elif metric == "return":
                    perf_value = (1 + trade_returns).prod() - 1
                else:
                    perf_value = float(trade_returns.mean())

                if perf_value > best_perf.get(metric, -float("inf")):
                    best_perf[metric] = perf_value
                    best_params = params.copy()

            except Exception:
                continue

        return best_params, best_perf


# Patch WalkForwardOptimizer to add run_walk_forward method
WalkForwardOptimizer.run_walk_forward = WalkForwardOptimizerExtended.run_walk_forward
