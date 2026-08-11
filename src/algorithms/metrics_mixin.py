from typing import Any

import numpy as np
import pandas as pd

# Canonical home is now FinancialMetricsLibrary — re-exported here under the
# original name since src/backtesting/advanced/advanced_engine.py,
# src/pipeline/stages/evaluation/metrics_calculator.py, and
# src/analytics/analyzers/performance_attribution_analyzer.py all import
# `_infer_periods_per_year` directly from this module.
from src.metrics.financial.financial_metrics_library import (
    FinancialMetricsLibrary,
    get_risk_free_rate,
    infer_periods_per_year as _infer_periods_per_year,
)


class PerformanceMetricsMixin:
    """Mixin для спільних метрик продуктивності та стабільності."""

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: float | None = None,
        periods_per_year: int | None = None,
    ) -> float:
        """Delegates to FinancialMetricsLibrary.calculate_sharpe_ratio (the
        canonical implementation).

        The hardcoded 0.02 default this used to carry was deliberately left
        in place by an earlier pass, on the grounds that unifying it would
        change every backtest metric the mixin produces. It changed them the
        other way instead: Stage 7 published this mixin's Sharpe as
        `backtest_stats.sharpe_ratio` (0.7023) beside the evaluation
        calculator's Sharpe as `metrics.sharpe_ratio` (1.0212) for ONE equity
        curve, and the gap reproduces exactly as (0.02/252)/std*sqrt(252).
        Two answers to one question is worse than one answer under a stated
        convention, so the rate now comes from get_risk_free_rate() and every
        producer records which rate it used.

        Unchanged: cadence-aware periods_per_year auto-inference when not
        given, and 0.0 (not NaN) on insufficient data or zero/non-finite
        excess-return std.
        """
        rate = get_risk_free_rate() if risk_free_rate is None else risk_free_rate
        return FinancialMetricsLibrary.calculate_sharpe_ratio(
            returns,
            risk_free_rate=rate,
            trading_days_per_year=periods_per_year,
            on_error=0.0,
        )

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return float(drawdown.min())

    def _calculate_stability_score(self, fold_results: list[dict[str, Any]]) -> float:
        """Розрахунок стабільності результатів по фолдам."""
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
            finite_sharpes = [float(value) for value in sharpe_values if np.isfinite(value)]
            if len(finite_sharpes) < 2:
                return 0.0
            std_sharpe = np.std(finite_sharpes)
            mean_sharpe = np.mean(finite_sharpes)
            if abs(mean_sharpe) > 1e-12:
                cv = abs(std_sharpe / mean_sharpe)
                return float(max(0, 1 - cv))
            return 0.0
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            return 0.0

    def _calculate_average_performance(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Розрахунок середніх метрик по фолдам."""
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
