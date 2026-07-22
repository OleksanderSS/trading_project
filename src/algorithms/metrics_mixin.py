from typing import Any

import numpy as np
import pandas as pd


def _infer_periods_per_year(returns: pd.Series) -> int:
    """
    Infer annualisation factor from the DatetimeIndex of *returns*.

    Falls back to 252 (daily) when the index is not a DatetimeIndex or the
    median gap cannot be determined reliably.

    Typical values:
        252  – daily  (1 bar ≈ 1 trading day)
        52   – weekly (1 bar ≈ 1 week)
        12   – monthly
        1440 – 1-minute bars  (252 * 390 ≈ ~98 280, but conventionally capped)
        390  – 1-minute intra-day per day * 252 trading days / 252 = 390

    We map calendar-day median gaps to the conventional annualisation buckets
    used in finance.
    """
    if not isinstance(returns.index, pd.DatetimeIndex) or len(returns) < 2:
        return 252

    gaps = returns.index.to_series().diff().dropna()
    if gaps.empty:
        return 252

    median_seconds = gaps.median().total_seconds()
    # Map to conventional annualisation buckets
    if median_seconds <= 90:           # ≤ 1.5 min → 1-minute bars
        return 252 * 390               # ~98 280  (1-min bars per trading year)
    if median_seconds <= 1200:         # ≤ 20 min → 15-minute bars
        return 252 * 26                # ~6 552
    if median_seconds <= 5400:         # ≤ 90 min → 1-hour bars
        return 252 * 7                 # ~1 764   (≈ 6-7 h per trading day)
    if median_seconds <= 100_000:      # ≤ ~1.15 days → daily
        return 252
    if median_seconds <= 800_000:      # ≤ ~9 days → weekly
        return 52
    if median_seconds <= 2_800_000:    # ≤ ~32 days → monthly
        return 12
    return 4                           # quarterly


class PerformanceMetricsMixin:
    """Mixin для спільних метрик продуктивності та стабільності."""

    def _calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int | None = None,
    ) -> float:
        if len(returns) < 2:
            return 0.0
        ppy = periods_per_year if periods_per_year is not None else _infer_periods_per_year(returns)
        excess_returns = returns - risk_free_rate / ppy
        std_val = excess_returns.std()
        if not np.isfinite(std_val) or std_val <= 1e-12:
            return 0.0
        sharpe = (np.sqrt(ppy) * excess_returns.mean()) / std_val
        return float(sharpe) if np.isfinite(sharpe) else 0.0

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
