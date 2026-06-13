from typing import Any
import numpy as np
import pandas as pd

class PerformanceMetricsMixin:
    """Mixin для спільних метрик продуктивності та стабільності."""

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        std_val = excess_returns.std()
        
        # Безпечний розрахунок
        if not np.isfinite(std_val) or std_val <= 1e-12:
            return 0.0
        
        sharpe = (np.sqrt(252) * excess_returns.mean()) / std_val
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
