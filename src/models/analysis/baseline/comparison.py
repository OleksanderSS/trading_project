import logging
from typing import Any

logger = logging.getLogger(__name__)

class BaselineComparisonEngine:
    """Порівнює продуктивність складної моделі з базовими."""

    def __init__(self, dominance_threshold: float = 0.05):
        self.dominance_threshold = dominance_threshold
        self.logger = logger

    def compare(self, complex_metrics: dict[str, float], baseline_results: dict[str, Any]) -> dict[str, Any]:
        """Порівнює результати складної моделі з базовими."""
        comparison_results = {
            'dominant_baselines': [],
            'performance_comparison': {},
            'dominance_detected': False
        }

        try:
            if not complex_metrics:
                self.logger.warning("No complex model metrics available for comparison")
                return comparison_results

            for baseline_name, baseline_result in baseline_results.items():
                if baseline_result.get('status') == 'error' or 'metrics' not in baseline_result:
                    continue

                baseline_metrics = baseline_result['metrics']
                performance_diff = self._calculate_performance_difference(complex_metrics, baseline_metrics)
                dominance_info = self._check_baseline_dominance(performance_diff, baseline_result.get('complexity_score', 1))

                comparison_results['performance_comparison'][baseline_name] = {
                    'performance_difference': performance_diff,
                    'dominance_info': dominance_info
                }

                if dominance_info['is_dominant']:
                    comparison_results['dominant_baselines'].append({
                        'baseline_name': baseline_name,
                        'dominance_strength': dominance_info['strength'],
                        'complexity_savings': dominance_info['complexity_savings']
                    })
                    comparison_results['dominance_detected'] = True

            comparison_results['dominant_baselines'].sort(key=lambda x: x['dominance_strength'], reverse=True)
            return comparison_results
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error comparing with baselines: {e}")
            raise

    def _calculate_performance_difference(self, complex_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
        differences = {}
        for metric in ['mse', 'mae', 'r2']:
            if metric in complex_metrics and metric in baseline_metrics:
                complex_val = complex_metrics[metric]
                baseline_val = baseline_metrics[metric]
                differences[metric] = (baseline_val - complex_val) if metric == 'r2' else (complex_val - baseline_val)
        return differences

    def _check_baseline_dominance(self, performance_diff: dict[str, float], baseline_complexity: float) -> dict[str, Any]:
        dominance_info = {'is_dominant': False, 'strength': 0.0, 'complexity_savings': 0.0}
        dominant_count = 0
        total_advantage = 0.0
        for diff in performance_diff.values():
            if diff > self.dominance_threshold:
                dominant_count += 1
                total_advantage += diff

        if dominant_count > 0:
            dominance_info['strength'] = total_advantage / dominant_count
            dominance_info['is_dominant'] = True
            complex_complexity = 10
            dominance_info['complexity_savings'] = (complex_complexity - baseline_complexity) / complex_complexity
        return dominance_info
