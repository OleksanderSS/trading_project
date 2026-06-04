from typing import Any

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class RegimeMetrics:
    """Розрахунок метрик продуктивності моделей у контексті режимів ринку."""

    @staticmethod
    def calculate_performance_score(metrics: dict[str, float]) ->float:
        """Розраховує уніфікований показник продуктивності."""
        try:
            metric_weights = {'accuracy': 0.3, 'precision': 0.2, 'recall':
                0.2, 'f1': 0.2, 'r2': 0.1, 'mse': -0.1, 'mae': -0.1, 'rmse':
                -0.1}
            score = 0.0
            total_weight = 0.0
            for metric_name, value in metrics.items():
                if metric_name in metric_weights:
                    weight = metric_weights[metric_name]
                    if metric_name in ['mse', 'mae', 'rmse']:
                        normalized_value = 1.0 / (1.0 + value) if value > 0 else 1.0
                    else:
                        normalized_value = max(0.0, min(1.0, value))
                    score += weight * normalized_value
                    total_weight += abs(weight)
            return score / total_weight if total_weight > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}", exc_info=True)
            raise DataProcessingError(f"Performance score calculation failed: {e}") from e

    @staticmethod
    def calculate_score_gap(ranked_models: list[tuple[str, dict[str, Any]]]
        ) ->float:
        """Розраховує розрив між лідером та наступною моделлю."""
        if len(ranked_models) < 2:
            return 0.0
        winner_score = ranked_models[0][1].get('performance_score', 0.0)
        runner_up_score = ranked_models[1][1].get('performance_score', 0.0)
        return float(winner_score - runner_up_score)
