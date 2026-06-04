from typing import Dict, List, Any
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('RegimeRecommendations')


class RegimeRecommendationEngine:
    """Генерує рекомендації на основі аналізу режимів ринку."""

    def __init__(self, regime_types: Dict[str, Any]):
        self.regime_types = regime_types
        self.logger = logger

    def generate_regime_recommendations(self, current_regime: str,
        consistency_metrics: Dict[str, Any], winner_patterns: Dict[str, Any]
        ) ->List[str]:
        """Генерує рекомендації на основі аналізу режимів."""
        recommendations = []
        try:
            overall_consistency = consistency_metrics.get('overall_consistency', 0.0)
            if overall_consistency < 0.5:
                recommendations.append(
                    f'⚠️ Low model consistency ({overall_consistency:.2f}). Consider ensemble methods for stability.'
                    )
            elif overall_consistency < 0.7:
                recommendations.append(
                    f'📊 Moderate model consistency ({overall_consistency:.2f}). Monitor for performance degradation.'
                    )
            else:
                recommendations.append(
                    f'✅ High model consistency ({overall_consistency:.2f}). Current model selection strategy is effective.'
                    )
            pattern_deviations = winner_patterns.get('pattern_deviations', [])
            if len(pattern_deviations) > 2:
                recommendations.append(
                    f'🚨 High pattern deviations ({len(pattern_deviations)}). Review model selection for current regime.'
                    )
            regime_insights = winner_patterns.get('regime_specific_insights', {})
            regime_recommendations = regime_insights.get('recommendations', [])
            recommendations.extend(regime_recommendations)
            return recommendations
        except Exception as e:
            self.logger.error(f'Error generating recommendations: {e}')
            return []
