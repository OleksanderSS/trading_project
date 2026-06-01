from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("RegimePatternAnalyzer")


class RegimePatternAnalyzer:
    """Аналізує патерни переможців серед моделей у різних режимах ринку."""

    def __init__(self, regime_types: Dict[str, Any]):
        self.regime_types = regime_types

    def calculate_model_consistency(self, regime_history: List[Dict[str, Any]]
        ) ->Dict[str, float]:
        """Розраховує оцінку стабільності моделей у певному режимі."""
        model_consistency = {}
        try:
            model_scores = defaultdict(list)
            for record in regime_history:
                model_name = record['model_name']
                score = record['score']
                model_scores[model_name].append(score)
            for model_name, scores in model_scores.items():
                if len(scores) >= 3:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    if mean_score > 0:
                        cv = std_score / mean_score
                        consistency_score = 1.0 / (1.0 + cv)
                    else:
                        consistency_score = 0.0
                    model_consistency[model_name] = float(consistency_score)
            return model_consistency
        except Exception as e:
            logger.error(f"Error calculating model consistency: {e}", exc_info=True)
            raise DataProcessingError(f"Model consistency calculation failed: {e}") from e

    def analyze_winner_patterns(self, regime_winners: Dict[str, Any],
        current_regime: str) ->Dict[str, Any]:
        """Аналізує патерни лідерів порівняно з очікуваннями."""
        winner_patterns = {'current_regime': current_regime,
            'expected_winners': [], 'actual_winners': [],
            'pattern_deviations': [], 'regime_specific_insights': {}}
        try:
            expected_winners = self.regime_types[current_regime][
                'typical_winners']
            winner_patterns['expected_winners'] = expected_winners
            actual_ranking = regime_winners.get('winner_ranking', [])
            actual_winners = [item['model_name'] for item in actual_ranking[:3]
                ]
            winner_patterns['actual_winners'] = actual_winners
            for i, expected_winner in enumerate(expected_winners):
                if i < len(actual_winners):
                    actual_winner = actual_winners[i]
                    if actual_winner != expected_winner:
                        winner_patterns['pattern_deviations'].append({
                            'position': i + 1, 'expected': expected_winner,
                            'actual': actual_winner, 'severity': self.
                            _calculate_deviation_severity(expected_winner,
                            actual_winner, i)})
            insights = self.generate_regime_insights(current_regime,
                expected_winners, actual_winners)
            winner_patterns['regime_specific_insights'] = insights
            return winner_patterns
        except Exception as e:
            logger.error(f"Error analyzing winner patterns: {e}", exc_info=True)
            raise DataProcessingError(f"Winner pattern analysis failed: {e}") from e

    def _calculate_deviation_severity(self, expected: str, actual: str,
        position: int) ->str:
        """Розраховує ступінь відхилення від очікуваного патерна."""
        position_weight = position / 3.0
        expected_for_normal = self.regime_types.get('normal', {}).get(
            'typical_winners', [])
        if actual in expected_for_normal:
            return 'low' if position_weight < 0.5 else 'medium'
        else:
            return 'high' if position_weight < 0.5 else 'critical'

    def generate_regime_insights(self, regime: str, expected_winners: List[
        str], actual_winners: List[str]) ->Dict[str, Any]:
        """Генерує інсайти щодо поточної відповідності режиму."""
        insights = {'regime_characteristics': self.regime_types[regime][
            'description'], 'alignment_score': 0.0, 'recommendations': []}
        try:
            if not expected_winners:
                return insights
            alignment_count = sum(1 for i, expected in enumerate(
                expected_winners) if i < len(actual_winners) and 
                actual_winners[i] == expected)
            insights['alignment_score'] = alignment_count / len(
                expected_winners)
            score = insights['alignment_score']
            if score >= 0.8:
                insights['recommendations'].append(
                    f'✅ Excellent alignment with {regime} regime expectations')
            elif score >= 0.5:
                insights['recommendations'].append(
                    f'⚠️ Moderate alignment with {regime} regime. Monitor for consistency'
                    )
            else:
                insights['recommendations'].append(
                    f'🚨 Poor alignment with {regime} regime. Consider model selection review'
                    )
            return insights
        except Exception as e:
            logger.error(f"Error generating regime insights: {e}", exc_info=True)
            raise DataProcessingError(f"Regime insights generation failed: {e}") from e
