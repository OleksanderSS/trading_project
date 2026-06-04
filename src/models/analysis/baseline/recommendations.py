import logging
from typing import Any

logger = logging.getLogger(__name__)

class BaselineRecommendationEngine:
    """Генерує рекомендації щодо спрощення моделі."""

    def __init__(self, complexity_penalty: float = 0.02, baseline_models: dict[str, Any] = None):
        self.complexity_penalty = complexity_penalty
        self.baseline_models = baseline_models or {}
        self.logger = logger

    def perform_cost_benefit_analysis(self, complex_model_results: dict[str, Any], performance_comparison: dict[str, Any]) -> dict[str, Any]:
        """Виконує аналіз витрат і вигод."""
        cost_benefit = {
            'complexity_cost': 0.0,
            'performance_benefit': 0.0,
            'net_benefit': 0.0,
            'recommendation': 'keep_complex'
        }
        try:
            complex_complexity = 10
            cost_benefit['complexity_cost'] = complex_complexity * self.complexity_penalty

            if not performance_comparison.get('dominance_detected', False):
                cost_benefit['performance_benefit'] = 0.1
            else:
                cost_benefit['performance_benefit'] = 0.0

            cost_benefit['net_benefit'] = cost_benefit['performance_benefit'] - cost_benefit['complexity_cost']

            if cost_benefit['net_benefit'] < 0:
                cost_benefit['recommendation'] = 'simplify'
            elif cost_benefit['net_benefit'] < 0.02:
                cost_benefit['recommendation'] = 'consider_simplification'

            return cost_benefit
        except Exception as e:
            self.logger.error(f"Error in cost-benefit analysis: {e}")
            raise

    def generate_simplification_recommendations(self, dominance_analysis: dict[str, Any], cost_benefit: dict[str, Any]) -> list[str]:
        """Генерує рекомендації щодо спрощення моделі."""
        recommendations = []
        try:
            if dominance_analysis.get('dominance_detected', False):
                dominant_baselines = dominance_analysis.get('dominant_baselines', [])
                for baseline_info in dominant_baselines[:2]:
                    baseline_name = baseline_info['baseline_name']
                    strength = baseline_info['dominance_strength']
                    savings = baseline_info['complexity_savings']
                    baseline_config = self.baseline_models.get(baseline_name, {})
                    description = baseline_config.get('description', baseline_name)
                    recommendations.append(f"🎯 Consider {description} - outperforms complex model by {strength:.3f} with {savings:.1%} complexity reduction")

            cost_benefit_rec = cost_benefit.get('recommendation', 'keep_complex')
            if cost_benefit_rec == 'simplify':
                recommendations.append("⚠️ High complexity cost detected. Model simplification recommended.")
            elif cost_benefit_rec == 'consider_simplification':
                recommendations.append("📊 Marginal complexity benefit. Consider simplification options.")

            if not dominance_analysis.get('dominance_detected', False):
                recommendations.append("✅ No baseline dominance detected. Complex model provides value.")

            if len(recommendations) == 0:
                recommendations.append("📈 Continue with current model.")

            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise
