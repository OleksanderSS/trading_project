#!/usr/bin/env python3
"""
Regime Adapter - Adaptation Recommendations and Method Weight Updates
Handles regime-specific adaptation recommendations and method weight updates.
"""

from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegimeAdapter")


class RegimeAdapter:
    """
    Regime adapter for feature selection adaptation.

    Handles:
    - Adaptation recommendations generation
    - Regime-specific recommendations
    - Method weight updates
    """

    def __init__(self, regime_types: dict[str, Any] | None = None):
        """
        Initialize Regime Adapter.

        Args:
            regime_types: Dictionary of regime type configurations
        """
        self.logger = logger
        self.regime_types = regime_types or {
            'normal': {
                'description': 'Normal market conditions',
                'volatility_range': (0.01, 0.02),
                'trend_strength': (-0.001, 0.001)
            },
            'volatile': {
                'description': 'High volatility market',
                'volatility_range': (0.02, 0.05),
                'trend_strength': (-0.003, 0.003)
            },
            'trending_up': {
                'description': 'Strong uptrend market',
                'volatility_range': (0.015, 0.025),
                'trend_strength': (0.002, 0.005)
            },
            'trending_down': {
                'description': 'Strong downtrend market',
                'volatility_range': (0.015, 0.025),
                'trend_strength': (-0.005, -0.002)
            },
            'crisis': {
                'description': 'Market crisis conditions',
                'volatility_range': (0.04, 0.1),
                'trend_strength': (-0.01, 0.01)
            }
        }
        self.logger.info("✅ RegimeAdapter initialized")

    def generate_adaptation_recommendations(self,
                                           current_regime: str,
                                           stability_analysis: dict[str, Any],
                                           importance_changes: dict[str, Any]) -> list[str]:
        """Generate adaptation recommendations based on analysis."""
        recommendations = []

        try:
            stability_score = stability_analysis.get('stability_score', 1.0)
            if stability_score < 0.5:
                recommendations.append(
                    f'⚠️ Low importance stability ({stability_score:.2f}) in {current_regime} regime. '
                    'Consider increasing feature selection frequency.'
                )

            unstable_features = stability_analysis.get('unstable_features', [])
            if unstable_features:
                recommendations.append(
                    f'🔄 {len(unstable_features)} features show unstable importance in {current_regime} regime. '
                    'Consider regime-specific feature selection.'
                )

            significant_changes = importance_changes.get('significant_changes', [])
            if significant_changes:
                recommendations.append(
                    f'📊 {len(significant_changes)} features show significant importance changes. '
                    'Review feature engineering pipeline.'
                )

            regime_recommendations = self.get_regime_specific_recommendations(current_regime)
            recommendations.extend(regime_recommendations)

            return recommendations
        except Exception as e:
            self.logger.error(f'Error generating recommendations: {e}')
            return []

    def get_regime_specific_recommendations(self, regime: str) -> list[str]:
        """Get regime-specific feature selection recommendations."""
        recommendations = []

        try:
            if regime == 'volatile':
                recommendations.extend([
                    '🌊 Volatile regime: Increase emphasis on volatility-based features',
                    '🌊 Use shorter lookback periods for technical indicators',
                    '🌊 Consider risk management features more heavily'
                ])
            elif regime == 'trending_up':
                recommendations.extend([
                    '📈 Uptrend regime: Emphasize momentum features',
                    '📈 Increase weight for trend-following indicators',
                    '📈 Consider breakout detection features'
                ])
            elif regime == 'trending_down':
                recommendations.extend([
                    '📉 Downtrend regime: Emphasize mean-reversion features',
                    '📉 Increase weight for contrarian indicators',
                    '📉 Consider short-selling signals'
                ])
            elif regime == 'crisis':
                recommendations.extend([
                    '🚨 Crisis regime: Emphasize safety features',
                    '🚨 Increase weight for defensive indicators',
                    '🚨 Consider market stress indicators'
                ])
            else:
                recommendations.extend([
                    '✅ Normal regime: Use balanced feature selection',
                    '✅ Maintain standard feature weights',
                    '✅ Regular model retraining schedule'
                ])

            return recommendations
        except Exception as e:
            self.logger.error(f'Error getting regime-specific recommendations: {e}')
            return recommendations

    def update_method_weights(self,
                             current_regime: str,
                             recommendations: list[str]) -> dict[str, float]:
        """Update feature selection method weights based on regime and recommendations."""
        base_weights = {
            'normal': {'correlation': 0.4, 'mutual_info': 0.3, 'lgbm': 0.2, 'rf': 0.1},
            'volatile': {'correlation': 0.1, 'mutual_info': 0.4, 'lgbm': 0.4, 'rf': 0.1},
            'trending_up': {'correlation': 0.3, 'mutual_info': 0.2, 'lgbm': 0.4, 'rf': 0.1},
            'trending_down': {'correlation': 0.3, 'mutual_info': 0.2, 'lgbm': 0.4, 'rf': 0.1},
            'crisis': {'correlation': 0.1, 'mutual_info': 0.5, 'lgbm': 0.3, 'rf': 0.1}
        }

        method_weights = base_weights.get(current_regime, base_weights['normal']).copy()

        for recommendation in recommendations:
            if 'unstable importance' in recommendation:
                method_weights['correlation'] *= 0.8
                method_weights['lgbm'] *= 1.2
                method_weights['rf'] *= 1.1
            elif 'significant importance changes' in recommendation:
                method_weights['mutual_info'] *= 1.3
            elif 'volatile regime' in recommendation:
                method_weights['lgbm'] *= 1.15
                method_weights['mutual_info'] *= 1.1
                method_weights['correlation'] *= 0.9

        total_weight = sum(method_weights.values())
        if total_weight > 0:
            method_weights = {k: (v / total_weight) for k, v in method_weights.items()}

        return method_weights
