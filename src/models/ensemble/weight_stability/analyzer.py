from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilityAnalyzer")

class WeightStabilityAnalyzer:
    """Analyzes weight stability and provides recommendations."""

    def __init__(self, config: Any):
        self.logger = logger
        self.config = config

    def _check_stability_score(self, overall_stability: float) -> list[str]:
        """Check overall stability score and generate recommendation."""
        if overall_stability < 0.5:
            return [f"🚨 CRITICAL: Very low stability score ({overall_stability:.3f}). Immediate action required."]
        elif overall_stability < 0.7:
            return [f"⚠️ WARNING: Low stability score ({overall_stability:.3f}). Consider stabilization measures."]
        elif overall_stability >= 0.8:
            return [f"✅ GOOD: High stability score ({overall_stability:.3f}). Weights are stable."]
        return []

    def _check_volatility(self, metrics: dict[str, Any]) -> list[str]:
        """Check weight volatility and generate recommendation."""
        if 'volatility' in metrics:
            vol = metrics['volatility'].get('average_volatility', 0.0)
            if vol > self.config.stability_threshold:
                return [f"📊 HIGH VOLATILITY: Weight volatility is {vol:.4f}. Consider reducing update frequency or increasing smoothing."]
        return []

    def _check_drift(self, metrics: dict[str, Any]) -> list[str]:
        """Check weight drift and generate recommendation."""
        if 'drift' in metrics:
            drift = metrics['drift'].get('total_drift', 0.0)
            if drift > self.config.STABILITY_METRICS['drift']['threshold']:
                return [f"📈 HIGH DRIFT: Weight drift is {drift:.4f}. Consider weight rebalancing or reset."]
        return []

    def _check_consistency(self, metrics: dict[str, Any]) -> list[str]:
        """Check weight consistency and generate recommendation."""
        if 'consistency' in metrics:
            cons = metrics['consistency']
            if cons < self.config.STABILITY_METRICS['consistency']['threshold']:
                return [f"🔄 LOW CONSISTENCY: Weight consistency is {cons:.3f}. Consider increasing smoothing factor."]
        return []

    def _check_excessive_changes(self, excessive_changes: dict[str, Any]) -> list[str]:
        """Check for excessive model weight changes."""
        if excessive_changes.get('has_excessive', False):
            return [f"⚠️ EXCESSIVE CHANGES: {len(excessive_changes['excessive_models'])} models exceeded threshold."]
        return []

    def generate_stability_recommendations(self,
                                         metrics: dict[str, Any],
                                         excessive_changes: dict[str, Any]) -> list[str]:
        """Generate stability recommendations."""
        recommendations = []
        try:
            overall_stability = metrics.get('overall_stability', 1.0)
            recommendations.extend(self._check_stability_score(overall_stability))
            recommendations.extend(self._check_volatility(metrics))
            recommendations.extend(self._check_drift(metrics))
            recommendations.extend(self._check_consistency(metrics))
            recommendations.extend(self._check_excessive_changes(excessive_changes))
            return recommendations
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return recommendations

    def determine_stability_status(self, overall_stability: float) -> str:
        """Update current stability status."""
        if overall_stability >= 0.8:
            return 'stable'
        elif overall_stability >= 0.6:
            return 'moderately_stable'
        elif overall_stability >= 0.4:
            return 'unstable'
        else:
            return 'highly_unstable'

    def is_action_required(self, recommendations: list[str]) -> bool:
        """Determine if action is required based on recommendations."""
        critical_keywords = ['CRITICAL', 'IMMEDIATE', 'HIGH VOLATILITY', 'HIGH DRIFT', 'EXCESSIVE CHANGES']
        return any(keyword in rec.upper() for rec in recommendations for keyword in critical_keywords)

    def analyze_stability_trend(self, scores: list[float]) -> str:
        """Analyze trend in stability scores."""
        if len(scores) < 5:
            return 'insufficient_data'
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
