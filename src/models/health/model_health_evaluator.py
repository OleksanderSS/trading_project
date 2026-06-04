#!/usr/bin/env python3
"""
Model Health Evaluator - Health Scoring and Recommendations
Handles health score calculation and recommendation generation.
"""

from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelHealthEvaluator")


class ModelHealthEvaluator:
    """
    Model health evaluator.

    Handles:
    - Overall health score calculation
    - Comprehensive recommendation generation
    - Action requirement determination
    - Retraining need assessment
    """

    def __init__(self):
        """Initialize Model Health Evaluator."""
        self.logger = logger
        self.logger.info("✅ ModelHealthEvaluator initialized")

    def calculate_overall_health_score(self, analysis_results: dict[str, Any]) -> float:
        """Calculate overall model health score from all analysis results."""

        try:
            health_scores = []

            # Baseline analysis score
            baseline_result = analysis_results.get('baseline', {})
            if baseline_result.get('status') == 'completed':
                # Score based on whether complex model dominates baselines
                if not baseline_result.get('baseline_dominance_detected', True):
                    health_scores.append(0.8)  # Good - no baseline dominance
                else:
                    health_scores.append(0.3)  # Poor - baseline dominates

            # Regime analysis score
            regime_result = analysis_results.get('regime', {})
            if regime_result.get('status') == 'completed':
                consistency = regime_result.get('consistency_analysis', {}).get('overall_consistency', 0.5)
                health_scores.append(consistency)

            # Overfitting analysis score
            overfitting_result = analysis_results.get('overfitting', {})
            if overfitting_result.get('status') == 'completed':
                signal_count = overfitting_result.get('overfitting_signals', {}).get('total_signals', 0)
                # Inverse relationship - fewer signals = better health
                overfitting_score = max(0.0, 1.0 - (signal_count * 0.2))
                health_scores.append(overfitting_score)

            # Drift analysis score
            drift_result = analysis_results.get('drift', {})
            if drift_result.get('status') == 'completed':
                drift_status = drift_result.get('drift_status', 'stable')
                if drift_status == 'stable':
                    health_scores.append(0.9)
                elif 'low' in drift_status:
                    health_scores.append(0.7)
                elif 'medium' in drift_status:
                    health_scores.append(0.5)
                elif 'high' in drift_status:
                    health_scores.append(0.3)
                elif 'critical' in drift_status:
                    health_scores.append(0.1)
                else:
                    health_scores.append(0.5)

            # Calculate overall score
            if health_scores:
                return float(np.mean(health_scores))
            else:
                return 0.5  # Default score if no analysis completed

        except Exception as e:
            self.logger.error(f"Error calculating overall health score: {e}")
            return 0.5

    def generate_comprehensive_recommendations(self,
                                            analysis_results: dict[str, Any],
                                            health_score: float) -> list[str]:
        """Generate comprehensive recommendations from all analysis results."""

        recommendations = []

        try:
            # Overall health recommendation
            if health_score >= 0.8:
                recommendations.append(
                    f"✅ EXCELLENT: Model health score is {health_score:.3f}. "
                    "Model is performing well."
                )
            elif health_score >= 0.6:
                recommendations.append(
                    f"⚠️ GOOD: Model health score is {health_score:.3f}. "
                    "Model is performing adequately but monitor closely."
                )
            elif health_score >= 0.4:
                recommendations.append(
                    f"⚠️ FAIR: Model health score is {health_score:.3f}. "
                    "Model has issues that need attention."
                )
            else:
                recommendations.append(
                    f"🚨 POOR: Model health score is {health_score:.3f}. "
                    "Model has significant issues requiring immediate action."
                )

            # Baseline analysis recommendations
            baseline_result = analysis_results.get('baseline', {})
            if baseline_result.get('baseline_dominance_detected', False):
                recommendations.append(
                    "🔧 BASELINE: Simple baseline models outperform complex model. "
                    "Consider simplifying the model architecture."
                )

            # Regime analysis recommendations
            regime_result = analysis_results.get('regime', {})
            if regime_result.get('status') == 'completed':
                regime_recommendations = regime_result.get('recommendations', [])
                recommendations.extend(regime_recommendations)

            # Overfitting analysis recommendations
            overfitting_result = analysis_results.get('overfitting', {})
            if overfitting_result.get('status') == 'completed':
                overfitting_recommendations = overfitting_result.get('recommendations', [])
                recommendations.extend(overfitting_recommendations)

            # Drift analysis recommendations
            drift_result = analysis_results.get('drift', {})
            if drift_result.get('status') == 'completed':
                drift_recommendations = drift_result.get('retraining_recommendations', [])
                recommendations.extend(drift_recommendations)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating comprehensive recommendations: {e}")
            return recommendations

    def determine_action_required(self, recommendations: list[str]) -> bool:
        """Determine if immediate action is required based on recommendations."""

        try:
            # Check for critical indicators
            critical_keywords = [
                'CRITICAL', 'IMMEDIATE', 'STOP', 'DANGER', 'HIGH RISK'
            ]

            for recommendation in recommendations:
                if any(keyword in recommendation.upper() for keyword in critical_keywords):
                    return True

            # Check for high severity issues
            high_severity_keywords = [
                'HIGH', 'RETRAINING REQUIRED', 'OVERFITTING', 'DRIFT DETECTED'
            ]

            high_severity_count = sum(
                1 for recommendation in recommendations
                if any(keyword in recommendation.upper() for keyword in high_severity_keywords)
            )

            return high_severity_count >= 2

        except Exception as e:
            self.logger.error(f"Error determining action required: {e}")
            return False

    def determine_retraining_needed(self, recommendations: list[str]) -> bool:
        """Determine if retraining is recommended based on recommendations."""

        try:
            retraining_keywords = [
                'RETRAIN', 'RETRAINING', 'DEGRADATION', 'DRIFT', 'OVERFITTING'
            ]

            return any(
                keyword in recommendation.upper()
                for recommendation in recommendations
                for keyword in retraining_keywords
            )

        except Exception as e:
            self.logger.error(f"Error determining retraining needed: {e}")
            return False
