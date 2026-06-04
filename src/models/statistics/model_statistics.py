#!/usr/bin/env python3
"""
Model Statistics - Health Trend Analysis
Handles health trend calculation and common issue analysis.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelStatistics")


class ModelStatistics:
    """
    Model statistics analyzer.

    Handles:
    - Health trend calculation
    - Common issue identification
    - Retraining frequency calculation
    - Model health summary
    """

    def __init__(self):
        """Initialize Model Statistics."""
        self.logger = logger
        self.logger.info("✅ ModelStatistics initialized")

    def calculate_health_trend(self, analyses: list[dict[str, Any]]) -> str:
        """Calculate health score trend."""

        try:
            if len(analyses) < 5:
                return 'insufficient_data'

            # Get recent health scores
            recent_scores = [analysis['overall_health_score'] for analysis in analyses[-10:]]

            # Calculate trend
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]

            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'degrading'
            else:
                return 'stable'

        except Exception as e:
            self.logger.error(f"Error calculating health trend: {e}")
            return 'unknown'

    def get_common_issues(self, analyses: list[dict[str, Any]]) -> list[str]:
        """Get most common issues from analyses."""

        try:
            issue_counts: dict[str, int] = {}

            for analysis in analyses:
                if isinstance(analysis, dict) and 'recommendations' in analysis:
                    recommendations = analysis['recommendations']
                else:
                    recommendations = []

                for recommendation in recommendations:
                    # Extract issue type from recommendation
                    if 'BASELINE' in recommendation.upper():
                        issue_counts['baseline_dominance'] = issue_counts.get('baseline_dominance', 0) + 1
                    elif 'OVERFITTING' in recommendation.upper():
                        issue_counts['overfitting'] = issue_counts.get('overfitting', 0) + 1
                    elif 'DRIFT' in recommendation.upper():
                        issue_counts['drift'] = issue_counts.get('drift', 0) + 1
                    elif 'REGIME' in recommendation.upper():
                        issue_counts['regime_inconsistency'] = issue_counts.get('regime_inconsistency', 0) + 1

            # Return top issues
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            return [issue[0] for issue in sorted_issues[:5]]

        except Exception as e:
            self.logger.error(f"Error getting common issues: {e}")
            raise RuntimeError("Failed to get common model issues") from e

    def calculate_retraining_frequency(self,
                                      retraining_history: list[dict[str, Any]],
                                      model_name: str | None = None) -> float:
        """Calculate retraining frequency for model."""

        try:
            if model_name:
                model_retrainings = [
                    record for record in retraining_history
                    if model_name in record.get('reason', '')
                ]
            else:
                model_retrainings = retraining_history

            if not model_retrainings:
                return 0.0

            # Calculate frequency over last 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            recent_retrainings = [
                record for record in model_retrainings
                if record['timestamp'] >= cutoff_time
            ]

            return len(recent_retrainings) / 30.0  # Retrainings per day

        except Exception as e:
            self.logger.error(f"Error calculating retraining frequency: {e}")
            return 0.0

    def get_model_health_summary(self,
                                analysis_history: list[dict[str, Any]],
                                retraining_history: list[dict[str, Any]],
                                model_name: str | None = None) -> dict[str, Any]:
        """Get health summary for specific model or all models."""

        try:
            if model_name:
                # Get specific model analysis
                model_analyses = [
                    analysis for analysis in analysis_history
                    if analysis['model_name'] == model_name
                ]
            else:
                # Get all model analyses
                model_analyses = analysis_history

            if not model_analyses:
                return {'error': 'No analysis data available'}

            # Calculate summary statistics
            summary = {
                'model_name': model_name or 'all_models',
                'total_analyses': len(model_analyses),
                'latest_health_score': model_analyses[-1]['overall_health_score'],
                'health_trend': self.calculate_health_trend(model_analyses),
                'common_issues': self.get_common_issues(model_analyses),
                'retraining_frequency': self.calculate_retraining_frequency(retraining_history, model_name)
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting model health summary: {e}")
            return {'error': str(e)}
