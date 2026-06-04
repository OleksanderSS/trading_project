#!/usr/bin/env python3
"""
Importance Stability Analyzer - Feature Importance Stability Analysis
Handles importance stability analysis and regime switch detection.
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ImportanceStabilityAnalyzer")


class ImportanceStabilityAnalyzer:
    """
    Importance stability analyzer.

    Handles:
    - Feature importance stability analysis
    - Regime switch detection
    - Importance changes calculation
    """

    def __init__(self, stability_threshold: float = 0.3, min_samples: int = 50):
        """
        Initialize Importance Stability Analyzer.

        Args:
            stability_threshold: Threshold for stability detection
            min_samples: Minimum samples required for analysis
        """
        self.logger = logger
        self.stability_threshold = stability_threshold
        self.min_samples = min_samples
        self.logger.info("✅ ImportanceStabilityAnalyzer initialized")

    def analyze_importance_stability(self,
                                    importance_history: list[dict[str, Any]],
                                    current_regime: str) -> dict[str, Any]:
        """Analyze feature importance stability for current regime."""
        stability_analysis = {
            'regime': current_regime,
            'stability_score': 1.0,
            'stable_features': [],
            'unstable_features': [],
            'importance_variance': {},
            'sample_count': 0
        }

        try:
            regime_importance = [
                record['importance'] for record in importance_history
                if record['regime'] == current_regime
            ]

            stability_analysis['sample_count'] = len(regime_importance)

            if len(regime_importance) < self.min_samples:
                stability_analysis['stability_score'] = 0.0
                return stability_analysis

            feature_importances: dict[str, list[float]] = {}
            for record in regime_importance:
                for feature_name, importance in record['importance'].items():
                    if feature_name not in feature_importances:
                        feature_importances[feature_name] = []
                    feature_importances[feature_name].append(importance)

            for feature_name, importance_values in feature_importances.items():
                if len(importance_values) >= 2:
                    importance_variance = np.var(importance_values)
                    importance_mean = np.mean(importance_values)
                    cv = (np.std(importance_values) / importance_mean
                          if importance_mean > 0 else float('inf'))

                    stability_analysis['importance_variance'][feature_name] = {
                        'variance': importance_variance,
                        'mean': importance_mean,
                        'cv': cv,
                        'values': importance_values
                    }

                    if cv <= self.stability_threshold:
                        stability_analysis['stable_features'].append(feature_name)
                    else:
                        stability_analysis['unstable_features'].append(feature_name)

            total_features = (len(stability_analysis.get('stable_features', [])) +
                            len(stability_analysis.get('unstable_features', [])))

            if total_features > 0:
                stability_analysis['stability_score'] = (
                    len(stability_analysis.get('stable_features', [])) / total_features
                )

            self.logger.info(
                f"📊 Regime {current_regime} stability: {stability_analysis['stability_score']:.2f}"
            )

            return stability_analysis
        except Exception as e:
            self.logger.error(f'Error analyzing importance stability: {e}', exc_info=True)
            return stability_analysis

    def detect_regime_switch(self,
                           importance_history: list[dict[str, Any]],
                           current_regime: str,
                           current_time: datetime,
                           regime_switch_points: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect if market regime has switched."""
        switch_detection = {
            'switch_detected': False,
            'previous_regime': None,
            'switch_confidence': 0.0,
            'switch_reason': '',
            'time_since_last_switch': None
        }

        try:
            if not importance_history:
                return switch_detection

            recent_records = importance_history[-10:]
            if len(recent_records) < 2:
                return switch_detection

            previous_regime = recent_records[-2]['regime']

            if previous_regime != current_regime:
                switch_detection['switch_detected'] = True
                switch_detection['previous_regime'] = previous_regime
                switch_detection['switch_confidence'] = 0.8
                switch_detection['switch_reason'] = (
                    f'Regime changed from {previous_regime} to {current_regime}'
                )

                last_switch_time = None
                for switch_point in reversed(regime_switch_points):
                    if switch_point['timestamp'] < current_time:
                        last_switch_time = switch_point['timestamp']
                        break

                if last_switch_time:
                    time_since_last = current_time - last_switch_time
                    switch_detection['time_since_last_switch'] = time_since_last.total_seconds() / 3600

                self.logger.info(
                    f'🔄 Regime switch detected: {previous_regime} -> {current_regime}'
                )

            return switch_detection
        except Exception as e:
            self.logger.error(f'Error detecting regime switch: {e}', exc_info=True)
            return switch_detection

    def calculate_importance_changes(self,
                                   importance_history: list[dict[str, Any]],
                                   current_importance: dict[str, float],
                                   current_regime: str) -> dict[str, Any]:
        """Calculate changes in feature importance."""
        changes_analysis: dict[str, Any] = {
            'significant_changes': [],
            'change_summary': {},
            'regime_comparison': {}
        }

        try:
            regime_importance = [
                record['importance'] for record in importance_history
                if record['regime'] == current_regime
            ]

            if not regime_importance:
                return changes_analysis

            last_importance = regime_importance[-1]

            for feature_name, current_imp in current_importance.items():
                if feature_name in last_importance:
                    last_imp = last_importance[feature_name]

                    if last_imp != 0:
                        relative_change = abs(current_imp - last_imp) / last_imp
                    else:
                        relative_change = 1.0 if current_imp != 0 else 0.0

                    changes_analysis['change_summary'][feature_name] = {
                        'last_importance': last_imp,
                        'current_importance': current_imp,
                        'absolute_change': abs(current_imp - last_imp),
                        'relative_change': relative_change,
                        'significant_change': relative_change > self.stability_threshold
                    }

                    if relative_change > self.stability_threshold:
                        changes_analysis['significant_changes'].append({
                            'feature': feature_name,
                            'change_type': 'importance_drift',
                            'relative_change': relative_change,
                            'last_value': last_imp,
                            'current_value': current_imp
                        })

            # Compare with other regimes
            other_regimes = {record['regime'] for record in importance_history} - {current_regime}
            for other_regime in other_regimes:
                other_regime_importance = [
                    record['importance'] for record in importance_history
                    if record['regime'] == other_regime
                ]

                if other_regime_importance:
                    other_avg_importance: dict[str, Any] = {}
                    for record in other_regime_importance:
                        for feature_name, importance in record['importance'].items():
                            if feature_name not in other_avg_importance:
                                other_avg_importance[feature_name] = []
                            other_avg_importance[feature_name].append(importance)

                    for feature_name in other_avg_importance:
                        other_avg_importance[feature_name] = np.mean(other_avg_importance[feature_name])

                    changes_analysis['regime_comparison'][other_regime] = other_avg_importance

            return changes_analysis
        except Exception as e:
            self.logger.error(f'Error calculating importance changes: {e}', exc_info=True)
            return changes_analysis
