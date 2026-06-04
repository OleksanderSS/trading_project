"""
Regime Importance Tracker - Dynamic Feature Importance Tracking Across Market Regimes
Tracks and adapts feature importance changes across different market regimes.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.features.analysis.importance_stability_analyzer import ImportanceStabilityAnalyzer
from src.features.analysis.market_conditions_analyzer import MarketConditionsAnalyzer
from src.features.analysis.regime_adapter import RegimeAdapter

logger = ProjectLogger.get_logger('RegimeImportanceTracker')


class RegimeImportanceTracker:
    """
    Tracks feature importance changes across market regimes and adapts selection strategies.

    This tracker monitors:
    - Feature importance stability across different market regimes
    - Regime-specific importance patterns
    - Automatic regime switching detection
    - Dynamic method weight adaptation for feature selection

    Critical for maintaining model performance across changing market conditions.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize RegimeImportanceTracker.

        Args:
            config: Configuration dictionary for regime tracking
        """
        self.logger = logger
        self.config = config or {}

        # Initialize modular components
        self.market_analyzer = MarketConditionsAnalyzer()
        self.stability_analyzer = ImportanceStabilityAnalyzer(
            stability_threshold=self.config.get('importance_stability', 0.3),
            min_samples=self.config.get('min_samples_per_regime', 50)
        )
        self.regime_adapter = RegimeAdapter()

        # Configuration
        self.importance_history: list[dict[str, Any]] = []
        self.regime_switch_points: list[dict[str, Any]] = []
        self.adaptation_enabled = self.config.get('adaptation_enabled', True)
        self.storage_path = Path(self.config.get('storage_path', 'data/analysis/regime_importance'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.logger.info('✅ RegimeImportanceTracker initialized with modular components')

    async def track_feature_importance(self,
                                       current_importance: dict[str, float],
                                       market_data: pd.DataFrame,
                                       model_metadata: dict[str, Any] | None = None,
                                       current_time: datetime | None = None) -> dict[str, Any]:
        """
        Track feature importance and analyze regime-specific patterns.

        Args:
            current_importance: Current feature importance dictionary
            market_data: Market data for regime detection
            model_metadata: Model metadata
            current_time: Current timestamp (uses now if None)

        Returns:
            Dict with regime analysis and recommendations
        """
        if current_time is None:
            current_time = datetime.now()

        self.logger.info(f'📊 Tracking feature importance at {current_time}')

        results = {
            'timestamp': current_time,
            'current_importance': current_importance,
            'current_regime': None,
            'regime_stability': {},
            'importance_changes': {},
            'regime_switch_detected': False,
            'adaptation_recommendations': [],
            'method_weights': {}
        }

        try:
            # Detect current regime
            current_regime = self.market_analyzer.detect_market_regime(market_data)
            results['current_regime'] = current_regime

            # Record importance with regime
            importance_record = {
                'timestamp': current_time,
                'importance': current_importance.copy(),
                'regime': current_regime,
                'market_conditions': self.market_analyzer.calculate_market_conditions(market_data)
            }
            self.importance_history.append(importance_record)

            # Analyze importance stability
            stability_analysis = self.stability_analyzer.analyze_importance_stability(
                self.importance_history, current_regime
            )
            results['regime_stability'] = stability_analysis

            # Detect regime switch
            regime_switch = self.stability_analyzer.detect_regime_switch(
                self.importance_history, current_regime, current_time, self.regime_switch_points
            )
            results['regime_switch_detected'] = regime_switch['switch_detected']

            if regime_switch['switch_detected']:
                results['regime_switch_info'] = regime_switch
                self.regime_switch_points.append(regime_switch)

            # Calculate importance changes
            importance_changes = self.stability_analyzer.calculate_importance_changes(
                self.importance_history, current_importance, current_regime
            )
            results['importance_changes'] = importance_changes

            # Generate adaptation recommendations if enabled
            if self.adaptation_enabled:
                recommendations = self.regime_adapter.generate_adaptation_recommendations(
                    current_regime, stability_analysis, importance_changes
                )
                results['adaptation_recommendations'] = recommendations

                method_weights = self.regime_adapter.update_method_weights(
                    current_regime, recommendations
                )
                results['method_weights'] = method_weights

            # Clean old data and store results
            self._clean_old_data()
            self._store_tracking_results(results)

            self.logger.info(f'✅ Regime importance tracking complete. Regime: {current_regime}')

            return results
        except Exception as e:
            self.logger.error(f'Error in regime importance tracking: {e}', exc_info=True)
            raise DataProcessingError(f"Regime importance tracking failed: {e}") from e

    def _clean_old_data(self) -> None:
        """Clean old data to prevent memory issues."""
        try:
            if len(self.importance_history) > 1000:
                self.importance_history = self.importance_history[-1000:]
            if len(self.regime_switch_points) > 100:
                self.regime_switch_points = self.regime_switch_points[-100:]
        except Exception as e:
            self.logger.error(f'Error cleaning old data: {e}', exc_info=True)

    def _store_tracking_results(self, results: dict[str, Any]) -> None:
        """Store tracking results for historical analysis."""
        try:
            timestamp = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f'regime_tracking_{timestamp}.json'
            filepath = self.storage_path / filename
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            files = list(self.storage_path.glob('regime_tracking_*.json'))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for file_to_delete in files[100:]:
                file_to_delete.unlink()
        except Exception as e:
            self.logger.error(f'Failed to store tracking results: {e}', exc_info=True)
            raise DataProcessingError(f"Failed to store tracking results: {e}") from e

    def get_regime_importance_summary(self, days: int = 30) -> dict[str, Any]:
        """Get summary of feature importance across regimes."""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_records = [record for record in self.importance_history
                         if record['timestamp'] >= cutoff_time]

        if not recent_records:
            return {'error': 'No recent regime importance data available'}

        regime_analysis = {}
        for regime in self.market_analyzer.regime_types.keys():
            regime_records = [record for record in recent_records
                           if record['regime'] == regime]
            if regime_records:
                all_features = set()
                feature_importances: dict[str, list[float]] = {}
                for record in regime_records:
                    for feature_name, importance in record['importance'].items():
                        all_features.add(feature_name)
                        if feature_name not in feature_importances:
                            feature_importances[feature_name] = []
                        feature_importances[feature_name].append(importance)

                regime_stats = {}
                for feature_name in all_features:
                    values = feature_importances[feature_name]
                    regime_stats[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

                regime_analysis[regime] = {
                    'record_count': len(regime_records),
                    'feature_count': len(all_features),
                    'feature_importance': regime_stats
                }

        regime_transitions = []
        for i in range(1, len(recent_records)):
            if recent_records[i]['regime'] != recent_records[i - 1]['regime']:
                regime_transitions.append({
                    'from_regime': recent_records[i - 1]['regime'],
                    'to_regime': recent_records[i]['regime'],
                    'timestamp': recent_records[i]['timestamp']
                })

        summary = {
            'period_days': days,
            'total_records': len(recent_records),
            'regime_analysis': regime_analysis,
            'regime_transitions': regime_transitions,
            'most_common_regime': self._get_most_common_regime(recent_records),
            'regime_stability_scores': self._calculate_regime_stability_scores(recent_records)
        }

        return summary

    def _get_most_common_regime(self, records: list[dict[str, Any]]) -> str:
        """Get the most common regime in the period."""
        regime_counts: dict[str, int] = {}
        for record in records:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        if regime_counts:
            return str(max(regime_counts.items(), key=lambda x: x[1])[0])
        return 'normal'

    def _calculate_regime_stability_scores(self, records: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate stability scores for each regime."""
        regime_stability = {}
        for regime in self.market_analyzer.regime_types.keys():
            regime_records = [record for record in records if record['regime'] == regime]
            if len(regime_records) >= 2:
                persistence_times = []
                current_persistence = 1
                for i in range(1, len(regime_records)):
                    if regime_records[i]['regime'] == regime:
                        current_persistence += 1
                    else:
                        persistence_times.append(current_persistence)
                        current_persistence = 1
                persistence_times.append(current_persistence)
                avg_persistence = np.mean(persistence_times)
                regime_stability[regime] = avg_persistence
        return regime_stability


def get_regime_importance_tracker(config: dict[str, Any] | None=None
    ) ->RegimeImportanceTracker:
    """Factory function to get RegimeImportanceTracker instance."""
    return RegimeImportanceTracker(config)


async def track_regime_importance_quick(current_importance: dict[str, float
    ], market_data: pd.DataFrame, config: dict[str, Any] | None=None
    ) ->dict[str, Any]:
    """
    Quick regime importance tracking.

    Args:
        current_importance: Current feature importance dictionary
        market_data: Market data for regime detection
        config: Configuration dictionary

    Returns:
        Regime tracking result dictionary
    """
    tracker = get_regime_importance_tracker(config)
    return await tracker.track_feature_importance(current_importance,
        market_data)
