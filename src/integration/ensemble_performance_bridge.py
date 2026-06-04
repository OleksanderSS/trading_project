"""
Ensemble Performance Bridge
Connects LiveAdaptiveEnsemble with ModelPerformanceTracker
Synchronizes performance data between ensemble and arena systems
"""

from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger


class EnsemblePerformanceBridge:
    """
    Bridge between LiveAdaptiveEnsemble and ModelPerformanceTracker
    Synchronizes performance metrics and provides unified interface
    """

    def __init__(self, live_ensemble, performance_tracker, logger=None):
        """
        Initialize bridge with ensemble and tracker

        Args:
            live_ensemble: LiveAdaptiveEnsemble instance
            performance_tracker: ModelPerformanceTracker instance
            logger: Logger instance
        """
        self.logger = logger or ProjectLogger.get_logger(__name__)
        self.live_ensemble = live_ensemble
        self.performance_tracker = performance_tracker

        # Cache for synchronized data
        self._sync_cache = {}
        self._last_sync_time = None

        self.logger.info("✅ EnsemblePerformanceBridge initialized")

    def sync_ensemble_performance_to_tracker(self, force_sync: bool = False) -> dict[str, Any]:
        """
        Synchronize ensemble performance data to ModelPerformanceTracker

        Args:
            force_sync: Force sync even if recently synced

        Returns:
            Dict with sync results
        """
        try:
            # Check if sync is needed (sync every 5 minutes)
            if not force_sync and self._last_sync_time:
                time_since_sync = datetime.now() - self._last_sync_time
                if time_since_sync.total_seconds() < 300:  # 5 minutes
                    cached_result = self._sync_cache.get('last_sync_result', {})
                    if isinstance(cached_result, dict):
                        return cached_result
                    else:
                        return {}

            # Get ensemble performance data
            ensemble_metrics = self._extract_ensemble_metrics()

            # Convert to tracker format
            tracker_records = self._convert_to_tracker_format(ensemble_metrics)

            # Update tracker
            updated_count = 0
            for record in tracker_records:
                if self.performance_tracker.update_performance(record):
                    updated_count += 1

            sync_result = {
                'sync_time': datetime.now(),
                'ensemble_metrics_count': len(ensemble_metrics),
                'tracker_records_created': len(tracker_records),
                'records_updated': updated_count,
                'success': True
            }

            # Cache sync result
            self._sync_cache['last_sync_result'] = sync_result
            self._last_sync_time = datetime.now()

            self.logger.info(f"✅ Synced {updated_count} performance records to tracker")
            return sync_result

        except Exception as e:
            self.logger.error(f"❌ Failed to sync ensemble performance: {e}")
            return {'success': False, 'error': str(e)}

    def get_unified_performance_view(self) -> dict[str, Any]:
        """
        Get unified view of performance from both ensemble and tracker

        Returns:
            Dict with unified performance data
        """
        try:
            # Get ensemble performance
            ensemble_data = self._extract_ensemble_metrics()

            # Get tracker performance
            tracker_data = self.performance_tracker.get_all_performance()

            # Merge and deduplicate
            unified_data = self._merge_performance_data(ensemble_data, tracker_data)

            return {
                'unified_performance': unified_data,
                'ensemble_models': len(ensemble_data),
                'tracker_models': len(tracker_data),
                'total_unique_models': len(unified_data),
                'last_updated': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get unified performance view: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_ensemble_metrics(self) -> list[dict[str, Any]]:
        """Extract performance metrics from LiveAdaptiveEnsemble"""
        try:
            metrics = []

            # Get model metrics history from ensemble
            if hasattr(self.live_ensemble, 'model_metrics_history'):
                for metric in self.live_ensemble.model_metrics_history:
                    # Convert to standard format
                    metrics.append({
                        'model_name': metric.model_id,
                        'model_type': metric.model_type,
                        'timestamp': metric.timestamp,
                        'sharpe_ratio': metric.sharpe_ratio,
                        'hit_rate': metric.hit_rate,
                        'precision': metric.precision,
                        'recall': metric.recall,
                        'avg_return_per_trade': metric.avg_return_per_trade,
                        'max_consecutive_losses': metric.max_consecutive_losses,
                        'predictions_count': metric.predictions_count,
                        'source': 'live_ensemble'
                    })

            # Get current weights as additional metric
            if hasattr(self.live_ensemble, 'current_weights'):
                for model_id, weight in self.live_ensemble.current_weights.items():
                    metrics.append({
                        'model_name': model_id,
                        'metric_type': 'ensemble_weight',
                        'value': weight,
                        'timestamp': datetime.now(),
                        'source': 'live_ensemble'
                    })

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to extract ensemble metrics: {e}")
            raise RuntimeError("Failed to extract ensemble metrics") from e

    def _convert_to_tracker_format(self, ensemble_metrics: list[dict]) -> list[dict]:
        """Convert ensemble metrics to ModelPerformanceTracker format"""
        try:
            tracker_records = []

            for metric in ensemble_metrics:
                if metric.get('metric_type') == 'ensemble_weight':
                    continue  # Skip weight metrics for tracker

                # Convert to tracker record format
                record = {
                    'model_name': metric['model_name'],
                    'model_type': metric['model_type'],
                    'avg_win_rate': metric.get('hit_rate', 0),
                    'avg_sharpe_ratio': metric.get('sharpe_ratio', 0),
                    'avg_precision': metric.get('precision', 0),
                    'avg_return': metric.get('avg_return_per_trade', 0),
                    'total_trades': metric.get('predictions_count', 0),
                    'max_consecutive_losses': metric.get('max_consecutive_losses', 0),
                    'last_updated': metric['timestamp'],
                    'source': 'live_ensemble_sync'
                }

                tracker_records.append(record)

            return tracker_records

        except Exception as e:
            self.logger.error(f"❌ Failed to convert to tracker format: {e}")
            raise RuntimeError("Failed to convert ensemble metrics to tracker format") from e

    def _merge_performance_data(self, ensemble_data: list[dict], tracker_data: list[dict]) -> dict[str, Any]:
        """Merge ensemble and tracker data, removing duplicates"""
        try:
            unified = {}

            # Add ensemble data
            for record in ensemble_data:
                model_name = record.get('model_name')
                if model_name:
                    unified[model_name] = {
                        'ensemble_data': record,
                        'tracker_data': None
                    }

            # Add tracker data
            for record in tracker_data:
                model_name = record.get('model_name')
                if model_name:
                    if model_name in unified:
                        unified[model_name]['tracker_data'] = record
                    else:
                        unified[model_name] = {
                            'ensemble_data': None,
                            'tracker_data': record
                        }

            return unified

        except Exception as e:
            self.logger.error(f"❌ Failed to merge performance data: {e}")
            raise RuntimeError("Failed to merge ensemble performance data") from e

    def get_ensemble_weights_for_prediction(self) -> dict[str, float]:
        """
        Get current ensemble weights formatted for prediction

        Returns:
            Dict of model_name -> weight
        """
        try:
            if hasattr(self.live_ensemble, 'current_weights'):
                weights = self.live_ensemble.current_weights.copy()
                if isinstance(weights, dict):
                    # Convert all values to float for type safety
                    return {k: float(v) for k, v in weights.items()}
                else:
                    return {}
            return {}

        except Exception as e:
            self.logger.error(f"❌ Failed to get ensemble weights: {e}")
            raise RuntimeError("Failed to get ensemble weights") from e

    def update_ensemble_from_tracker(self, model_names: list[str]) -> bool:
        """
        Update ensemble with models from tracker

        Args:
            model_names: List of model names to add to ensemble

        Returns:
            True if successful
        """
        try:
            # Get model data from tracker
            tracker_models = {}
            for model_name in model_names:
                model_data = self.performance_tracker.get_model_performance(model_name)
                if model_data:
                    tracker_models[model_name] = model_data

            # Update ensemble with new models
            if hasattr(self.live_ensemble, 'add_models_from_tracker'):
                self.live_ensemble.add_models_from_tracker(tracker_models)
                self.logger.info(f"✅ Updated ensemble with {len(tracker_models)} models from tracker")
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Failed to update ensemble from tracker: {e}")
            return False
