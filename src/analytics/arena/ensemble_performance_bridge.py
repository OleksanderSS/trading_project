"""
EnsemblePerformanceBridge
Syncs LiveAdaptiveEnsemble performance data into ModelPerformanceTracker.
Moved from src/integration/ to src/analytics/arena/ where ModelPerformanceTracker lives.
"""

from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger


class EnsemblePerformanceBridge:
    """
    Bridge between LiveAdaptiveEnsemble and ModelPerformanceTracker.
    Synchronizes performance metrics so the tracker has a unified view
    of all model performance including live ensemble data.
    """

    _SYNC_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(self, live_ensemble: Any, performance_tracker: Any) -> None:
        self.live_ensemble = live_ensemble
        self.performance_tracker = performance_tracker
        self.logger = ProjectLogger.get_logger(__name__)
        self._last_sync: datetime | None = None
        self._last_result: dict[str, Any] = {}
        self.logger.info("EnsemblePerformanceBridge initialized")

    def sync_ensemble_performance_to_tracker(self, force: bool = False) -> dict[str, Any]:
        """
        Sync ensemble performance data into ModelPerformanceTracker.

        Args:
            force: Skip the 5-minute cooldown and sync immediately.

        Returns:
            Dict with sync results including records_updated and success flag.
        """
        if not force and self._last_sync:
            elapsed = (datetime.now() - self._last_sync).total_seconds()
            if elapsed < self._SYNC_INTERVAL_SECONDS:
                return self._last_result

        try:
            metrics = self._extract_ensemble_metrics()
            records = self._convert_to_tracker_format(metrics)

            updated = 0
            for record in records:
                if self.performance_tracker.update_performance(record):
                    updated += 1

            result = {
                'success': True,
                'sync_time': datetime.now().isoformat(),
                'ensemble_metrics_count': len(metrics),
                'records_updated': updated,
            }
            self._last_sync = datetime.now()
            self._last_result = result
            self.logger.info(f"Ensemble sync: {updated} records updated")
            return result

        except Exception as e:
            self.logger.error(f"Ensemble sync failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_unified_performance_view(self) -> dict[str, Any]:
        """Merge ensemble and tracker data into a single view."""
        try:
            ensemble_data = self._extract_ensemble_metrics()
            tracker_data = self.performance_tracker.get_all_performance() if hasattr(
                self.performance_tracker, 'get_all_performance') else []

            unified: dict[str, Any] = {}
            for record in ensemble_data:
                name = record.get('model_name')
                if name:
                    unified[name] = {'ensemble': record, 'tracker': None}
            for record in tracker_data:
                name = record.get('model_name')
                if name:
                    if name in unified:
                        unified[name]['tracker'] = record
                    else:
                        unified[name] = {'ensemble': None, 'tracker': record}

            return {
                'unified_performance': unified,
                'total_unique_models': len(unified),
                'last_updated': datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to get unified performance view: {e}")
            return {'success': False, 'error': str(e)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_ensemble_metrics(self) -> list[dict[str, Any]]:
        """Extract performance metrics from LiveAdaptiveEnsemble."""
        metrics: list[dict[str, Any]] = []
        try:
            history = getattr(self.live_ensemble, 'model_metrics_history', [])
            for m in history:
                metrics.append({
                    'model_name': getattr(m, 'model_id', ''),
                    'model_type': getattr(m, 'model_type', ''),
                    'timestamp': getattr(m, 'timestamp', datetime.now()),
                    'sharpe_ratio': getattr(m, 'sharpe_ratio', 0.0),
                    'hit_rate': getattr(m, 'hit_rate', 0.0),
                    'precision': getattr(m, 'precision', 0.0),
                    'avg_return_per_trade': getattr(m, 'avg_return_per_trade', 0.0),
                    'predictions_count': getattr(m, 'predictions_count', 0),
                    'source': 'live_ensemble',
                })
        except Exception as e:
            self.logger.warning(f"Could not extract ensemble metrics: {e}")
        return metrics

    def _convert_to_tracker_format(self, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert ensemble metrics to ModelPerformanceTracker record format."""
        records = []
        for m in metrics:
            records.append({
                'model_name': m['model_name'],
                'model_type': m['model_type'],
                'avg_win_rate': m.get('hit_rate', 0.0),
                'avg_sharpe_ratio': m.get('sharpe_ratio', 0.0),
                'avg_precision': m.get('precision', 0.0),
                'avg_return': m.get('avg_return_per_trade', 0.0),
                'total_trades': m.get('predictions_count', 0),
                'last_updated': m.get('timestamp', datetime.now()),
                'source': 'live_ensemble_sync',
            })
        return records
