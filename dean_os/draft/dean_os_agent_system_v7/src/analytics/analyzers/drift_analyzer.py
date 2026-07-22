from typing import Any

from src.analytics.interfaces import IAnalyzer
from src.monitoring.feature_drift_monitor import FeatureDriftMonitor


class DriftAnalyzer(IAnalyzer):
    """
    Adapter for FeatureDriftMonitor to integrate with UnifiedAnalyticsEngine.
    """

    def __init__(self, threshold: float = 0.05, config: dict = None):
        if config:
            threshold = config.get('threshold', threshold)
        self.monitor = FeatureDriftMonitor(drift_threshold=threshold)

    def analyze(self, data: Any, **kwargs) -> dict[str, Any]:
        """
        Executes drift analysis.

        Expects 'features_data' in the data dictionary.
        """
        if isinstance(data, dict) and "features_data" in data:
            return self.monitor.detect_drift(data["features_data"])

        # Fallback if raw DataFrame provided
        return self.monitor.detect_drift(data)
