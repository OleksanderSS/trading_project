from src.core.logging.logger import ProjectLogger
from src.features.analysis.regime_importance_tracker import get_regime_importance_tracker
from src.monitoring.data_freshness_monitor import get_data_freshness_monitor
from src.monitoring.feature_drift_monitor import get_feature_drift_monitor

logger = ProjectLogger.get_logger("FeatureEngineeringMonitor")

class FeatureEngineeringMonitor:
    def __init__(self):
        self.drift_monitor = get_feature_drift_monitor()
        self.freshness_monitor = get_data_freshness_monitor()
        self.regime_tracker = get_regime_importance_tracker()
        logger.info("✅ Enhanced monitoring components initialized")

    def run_pre_checks(self, data):
        logger.info("Running pre-engineering monitoring checks...")
        result = self._build_data_summary(data)
        if getattr(data, "empty", True):
            result["status"] = "skipped"
            result["reason"] = "empty_data"
            return result

        if getattr(self.drift_monitor, "reference_data", None) is None:
            self.drift_monitor.set_reference_data(data)
            result["drift"] = {"status": "reference_initialized"}
        else:
            result["drift"] = {"status": "reference_available"}

        return result

    def run_post_checks(self, data):
        logger.info("Running post-engineering monitoring checks...")
        result = self._build_data_summary(data)
        if getattr(data, "empty", True):
            result["status"] = "skipped"
            result["reason"] = "empty_data"
            return result

        try:
            result["drift"] = self.drift_monitor.check_drift(data)
        except Exception as exc:
            result["drift"] = {"status": "unavailable", "reason": str(exc)}

        try:
            result["freshness"] = self.freshness_monitor.get_freshness_metrics()
        except Exception as exc:
            result["freshness"] = {"status": "unavailable", "reason": str(exc)}

        try:
            result["regime_importance"] = self.regime_tracker.get_regime_importance_summary()
        except Exception as exc:
            result["regime_importance"] = {"status": "unavailable", "reason": str(exc)}

        return result

    def _build_data_summary(self, data):
        return {
            "status": "ok",
            "rows": int(getattr(data, "shape", (0, 0))[0]),
            "columns": int(getattr(data, "shape", (0, 0))[1]),
        }
