import pandas as pd

from src.pipeline.stages.monitoring.feature_monitoring import FeatureEngineeringMonitor


class FakeDriftMonitor:
    def __init__(self):
        self.reference_data = None

    def set_reference_data(self, data):
        self.reference_data = data.copy()

    def check_drift(self, data):
        return {"status": "OK", "total_features": len(data.columns)}


class FakeFreshnessMonitor:
    def get_freshness_metrics(self):
        return {"total_checks": 0}


class FakeRegimeTracker:
    def get_regime_importance_summary(self):
        return {"error": "No recent regime importance data available"}


def make_monitor():
    monitor = FeatureEngineeringMonitor.__new__(FeatureEngineeringMonitor)
    monitor.drift_monitor = FakeDriftMonitor()
    monitor.freshness_monitor = FakeFreshnessMonitor()
    monitor.regime_tracker = FakeRegimeTracker()
    return monitor


def test_pre_checks_initialize_drift_reference():
    monitor = make_monitor()
    data = pd.DataFrame({"feature": [1.0, 2.0]})

    result = monitor.run_pre_checks(data)

    assert result["status"] == "ok"
    assert result["rows"] == 2
    assert result["drift"]["status"] == "reference_initialized"
    assert monitor.drift_monitor.reference_data.equals(data)


def test_post_checks_return_monitoring_results():
    monitor = make_monitor()
    data = pd.DataFrame({"feature": [1.0, 2.0]})

    result = monitor.run_post_checks(data)

    assert result["status"] == "ok"
    assert result["drift"]["status"] == "OK"
    assert result["freshness"]["total_checks"] == 0
    assert "regime_importance" in result
