import pandas as pd
import pytest

from src.monitoring.health_hub import HealthHub


@pytest.fixture
def health_hub():
    return HealthHub(data_manager=None)


def test_extract_features_from_metrics_reads_real_disk_usage_shape(health_hub):
    """extract_features_from_metrics previously read metrics['system']['disk'],
    but the real producer (ResourceMonitor.get_health_status) nests disk under
    a top-level 'disk' key as disk['usage']['percent'], not under 'system'."""
    metrics = {
        "system": {"cpu": {"percent": 10.0}, "memory": {"percent": 20.0}},
        "disk": {"usage": {"percent": 30.0}},
    }

    features = health_hub.extract_features_from_metrics(metrics)

    assert features[0] == pytest.approx(0.10)
    assert features[1] == pytest.approx(0.20)
    assert features[2] == pytest.approx(0.30)


class FakeDataManager:
    def __init__(self, df):
        self._df = df
        self.last_query = None
        self.last_params = None

    def fetch_df(self, query, params=None):
        self.last_query = query
        self.last_params = params
        return self._df

    def execute_query(self, query, params=None):
        return None


def test_load_performance_data_uses_real_fetch_df_method():
    """_load_performance_data previously called data_manager.load_data(...)
    then a query_data(...) fallback - neither method exists on the real
    DataManager (only fetch_df does), so this always raised AttributeError."""
    df = pd.DataFrame({
        "model_name": ["m1"] * 15,
        "win_rate": [0.5] * 15,
        "sharpe_ratio": [1.0] * 15,
        "timestamp": pd.date_range("2026-01-01", periods=15),
    })
    fake_dm = FakeDataManager(df)
    hub = HealthHub(data_manager=fake_dm)

    result = hub._load_performance_data("m1")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 15
    assert fake_dm.last_params == ["m1"]


def test_load_performance_data_reports_insufficient_data():
    df = pd.DataFrame({"model_name": ["m1"], "timestamp": [pd.Timestamp("2026-01-01")]})
    fake_dm = FakeDataManager(df)
    hub = HealthHub(data_manager=fake_dm)

    result = hub._load_performance_data("m1")

    assert result == {"status": "insufficient_data", "message": "Threshold for historical comparison not met"}
