import pandas as pd
import pytest

from src.dashboard.dashboard_data_bridge import DashboardDataBridge


class FakeDataManager:
    def __init__(self, result):
        self.result = result
        self.last_query = None
        self.last_params = None

    def fetch_df(self, query, params=None):
        self.last_query = query
        self.last_params = params
        return self.result


def make_bridge(result):
    bridge = DashboardDataBridge.__new__(DashboardDataBridge)
    bridge.logger = None
    bridge.config_manager = None
    bridge.error_handler = None
    bridge._data_cache = {}
    bridge._cache_timestamps = {}
    bridge._cache_ttl = 300
    bridge.data_manager = FakeDataManager(result)
    return bridge


def test_dashboard_bridge_uses_fetch_df_for_database_results():
    bridge = make_bridge(pd.DataFrame([{
        "model_name": "model_a",
        "model_type": "Tree",
        "avg_win_rate": 0.6,
        "avg_sharpe_ratio": 1.2,
        "avg_precision": 0.7,
        "total_trades": 10,
    }]))

    result = bridge._get_model_performance_data()

    assert result["is_sample_data"] is False
    assert result["data_source"] == "database"
    assert result["models"][0]["model_name"] == "model_a"


def test_dashboard_bridge_marks_sample_fallbacks_explicitly():
    bridge = make_bridge(pd.DataFrame())

    result = bridge._get_portfolio_metrics_data()

    assert result["is_sample_data"] is True
    assert result["data_source"] == "sample"


def test_dashboard_market_query_uses_parameters():
    bridge = make_bridge(pd.DataFrame())

    bridge._get_market_data(ticker="AMD")

    assert bridge.data_manager.last_params == ["AMD"]
    assert "WHERE ticker = ?" in bridge.data_manager.last_query
