import asyncio

import pandas as pd

import logging

from src.pipeline.stages.stage_7_evaluation import EvaluationStage


class _BacktestAnalyzer:
    async def run_backtest(self, signals_df):
        return {
            "portfolio_history": pd.DataFrame(
                {"total_value": [100.0, 101.0]},
                index=pd.date_range("2026-01-01", periods=2, tz="UTC"),
            )
        }


class _Metrics:
    def calculate_financial_metrics(self, portfolio_history):
        return {"total_return_pct": 1.0, "max_drawdown_pct": 0.0}


class _ReportGenerator:
    def create_evaluation_summary(
        self, financial_metrics, backtest_results, analysis_results, signals_df
    ):
        return {"metrics": financial_metrics}

    def save_summary(self, summary, results_dir):
        return None

    def plot_equity_curve(self, portfolio_history, financial_metrics):
        return None

    def generate_notification_message(self, financial_metrics):
        return "evaluation"


class _Analytics:
    def __init__(self):
        self.data_map = None
        self.data_maps = []

    def run_full_analysis(self, data_map, *, timeout=None, **kwargs):
        # Mirrors the real signature, timeout included. The double accepted
        # only data_map, so when Stage 7 began passing a per-context budget
        # this raised TypeError -- and the stage's own handler then failed on
        # a missing logger, reporting the AttributeError instead. A double
        # that drifts from its original tests the double.
        self.data_map = data_map
        self.data_maps.append(data_map)
        self.timeout = timeout
        return {}


class _Notifier:
    def __init__(self):
        self.calls = 0

    async def send_report(self, message, image_path=None):
        self.calls += 1


class _Config:
    def get(self, key, default=None):
        return default


def _stage() -> EvaluationStage:
    stage = object.__new__(EvaluationStage)
    # Without this the stage's exception handler raises AttributeError and
    # buries whatever actually went wrong.
    stage.logger = logging.getLogger("stage7-test")
    stage.config_manager = _Config()
    stage.backtest_analyzer = _BacktestAnalyzer()
    stage.metrics_calc = _Metrics()
    stage.report_gen = _ReportGenerator()
    stage.analytics_engine = _Analytics()
    stage.notifier = _Notifier()
    stage.results_dir = None
    stage._write_pipeline_control_evaluation_candidate = (
        lambda **kwargs: {}
    )
    return stage


def _signals() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "price": [100.0, 101.0],
        "signal": ["BUY", "HOLD"],
    })


def test_stage7_turns_trading_activity_into_review_candidate_not_learning():
    stage = _stage()

    result = asyncio.run(stage._run_comprehensive_evaluation(
        _signals(),
        {
            "trading_activity": [{"ticker": "NVDA", "profit_loss": 1.0}],
            "portfolio_summary": {},
            "notification_authorized": False,
        },
    ))

    summary = result["evaluation_summary"]
    candidate = summary["learning_review_candidate"]
    assert candidate["status"] == "proposal_only_pending_dean_os_review"
    assert candidate["observed_trade_count"] == 1
    assert candidate["learning_applied"] is False
    assert candidate["learning_memory_written"] is False
    assert summary["notification_status"] == "review_only_not_sent"
    assert stage.notifier.calls == 0
    assert not hasattr(stage, "real_time_learning")


def test_stage7_notification_requires_explicit_per_run_authorization():
    stage = _stage()

    result = asyncio.run(stage._run_comprehensive_evaluation(
        _signals(),
        {
            "trading_activity": [],
            "portfolio_summary": {},
            "notification_authorized": True,
        },
    ))

    assert result["evaluation_summary"]["notification_status"] == (
        "authorized_delivery_attempted"
    )
    assert stage.notifier.calls == 1


def test_stage7_routes_available_inputs_and_marks_analysis_supporting_only():
    stage = _stage()
    signals = pd.DataFrame({
        "ticker": ["NVDA", "NVDA"],
        "price": [100.0, 101.0],
        "signal": ["BUY", "HOLD"],
    })
    features = pd.DataFrame({"feature_a": [1.0, 2.0]})
    portfolio = pd.DataFrame({"total_value": [100.0, 101.0]})

    result = stage._run_deep_analysis(
        signals,
        portfolio,
        {"features_data": features},
    )

    routed = stage.analytics_engine.data_map
    assert routed["price_data"]["close"].tolist() == [100.0, 101.0]
    assert routed["features_data"] is features
    assert routed["portfolio_data"] is portfolio
    assert routed["signals"].tolist() == ["BUY", "HOLD"]
    assert result["_stage7_analysis_contract"]["price_data_source"] == (
        "derived_from_stage5_signals"
    )
    assert result["_stage7_analysis_contract"]["can_promote_model"] is False
    assert result["_stage7_analysis_contract"]["can_trade"] is False


def test_stage7_partitions_feature_prices_by_market_context():
    stage = _stage()
    features = pd.DataFrame({
        "ticker": ["NVDA", "NVDA", "MSFT", "MSFT"],
        "interval": ["15m", "15m", "15m", "15m"],
        "close": [100.0, 101.0, 200.0, 202.0],
        "timestamp": pd.to_datetime([
            "2026-06-29T12:00:00Z",
            "2026-06-29T12:15:00Z",
            "2026-06-29T12:00:00Z",
            "2026-06-29T12:15:00Z",
        ]),
    })
    portfolio = pd.DataFrame({"total_value": [100.0, 101.0]})

    result = stage._run_deep_analysis(
        _signals(),
        portfolio,
        {"features_data": features},
    )

    routed_tickers = {
        data_map["price_data"]["ticker"].iloc[0]
        for data_map in stage.analytics_engine.data_maps
    }
    coverage = result["_analysis_coverage"]
    contract = result["_stage7_analysis_contract"]

    assert routed_tickers == {"MSFT", "NVDA"}
    assert coverage["context_count"] == 2
    assert sorted(result["analysis_by_context"]) == [
        "ticker=MSFT|interval=15m",
        "ticker=NVDA|interval=15m",
    ]
    assert contract["price_data_source"] == "derived_from_features_data"
    assert contract["price_context_partitioned"] is True
    assert contract["can_trade"] is False
    nvda_window = result["analysis_by_context"][
        "ticker=NVDA|interval=15m"
    ]["_stage7_context_window"]
    assert nvda_window == {
        "row_count": 2,
        "start": "2026-06-29T12:00:00+00:00",
        "end": "2026-06-29T12:15:00+00:00",
        "timestamp_source": "timestamp",
    }
