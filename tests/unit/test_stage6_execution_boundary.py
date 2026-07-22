import asyncio

import pandas as pd
import pytest

import src.pipeline.hybrid.final_stages_orchestrator as final_stages_module
from src.pipeline.hybrid.final_stages_orchestrator import FinalStagesOrchestrator
from src.pipeline.stages.stage_6_trading_execution import TradingExecutionStage
from src.pipeline.stages.stage_7_evaluation import EvaluationStage
from src.trading.trader import Trader


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _bare_stage() -> TradingExecutionStage:
    stage = object.__new__(TradingExecutionStage)
    stage.logger = _Logger()
    stage._trading_stack_initialized = False
    return stage


def _prediction() -> dict:
    return {
        "ticker": "NVDA",
        "predictions": [0.03],
        "confidence": 0.8,
        "context_velocity": 0.9,
    }


def test_stage6_defaults_to_review_only_without_initializing_stateful_stack():
    stage = _bare_stage()
    prediction = _prediction()

    result = asyncio.run(stage.run(
        predictions=[prediction],
        current_prices={"NVDA": 100.0},
    ))

    assert result["execution_status"] == "review_only_no_execution"
    assert result["execution_authorized"] is False
    assert result["trading_activity"] == []
    assert result["diary_records_written"] == 0
    assert result["execution_boundary"]["portfolio_mutated"] is False
    assert not hasattr(stage, "portfolio")
    assert prediction["confidence"] == 0.8
    assert result["signals"][0]["confidence"] == 0.0


def test_stage6_blocks_live_and_paper_without_initializing_stack():
    for kwargs, expected_status in [
        ({"execution_mode": "live"}, "blocked_live_execution_disabled"),
        (
            {"execution_mode": "paper", "paper_execution_authorized": True},
            "blocked_paper_execution_requires_isolated_executor",
        ),
    ]:
        stage = _bare_stage()
        result = asyncio.run(stage.run(
            predictions=[_prediction()],
            current_prices={"NVDA": 100.0},
            **kwargs,
        ))

        assert result["execution_status"] == expected_status
        assert result["execution_authorized"] is False
        assert not hasattr(stage, "portfolio")


def test_final_stages_default_excludes_stage6_but_explicit_request_keeps_it():
    orchestrator = object.__new__(FinalStagesOrchestrator)
    orchestrator.batch_name = "batch"

    _, default_stages = orchestrator._prepare_final_stages_params(None, None, None)
    _, explicit_stages = orchestrator._prepare_final_stages_params(
        None, None, [6, 7]
    )

    assert default_stages == [5, 7]
    assert explicit_stages == [5, 6, 7]


def test_stage7_accepts_stage5_predictions_when_stage6_is_skipped():
    stage = object.__new__(EvaluationStage)
    stage.logger = _Logger()
    predictions = [_prediction()]

    loaded = stage._load_signals_data(predictions=predictions)

    assert loaded["signals"] is predictions
    assert loaded["trading_activity"] == []
    assert loaded["portfolio_summary"] == {}


def test_final_stages_pass_optional_analysis_inputs_to_stage7(monkeypatch):
    captured = {}

    class _PipelineOrchestrator:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def run(self, **kwargs):
            captured["run"] = kwargs
            return {}

    monkeypatch.setattr(
        final_stages_module,
        "PipelineOrchestrator",
        _PipelineOrchestrator,
    )
    orchestrator = object.__new__(FinalStagesOrchestrator)
    orchestrator.config_manager = object()
    features = pd.DataFrame({"close": [100.0]})
    news = pd.DataFrame({"headline": ["example"]})
    economic = pd.DataFrame({"value": [1.0]})
    indicators = pd.DataFrame({"vix": [15.0]})

    asyncio.run(orchestrator._run_stages_5_to_7(
        features_df=features,
        targets_df=pd.DataFrame({"target": [0.1]}),
        tickers=["NVDA"],
        timeframes=["15m"],
        batch_name="batch",
        stages_to_run=[5, 7],
        models_metadata={},
        news_data=news,
        economic_data=economic,
        market_indicators=indicators,
    ))

    assert captured["run"]["features_data"] is features
    assert captured["run"]["news_data"] is news
    assert captured["run"]["economic_data"] is economic
    assert captured["run"]["market_indicators"] is indicators


def test_trader_rejects_live_mode_during_initialization():
    with pytest.raises(ValueError, match="Live trading is intentionally disabled"):
        Trader(paper_trading=False)
