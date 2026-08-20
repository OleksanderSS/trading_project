import asyncio

import pandas as pd
import pytest

import src.pipeline.hybrid.final_stages_orchestrator as final_stages_module
from src.pipeline.hybrid.final_stages_orchestrator import FinalStagesOrchestrator
from src.pipeline.stages.stage_6_trading_execution import TradingExecutionStage
from src.pipeline.stages.stage_7_evaluation import EvaluationStage
from src.trading.trader import Trader


class _Logger:
    """Stand-in for ProjectLogger while the stateful stack is not initialised.

    Carries every level the real logger has. An incomplete double turns a
    perfectly ordinary logging call into an AttributeError in production code,
    which is a failure of the test rather than of the code under test — it
    happened on 2026-08-20 when the context gate moved its per-signal messages
    from warning to debug.
    """

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
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


def test_final_stages_default_includes_stage6():
    """Stage 6 is in the default set, and its safety does not depend on that.

    It used to be excluded, and the run then reported
    `execution_status: stage_6_not_requested` -- which reads as an
    operator's decision when it was a literal in
    _prepare_final_stages_params. Three consecutive runs were summarised as
    "trading simulation performed" on that basis while Stage 6 had never
    executed, and every financial number came from Stage 7's backtest.

    The exclusion was not a safety measure. It was recorded by a bulk
    coverage commit (794f518d) that documented what the code did, with no
    rationale given, and it protects nothing: the guarantee lives INSIDE the
    stage, which defaults to execution_mode='review_only' and blocks even
    paper execution (see the two tests above). Excluding it from the default
    did not make execution safer, it made a stage invisible.
    """
    orchestrator = object.__new__(FinalStagesOrchestrator)
    orchestrator.batch_name = "batch"

    _, default_stages = orchestrator._prepare_final_stages_params(None, None, None)
    _, explicit_stages = orchestrator._prepare_final_stages_params(
        None, None, [6, 7]
    )

    assert default_stages == [5, 6, 7]
    assert explicit_stages == [5, 6, 7]


def test_an_explicit_narrower_request_is_still_honoured():
    """Including 6 by default must not mean forcing it: a caller asking for
    5 and 7 alone still gets exactly that."""
    orchestrator = object.__new__(FinalStagesOrchestrator)
    orchestrator.batch_name = "batch"

    _, stages = orchestrator._prepare_final_stages_params(None, None, [5, 7])

    assert stages == [5, 7]


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
