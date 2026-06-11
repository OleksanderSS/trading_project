import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.cli.pipeline_executor import PipelineExecutor


class FullModeOrchestrator:
    def __init__(self):
        self.request = None

    async def run_full_hybrid_pipeline(self, request):
        self.request = request
        return {"status": "paused_for_colab"}


class ContinueModeOrchestrator:
    def __init__(self, batch_dir, colab_results):
        self.config = SimpleNamespace(output_dir=batch_dir)
        self.colab_results = colab_results

    def load_colab_results(self, batch_name):
        return self.colab_results


class ContinueExecutionOrchestrator(ContinueModeOrchestrator):
    def __init__(self, batch_dir, colab_results):
        super().__init__(batch_dir, colab_results)
        self.light_kwargs = None
        self.final_request = None

    async def run_light_models(self, **kwargs):
        self.light_kwargs = kwargs
        return {"models_metadata": {"light_model": {"score": 1.0}}}

    async def run_final_stages(self, request):
        self.final_request = request
        return {"status": "completed"}


def test_execute_full_mode_uses_hybrid_request():
    orchestrator = FullModeOrchestrator()

    result = asyncio.run(PipelineExecutor.execute_full_mode(
        orchestrator,
        tickers=["AMD"],
        timeframes=["15m", "1d"],
    ))

    assert result["status"] == "paused_for_colab"
    assert orchestrator.request.tickers == ["AMD"]
    assert orchestrator.request.timeframes == ["15m", "1d"]
    assert orchestrator.request.accumulate is True


def test_load_continue_data_returns_stable_tuple_on_missing_colab_results():
    args = SimpleNamespace(batch_name="missing_batch")
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = ContinueModeOrchestrator(
            batch_dir=Path(tmp_dir),
            colab_results={"status": "error", "message": "not found"},
        )

        result = PipelineExecutor._load_continue_data(orchestrator, args)

        assert len(result) == 5
        assert result[0] is None
        assert result[2]["status"] == "error"


def test_validate_continue_inputs_rejects_missing_targets():
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"]})
    colab_results = {"status": "success", "models_metadata": {"m1": {"model_path": "model.pkl"}}}

    result = PipelineExecutor._validate_continue_inputs(
        features_df=features_df,
        targets_df=None,
        colab_results=colab_results,
        batch_name="main_database",
    )

    assert result == {"status": "failed", "reason": "missing_targets"}


def test_validate_continue_inputs_rejects_targets_without_target_columns():
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"]})
    targets_df = pd.DataFrame({"volatility_15m": [0.1], "ticker": ["AMD"]})
    colab_results = {"status": "success", "models_metadata": {"m1": {"model_path": "model.pkl"}}}

    result = PipelineExecutor._validate_continue_inputs(
        features_df=features_df,
        targets_df=targets_df,
        colab_results=colab_results,
        batch_name="main_database",
    )

    assert result == {"status": "failed", "reason": "missing_target_columns"}


def test_merge_results_data_initializes_models_metadata():
    merged = PipelineExecutor._merge_results_data(
        colab_results={},
        light_results={"models_metadata": {"light_model": {"score": 1.0}}},
    )

    assert merged["models_metadata"]["light_model"]["score"] == 1.0


def test_execute_continue_mode_trains_light_models_on_loaded_data(monkeypatch):
    features_df = pd.DataFrame({"datetime": ["2026-05-08"], "ticker": ["AMD"], "f1": [1.0]})
    targets_df = pd.DataFrame({"target_return": [0.1], "ticker": ["AMD"]})
    colab_results = {"status": "success", "ticker_results": {"AMD": {}}}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = ContinueExecutionOrchestrator(Path(tmp_dir), colab_results)
        args = SimpleNamespace(
            batch_name="main_database",
            test_ticker=None,
            test_target=None,
            stages=None,
        )

        monkeypatch.setattr(
            PipelineExecutor,
            "_validate_batch_contract",
            staticmethod(lambda _orchestrator: {
                "valid": True,
                "manifest": {"timeframes": ["1d"]},
                "errors": [],
            }),
        )
        monkeypatch.setattr(
            PipelineExecutor,
            "_load_continue_data",
            staticmethod(lambda _orchestrator, _args: (
                features_df,
                targets_df,
                colab_results,
                None,
                None,
            )),
        )

        result = asyncio.run(PipelineExecutor.execute_continue_mode(orchestrator, args))

        assert result["status"] == "completed"
        assert orchestrator.light_kwargs["features_df"].equals(features_df)
        assert orchestrator.light_kwargs["targets_df"].equals(targets_df)
        assert orchestrator.final_request["light_results"]["models_metadata"]["light_model"]["score"] == 1.0
