import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.pipeline.hybrid.colab_manager import BatchPreparationConfig
from src.pipeline.hybrid.contracts import HybridFinalStagesRequest
from src.pipeline.hybrid.pipeline_config import FinalStagesParams, PipelineParams
from src.pipeline.hybrid.pipeline_manager import PipelineManager


class DummyDataCacheManager:
    def __init__(self):
        self.received_output_dir = None

    def handle_data_caching(self, local_res, force_training, batch_name, output_dir):
        self.received_output_dir = output_dir
        results = local_res["results"]
        return results["features_df"], results["targets_df"]


class DummyColabManager:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.received_config = None

    def prepare_colab_batch(self, features_df, targets_df, config):
        self.received_config = config
        return {
            "batch_dir": str(self.tmp_path),
            "batch_name": config.batch_name,
            "features_shape": features_df.shape,
            "targets_shape": targets_df.shape,
        }


class DummyFinalStagesOrchestrator:
    def __init__(self):
        self.request = None

    async def run_final_stages(self, request):
        self.request = request
        return {"status": "completed"}


class DummyHybridOrchestrator:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.batch_name = "main_database"
        self.config = SimpleNamespace(output_dir=tmp_path)
        self.data_cache_manager = DummyDataCacheManager()
        self.colab_manager = DummyColabManager(tmp_path)
        self.final_stages_orchestrator = DummyFinalStagesOrchestrator()

    async def run_local_pipeline(self, tickers, timeframes):
        return {
            "status": "local_complete",
            "results": {
                "features_df": pd.DataFrame({"ticker": ["AMD"], "f1": [1.0]}),
                "targets_df": pd.DataFrame({"target_return": [0.1]}),
            },
        }


def test_run_full_hybrid_pipeline_uses_data_cache_and_batch_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        orchestrator = DummyHybridOrchestrator(tmp_path)
        manager = PipelineManager(orchestrator)

        result = asyncio.run(manager.run_full_hybrid_pipeline(PipelineParams(
            tickers=["AMD"],
            timeframes=["1d"],
        )))

        assert result["status"] == "paused_for_colab"
        assert orchestrator.data_cache_manager.received_output_dir == tmp_path
        assert isinstance(orchestrator.colab_manager.received_config, BatchPreparationConfig)
        assert orchestrator.colab_manager.received_config.tickers == ["AMD"]
        assert orchestrator.colab_manager.received_config.timeframes == ["1d"]


def test_run_final_stages_uses_shared_contract_request():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        orchestrator = DummyHybridOrchestrator(tmp_path)
        manager = PipelineManager(orchestrator)

        result = asyncio.run(manager.run_final_stages(FinalStagesParams(
            features_df=pd.DataFrame({"f1": [1.0]}),
            targets_df=pd.DataFrame({"target_return": [0.1]}),
            tickers=["AMD"],
            batch_name="main_database",
            stages_to_run=[5, 6, 7],
            execution_mode="paper",
            evaluation_notification_authorized=True,
        )))

        assert result["status"] == "completed"
        assert isinstance(orchestrator.final_stages_orchestrator.request, HybridFinalStagesRequest)
        assert orchestrator.final_stages_orchestrator.request.tickers == ["AMD"]
        assert orchestrator.final_stages_orchestrator.request.stages_to_run == [5, 6, 7]
        assert orchestrator.final_stages_orchestrator.request.execution_mode == "paper"
        assert (
            orchestrator.final_stages_orchestrator.request
            .evaluation_notification_authorized
            is True
        )
