"""
Hybrid Pipeline Orchestrator:
- Local: Parsing, feature selection, light models
- Colab: Heavy models, heavy analyzers
- State persistence for long-running sessions
"""
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.hybrid.component_factory import OrchestratorComponentFactory
from src.pipeline.hybrid.contracts import HybridFinalStagesRequest, HybridPipelineRequest
from src.pipeline.hybrid.orchestrator_config import OrchestratorConfigManager
from src.pipeline.hybrid.pipeline_config import FinalStagesParams, PipelineParams

logger = ProjectLogger.get_logger(__name__)

class HybridOrchestrator:
    """
    Hybrid orchestrator for distributed pipeline execution.
    Facade for modular components in src.pipeline.hybrid.*
    """

    def __init__(self, config_manager: UnifiedConfigManager, batch_name: str = 'main_database'):
        self.config_manager = config_manager
        self.logger = logger
        self.batch_name = batch_name
        self.orchestrator_config_manager = OrchestratorConfigManager(config_manager)
        self.config = self.orchestrator_config_manager.build_pipeline_config(batch_name)

        # Initialize all sub-components via factory
        OrchestratorComponentFactory.initialize_components(self)

        self.logger.info(f'✅ HybridOrchestrator initialized for batch: {self.batch_name}')

    async def run_local_pipeline(self, tickers: list[str] | None = None,
                               timeframes: list[str] | None = None,
                               stages_to_run: list[int] | None = None) -> dict[str, Any]:
        """Execute local pipeline stages."""
        return await self.pipeline_runner.run_local_pipeline(tickers, timeframes, stages_to_run)

    async def run_full_hybrid_pipeline(self, request: HybridPipelineRequest) -> dict[str, Any]:
        """Run full hybrid pipeline with all parameters."""
        params = PipelineParams(
            tickers=request.tickers,
            timeframes=request.timeframes,
            accumulate=request.accumulate,
            force_training=request.force_training,
            skip_colab=request.skip_colab,
            force_feature_selection=request.force_feature_selection
        )
        return await self.pipeline_manager.run_full_hybrid_pipeline(params)

    async def prepare_colab_data(self, tickers: list[str], timeframes: list[str], **kwargs) -> dict[str, Any]:
        """Delegate data preparation to ColabManager."""
        # Assemble config
        from src.pipeline.hybrid.colab_manager import BatchPreparationConfig
        config = BatchPreparationConfig(
            tickers=tickers,
            timeframes=timeframes,
            batch_name=self.batch_name,
            accumulate=kwargs.get('accumulate', True),
            force_feature_selection=kwargs.get('force_feature_selection', False),
            test_ticker=kwargs.get('test_ticker'),
            test_target=kwargs.get('test_target')
        )
        # This assumes features_df and targets_df are available in kwargs or context
        features_df = kwargs.get('features_df', pd.DataFrame())
        targets_df = kwargs.get('targets_df', pd.DataFrame())

        return self.colab_manager.prepare_colab_batch(features_df, targets_df, config)


    async def run_light_models(
        self,
        tickers: list[str] | None = None,
        test_ticker: str | None = None,
        test_target: str | None = None,
        features_df: pd.DataFrame | None = None,
        targets_df: pd.DataFrame | None = None,
        timeframes: list[str] | None = None,
        batch_name: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Run local light-model training on prepared feature/target data."""
        effective_tickers = [test_ticker] if test_ticker else tickers

        if features_df is None or targets_df is None:
            local_result = await self.run_local_pipeline(
                tickers=effective_tickers,
                timeframes=timeframes,
            )
            stage_outputs = local_result.get("results", {}) if local_result else {}
            features_df = stage_outputs.get("features_df")
            targets_df = stage_outputs.get("targets_df")

        if features_df is None or getattr(features_df, "empty", True):
            return {"status": "failed", "reason": "missing_features"}
        if targets_df is None or getattr(targets_df, "empty", True):
            return {"status": "failed", "reason": "missing_targets"}

        return await self.light_models_trainer.run_light_models(
            features_df,
            targets_df,
            effective_tickers,
        )

    async def run_final_stages(self, request: HybridFinalStagesRequest | dict[str, Any]) -> dict[str, Any]:
        """Run final stages from a request object or CLI request dictionary."""
        if isinstance(request, dict):
            request = HybridFinalStagesRequest(**request)

        params = FinalStagesParams(
            features_df=request.features_df,
            targets_df=request.targets_df,
            colab_results=request.colab_results,
            light_results=request.light_results,
            tickers=request.tickers,
            timeframes=request.timeframes,
            batch_name=request.batch_name or self.batch_name,
        )

        return await self.pipeline_manager.run_final_stages(params)

    def load_colab_results(self, batch_name: str) -> dict[str, Any]:
        """Load training results from Colab batch."""
        return self.colab_manager.load_colab_results(batch_name)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of pipeline state."""
        return {
            'batch_name': self.batch_name,
            'output_dir': str(self.config.output_dir),
            'light_models': self.config.light_models,
            'heavy_models': self.config.heavy_models
        }
