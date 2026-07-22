# audit-ignore: ARCHITECTURAL_USAGE
"""
Pipeline Manager for Hybrid Orchestrator.
Handles pipeline execution and coordination.
"""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.hybrid.colab_manager import BatchPreparationConfig
from src.pipeline.hybrid.contracts import HybridFinalStagesRequest

from .pipeline_cache_checker import PipelineCacheChecker
from .pipeline_config import FinalStagesParams, PipelineParams


class PipelineManager:
    """Manages pipeline execution for hybrid orchestrator."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = ProjectLogger.get_logger(__name__)
        self.cache_checker = PipelineCacheChecker(self.orchestrator)

    async def run_full_hybrid_pipeline(self, params: PipelineParams | None = None) -> dict[str, Any]:
        """Full hybrid pipeline with smart caching logic."""
        if params is None:
            params = PipelineParams()

        self.logger.info(f"Launching full hybrid pipeline for batch: {self.orchestrator.batch_name}")

        # Step 1: Check cache before running pipeline
        cached_data = self.cache_checker.check_cache_before_run(params.force_training)
        if cached_data is not None:
            self.logger.info("✅ Using cached data - skipping pipeline stages 0-3")
            features_df, targets_df = cached_data
        else:
            self.logger.info("🔄 No valid cache found - running pipeline stages 0-3")
            # Step 2: Collect local data
            local_res = await self._collect_local_data(params.tickers, params.timeframes)
            if local_res['status'] != 'local_complete':
                return local_res

            # Step 3: Check cache and handle data
            features_df, targets_df = self._handle_data_caching(local_res, params.force_training)
            if features_df is None or targets_df is None:
                return {'status': 'no_data', 'message': 'No data collected'}

        # Step 4: Prepare Colab package
        self.logger.info("Preparing Colab package...")
        colab_config = BatchPreparationConfig(
            tickers=params.tickers or [],
            timeframes=params.timeframes or [],
            batch_name=self.orchestrator.batch_name,
            accumulate=params.accumulate,
            force_feature_selection=params.force_feature_selection,
        )
        b_info = self.orchestrator.colab_manager.prepare_colab_batch(
            features_df,
            targets_df,
            colab_config,
        )

        # Step 5: Handle Colab or skip path
        if params.skip_colab:
            return await self._handle_skip_colab_path(
                b_info,
                features_df,
                targets_df,
                params.tickers,
                params.timeframes,
            )
        else:
            return self._handle_colab_path(b_info)

    async def run_final_stages(self, params: FinalStagesParams | None = None) -> dict[str, Any]:
        """Run final stages 4-7 of pipeline."""
        if params is None:
            params = FinalStagesParams()

        request = HybridFinalStagesRequest(
            features_df=params.features_df,
            targets_df=params.targets_df,
            colab_results=params.colab_results,
            light_results=params.light_results,
            tickers=params.tickers,
            timeframes=params.timeframes,
            batch_name=params.batch_name or self.orchestrator.batch_name,
            stages_to_run=params.stages_to_run,
            execution_mode=params.execution_mode,
            evaluation_notification_authorized=(
                params.evaluation_notification_authorized
            ),
        )

        return cast(dict[str, Any], await self.orchestrator.final_stages_orchestrator.run_final_stages(request))

    async def _collect_local_data(self, tickers: list[str] | None,
                                  timeframes: list[str] | None) -> dict[str, Any]:
        """Collect local pipeline data."""
        self.logger.info("Collecting new data...")
        return cast(dict[str, Any], await self.orchestrator.run_local_pipeline(tickers, timeframes))

    def _handle_data_caching(self, local_res: dict[str, Any], force_training: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Handle data caching logic."""
        cache_manager = getattr(self.orchestrator, "data_cache_manager", None)
        if cache_manager is None or not hasattr(cache_manager, "handle_data_caching"):
            raise AttributeError("Hybrid orchestrator has no data_cache_manager.handle_data_caching")

        return cast(tuple[pd.DataFrame | None, pd.DataFrame | None], cache_manager.handle_data_caching(
            local_res, force_training, self.orchestrator.batch_name, self.orchestrator.config.output_dir
        ))

    async def _handle_skip_colab_path(self, b_info: dict[str, Any],
                                     features_df: pd.DataFrame,
                                     targets_df: pd.DataFrame,
                                     tickers: list[str] | None,
                                     timeframes: list[str] | None) -> dict[str, Any]:
        """Handle skip Colab path."""
        self._create_fallback_selected_features(b_info, features_df)
        final_results = await self.run_final_stages(FinalStagesParams(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            timeframes=timeframes,
            batch_name=self.orchestrator.batch_name,
        ))
        return {'status': 'completed_without_colab', 'final_results': final_results}

    def _handle_colab_path(self, b_info: dict[str, Any]) -> dict[str, Any]:
        """Handle Colab training path."""
        instr = self._generate_colab_instructions(b_info)
        self.logger.info(f"PAUSED: Colab training required.\n{instr}")
        return {'status': 'paused_for_colab', 'colab_batch': b_info, 'colab_instructions': instr}

    def _create_fallback_selected_features(self, batch_info: dict[str, Any], features_df: pd.DataFrame):
        """Create fallback selected features when skipping Colab."""
        batch_dir = Path(batch_info['batch_dir'])
        batch_dir.mkdir(parents=True, exist_ok=True)
        selected_features = [
            col for col in features_df.columns
            if not str(col).startswith("target_")
        ]

        # Save fallback features
        import json
        features_file = batch_dir / 'selected_features_fallback.json'
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump({'features': selected_features, 'method': 'fallback'}, f, indent=2, default=str)

    def _generate_colab_instructions(self, batch_info: dict[str, Any]) -> str:
        """Generates instructions for running in Colab."""
        name = batch_info['batch_name']
        return f"""
COLAB INSTRUCTIONS:
1. Transfer the batch folder '{name}' to your Google Drive.
2. Run the Colab notebook and mount your drive.
3. Perform feature selection and heavy model training.
4. Once finished, run: python run_hybrid_pipeline.py --mode continue --batch-name {name}
"""
