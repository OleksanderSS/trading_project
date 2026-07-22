# src/pipeline/hybrid/final_stages_orchestrator.py
"""
Final Stages Orchestrator for Hybrid Orchestrator.

Orchestrates stages 4-7 (light models + predictions) and finalizes results.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import aiofiles
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class FinalStagesOrchestrator:
    """
    Orchestrates final pipeline stages.

    Handles execution of stages 4-7 and result finalization.
    """

    def __init__(self, config_manager: UnifiedConfigManager, output_dir: Path, batch_name: str):
        self.config_manager = config_manager
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(__name__)

    def _prepare_final_stages_params(self, colab_results: dict[str, Any] | None, batch_name: str | None,
                                    stages_to_run: list[int] | None) -> tuple[str, list[int]]:
        """Prepare and validate parameters for final stages."""
        if colab_results is None:
            colab_results = {}

        batch_name = batch_name or colab_results.get('batch_name', self.batch_name)
        stages_to_run = stages_to_run or [5, 7]

        # Ensure stage 5 is included if stages 6 or 7 are requested
        if 6 in stages_to_run or 7 in stages_to_run:
            stages_to_run = sorted(set(stages_to_run) | {5})

        return batch_name, stages_to_run

    async def _run_stage_4_if_needed(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                    batch_name: str, models_metadata: dict[str, Any],
                                    light_models_runner) -> dict[str, Any]:
        """Run stage 4 (light models) if needed and update metadata."""
        light_results = await light_models_runner(features_df, targets_df, batch_name)

        if light_results.get('status') == 'success':
            models_metadata.update(light_results.get('models_metadata', {}))

        return models_metadata

    async def _run_stages_5_to_7(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                tickers: list[str] | None, timeframes: list[str] | None,
                                batch_name: str, stages_to_run: list[int],
                                models_metadata: dict[str, Any],
                                execution_mode: str = 'review_only',
                                evaluation_notification_authorized: bool = False,
                                news_data: pd.DataFrame | None = None,
                                economic_data: pd.DataFrame | None = None,
                                market_indicators: pd.DataFrame | None = None) -> dict[str, Any]:
        """Run stages 5-7 using PipelineOrchestrator."""
        valid_stages = [s for s in stages_to_run if s in [5, 6, 7]]
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=valid_stages
        )

        return cast(dict[str, Any], await orchestrator.run(
            tickers=tickers,
            timeframes=timeframes,
            run_mode='predict',
            features_data=features_df,
            features_df=features_df,
            targets_df=targets_df,
            models_metadata=models_metadata,
            news_data=news_data,
            economic_data=economic_data,
            market_indicators=market_indicators,
            batch_name=batch_name,
            execution_mode=execution_mode,
            evaluation_notification_authorized=(
                evaluation_notification_authorized
            ),
            stages_to_run=valid_stages
        ))

    def _create_final_summary(
        self,
        results: dict[str, Any],
        tickers: list[str] | None,
        duration_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Create final summary dictionary.

        PipelineOrchestrator merges all stage outputs so keys come directly
        from each stage's return dict:
          Stage 5 → 'predictions', 'prediction_results', 'current_prices'
          Stage 6 → 'trading_activity', 'portfolio_summary', 'signals'
          Stage 7 → 'evaluation_summary'
        """
        # Stage 5: prediction_results is a dict keyed by context_id;
        # 'predictions' is the list form — expose both for consumers.
        prediction_results = (
            results.get('prediction_results')
            or {p['ticker']: p for p in results.get('predictions', []) if p.get('ticker')}
        )

        return {
            'timestamp': datetime.now().isoformat(),
            'tickers': tickers,
            'prediction_results': prediction_results,
            'predictions': results.get('predictions', []),
            'current_prices': results.get('current_prices', {}),
            'trading_summary': results.get('portfolio_summary', {}),
            'trading_activity': results.get('trading_activity', []),
            'execution_status': results.get('execution_status', 'stage_6_not_requested'),
            'execution_boundary': results.get('execution_boundary', {
                'effective_mode': 'review_only',
                'live_execution_supported': False,
                'portfolio_mutated': False,
                'reason': 'Stage 6 was not requested.',
            }),
            'evaluation_summary': results.get('evaluation_summary', {}),
            'duration_seconds': duration_seconds,
        }

    async def _save_final_results(self, final_summary: dict[str, Any]) -> Path:
        """Save final results to JSON file."""
        output_path = self.output_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        content = json.dumps(final_summary, indent=2, default=str)

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        return output_path

    async def run_final_stages(self, request) -> dict[str, Any]:
        """Main entry point for final stages execution."""
        from src.pipeline.hybrid.contracts import HybridFinalStagesRequest

        if not isinstance(request, HybridFinalStagesRequest):
            self.logger.error("Invalid request type for run_final_stages")
            return {'status': 'error', 'message': 'Invalid request type'}

        self.logger.info(f"🏁 Starting final stages for batch: {request.batch_name}")

        started_at = time.time()

        # 1. Build models metadata
        from src.pipeline.hybrid.results_processor import ResultsProcessor
        rp = ResultsProcessor()
        models_metadata = rp.build_models_metadata(request.colab_results or {}, request.light_results)

        # 2. Run prediction and evaluation by default. Stage 6 is explicit.
        _, stages_to_run = self._prepare_final_stages_params(
            request.colab_results,
            request.batch_name,
            request.stages_to_run,
        )
        results = await self._run_stages_5_to_7(
            features_df=request.features_df,
            targets_df=request.targets_df,
            tickers=request.tickers,
            timeframes=request.timeframes or ['15m', '60m', '1d'],
            batch_name=request.batch_name or self.batch_name,
            stages_to_run=stages_to_run,
            models_metadata=models_metadata,
            execution_mode=request.execution_mode,
            evaluation_notification_authorized=(
                request.evaluation_notification_authorized
            ),
            news_data=request.news_data,
            economic_data=request.economic_data,
            market_indicators=request.market_indicators,
        )

        # 3. Create and save summary
        summary = self._create_final_summary(
            results,
            request.tickers,
            duration_seconds=time.time() - started_at,
        )
        saved_path = await self._save_final_results(summary)

        self.logger.info(f"✅ Final results saved to {saved_path}")

        return {
            'status': 'completed',
            'summary': summary,
            'results': results,
            'saved_path': str(saved_path)
        }
