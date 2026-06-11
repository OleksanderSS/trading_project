"""
Final Stages Executor - Handles final stages execution
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

logger = ProjectLogger.get_logger(__name__)


class FinalStagesExecutor:
    """Handles execution of final pipeline stages."""

    def __init__(self, config_manager, output_dir: str, batch_name: str):
        self.config_manager = config_manager
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(__name__)

    async def run_final_stages(self, features_df, targets_df, colab_results: dict[str, Any] | None = None,
                              light_results: dict[str, Any] | None = None, tickers: list[str] | None = None,
                              timeframes: list[str] | None = None, batch_name: str | None = None) -> dict[str, Any]:
        """Runs final stages 4-7 after Colab results are loaded."""
        batch_name, stages_to_run = self._prepare_final_stages_params(colab_results, batch_name, [4, 5, 6, 7])

        self.logger.info(f"Running final stages {stages_to_run} for batch: {batch_name}")

        # Heavy models must come from Colab - bail early with a clear message if missing
        if not colab_results or not self._has_heavy_models(colab_results):
            self.logger.error("Heavy model metadata is missing. Run Colab training before final stages.")
            return {
                'status': 'failed',
                'reason': 'missing_heavy_model_results',
                'message': 'Run Colab training and place the returned results in the batch directory.'
            }

        heavy_results = colab_results

        # Merge light and heavy results
        all_results = self._merge_model_results(light_results, heavy_results)

        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=stages_to_run
        )

        start_time = time.time()

        results = await orchestrator.run(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            timeframes=timeframes,
            run_mode='train',
            colab_results=all_results,  # 🔧 FIXED: Pass all models
            light_results=light_results
        )

        duration = time.time() - start_time

        # Build models metadata
        models_metadata = self._build_models_metadata(all_results, light_results)

        # Create final summary
        final_summary = self._create_final_summary(results, models_metadata, duration, tickers)

        return {
            'results': results,
            'models_metadata': models_metadata,
            'final_summary': final_summary,
            'duration': duration,
            'heavy_models_trained': not colab_results or not self._has_heavy_models(colab_results)
        }

    def _has_heavy_models(self, colab_results) -> bool:
        """Return True if colab_results contains at least one recognised heavy-model entry."""
        if not colab_results:
            return False

        heavy_model_names = {'cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder', 'mlp'}

        # 1. Flat models_metadata dict (structure produced by colab_clean_cell.py)
        #    Keys look like "AAPL_target_return_lstm"
        models_metadata = colab_results.get('models_metadata', {})
        if models_metadata:
            for key, meta in models_metadata.items():
                # Check suffix of the composite key
                if key.rsplit('_', 1)[-1].lower() in heavy_model_names:
                    return True
                # Check explicit model_type field
                if isinstance(meta, dict) and meta.get('model_type', '').lower() in heavy_model_names:
                    return True

        # 2. Nested ticker_results dict (legacy structure)
        ticker_results = colab_results.get('ticker_results', {})
        first_ticker = next(iter(ticker_results), None)
        if not first_ticker:
            return False

        timeframes = ticker_results[first_ticker].get('timeframes', {})
        all_results = timeframes.get('all', {}).get('results', {})
        for target_data in all_results.values():
            if any(m in heavy_model_names for m in target_data.get('models', {})):
                return True

        return False

    async def _train_heavy_models(self, features_df, targets_df, tickers):
        """Heavy models must be trained in Colab — not supported locally."""
        self.logger.error(
            "_train_heavy_models called locally. Heavy models (LSTM, GRU, Transformer, TabNet, etc.) "
            "must be trained in Google Colab. "
            "Workflow: python run_hybrid_pipeline.py --mode prepare "
            "→ train in Colab → python run_hybrid_pipeline.py --mode continue"
        )
        raise RuntimeError(
            "Heavy model training is not supported locally. "
            "Use --mode prepare → Colab → --mode continue workflow."
        )

    def _merge_model_results(self, light_results, heavy_results):
        """Об'єднує результати light та heavy моделей"""
        if not light_results:
            return heavy_results
        if not heavy_results:
            return light_results

        # Просте об'єднання - в реальності потрібна складніша логіка
        merged = light_results.copy()

        if 'ticker_results' in heavy_results:
            if 'ticker_results' not in merged:
                merged['ticker_results'] = {}

            for ticker, ticker_data in heavy_results['ticker_results'].items():
                if ticker not in merged['ticker_results']:
                    merged['ticker_results'][ticker] = ticker_data
                else:
                    # Об'єднуємо існуючі дані
                    existing = merged['ticker_results'][ticker]
                    if 'timeframes' in existing and 'timeframes' in ticker_data:
                        if 'all' in existing['timeframes'] and 'all' in ticker_data['timeframes']:
                            if 'results' in existing['timeframes']['all'] and 'results' in ticker_data['timeframes']['all']:
                                for target, target_data in ticker_data['timeframes']['all']['results'].items():
                                    if target not in existing['timeframes']['all']['results']:
                                        existing['timeframes']['all']['results'][target] = target_data
                                    else:
                                        # Об'єднуємо моделі
                                        existing_models = existing['timeframes']['all']['results'][target].get('models', {})
                                        new_models = target_data.get('models', {})
                                        existing_models.update(new_models)

        return merged

    def _prepare_final_stages_params(self, colab_results: dict[str, Any] | None, batch_name: str | None,
                                    stages_to_run: list[int] | None) -> tuple[str, list[int]]:
        """Prepare and validate parameters for final stages."""
        if colab_results is None:
            colab_results = {}

        batch_name = batch_name or colab_results.get('batch_name', self.batch_name)
        stages_to_run = stages_to_run or [5, 6, 7]

        # Ensure stage 5 is included if stages 6 or 7 are requested
        if 6 in stages_to_run or 7 in stages_to_run:
            stages_to_run = sorted(set(stages_to_run) | {5})

        return batch_name, stages_to_run

    def _build_models_metadata(self, colab_results: dict[str, Any],
                              light_results: dict[str, Any] | None) -> dict[str, Any]:
        """Build comprehensive models metadata from all sources."""
        models_metadata = {}

        # Add Colab heavy models metadata
        if colab_results and 'models_metadata' in colab_results:
            models_metadata.update(colab_results['models_metadata'])

        # Add light models metadata
        if light_results and 'models_metadata' in light_results:
            models_metadata.update(light_results['models_metadata'])

        # Add metadata from accumulated results
        accumulated_results_path = Path(self.output_dir) / "light_models_results.json"
        if accumulated_results_path.exists():
            try:
                with open(accumulated_results_path, encoding='utf-8') as f:
                    accumulated = json.load(f)

                if 'runs' in accumulated:
                    for run in accumulated['runs']:
                        if 'models_metadata' in run:
                            models_metadata.update(run['models_metadata'])
            except Exception as e:
                self.logger.warning(f"Could not load accumulated results: {e}")

        return models_metadata

    def _create_final_summary(self, results: dict[str, Any], models_metadata: dict[str, Any],
                             duration: float, tickers: list[str] | None) -> dict[str, Any]:
        """Create final summary of pipeline execution."""
        return {
            'timestamp': datetime.now().isoformat(),
            'batch_name': self.batch_name,
            'tickers': tickers or [],
            'models_trained': list(models_metadata.keys()),
            'models_count': len(models_metadata),
            'pipeline_results': results,
            'duration_seconds': duration,
            'status': 'completed'
        }

    async def _save_final_results(self, final_summary: dict[str, Any]) -> Path:
        """Save final results to JSON file."""
        import aiofiles

        output_path = Path(self.output_dir) / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        content = json.dumps(final_summary, indent=2, default=str)

        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)

        return output_path
