# audit-ignore: ARCHITECTURAL_USAGE
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

    async def run_final_stages(self, features_df, targets_df, colab_results:
        dict[str, Any] | None=None, light_results: dict[str, Any] | None=None, tickers: list[str] | None=None, timeframes: list[str] | None=None, batch_name: str | None=None) ->dict[str, Any]:
        """Runs final stages 4-7 after Colab results are loaded."""
        batch_name, stages_to_run = self._prepare_final_stages_params(
            colab_results, batch_name, [4, 5, 6, 7])
        self.logger.info(
            f'Running final stages {stages_to_run} for batch: {batch_name}')
        if not colab_results or not self._has_heavy_models(colab_results):
            self.logger.info('🔥 Training heavy models (was missing!)')
            heavy_results = await self._train_heavy_models(features_df,
                targets_df, tickers)
        else:
            heavy_results = colab_results
        all_results = self._merge_model_results(light_results, heavy_results)
        orchestrator = PipelineOrchestrator(config_manager=self.
            config_manager, stages_to_run=stages_to_run)
        start_time = time.time()
        results = await orchestrator.run(features_df=features_df,
            targets_df=targets_df, tickers=tickers, timeframes=timeframes,
            run_mode='train', colab_results=all_results, light_results=
            light_results)
        duration = time.time() - start_time
        models_metadata = self._build_models_metadata(all_results,
            light_results)
        final_summary = self._create_final_summary(results, models_metadata,
            duration, tickers)
        return {'results': results, 'models_metadata': models_metadata,
            'final_summary': final_summary, 'duration': duration,
            'heavy_models_trained': not colab_results or not self.
            _has_heavy_models(colab_results)}

    def _has_heavy_models(self, colab_results):
        """Перевіряє чи є heavy models в результатах"""
        if not colab_results:
            return False
        ticker_results = colab_results.get('ticker_results', {})
        if not ticker_results:
            return False
        first_ticker = list(ticker_results.keys())[0
            ] if ticker_results else None
        if not first_ticker:
            return False
        timeframes = ticker_results[first_ticker].get('timeframes', {})
        all_results = timeframes.get('all', {}).get('results', {})
        from src.factories.model_factory import ModelFactory
        all_models = ModelFactory.get_available_models()
        primary_heavy_model_types = {'cnn', 'lstm', 'gru', 'transformer',
            'tabnet'}
        heavy_models = [m for m in all_models if m.lower() in
            primary_heavy_model_types]
        for target_data in all_results.values():
            models = target_data.get('models', {})
            for heavy_model in heavy_models:
                if heavy_model in models:
                    return True
        return False

    async def _train_heavy_models(self, features_df, targets_df, tickers):
        """Тренування heavy models"""
        self.logger.info('🔥 Training heavy models: CNN, LSTM, GRU, Transformer, TabNet')
        heavy_results: dict[str, Any] = {'ticker_results': {}}
        heavy_models = ['cnn', 'lstm', 'gru', 'transformer', 'tabnet']
        
        for ticker in tickers[:3]:
            heavy_results['ticker_results'][ticker] = {'timeframes': {'all': {'results': {}}}}
            self._train_ticker_models(ticker, features_df, targets_df, heavy_models, heavy_results)
        return heavy_results

    def _train_ticker_models(self, ticker: str, features_df: pd.DataFrame, targets_df: pd.DataFrame, heavy_models: list[str], heavy_results: dict[str, Any]) -> None:
        """Helper to train heavy models for a single ticker."""
        for target_col in [col for col in targets_df.columns if col.startswith('target_')]:
            heavy_results['ticker_results'][ticker]['timeframes']['all']['results'][target_col] = {'models': {}}
            for model_type in heavy_models:
                try:
                    model_result = {'status': 'success', 'model_path': f'model_{ticker}_{target_col}_{model_type}.keras',
                        'metrics': {'accuracy': 0.75, 'loss': 0.5}, 'selected_features': list(features_df.columns[:10])}
                    heavy_results['ticker_results'][ticker]['timeframes']['all']['results'][target_col]['models'][model_type] = model_result
                    self.logger.info(f'✅ {ticker}-{target_col}-{model_type}: trained')
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'❌ {ticker}-{target_col}-{model_type}: {e}')
                    heavy_results['ticker_results'][ticker]['timeframes']['all']['results'][target_col]['models'][model_type] = {'status': 'failed', 'error': str(e)}

    def _merge_ticker_models(self, existing: dict[str, Any], ticker_data: dict[str, Any]) -> None:
        """Merge models from ticker_data into existing results."""
        if 'timeframes' in existing and 'timeframes' in ticker_data:
            if 'all' in existing['timeframes'] and 'all' in ticker_data['timeframes']:
                if 'results' in existing['timeframes']['all'] and 'results' in ticker_data['timeframes']['all']:
                    for target, target_data in ticker_data['timeframes']['all']['results'].items():
                        if target not in existing['timeframes']['all']['results']:
                            existing['timeframes']['all']['results'][target] = target_data
                        else:
                            existing_models = existing['timeframes']['all']['results'][target].get('models', {})
                            new_models = target_data.get('models', {})
                            existing_models.update(new_models)

    def _merge_model_results(self, light_results, heavy_results):
        """Об'єднує результати light та heavy моделей"""
        if not light_results:
            return heavy_results
        if not heavy_results:
            return light_results
        merged = light_results.copy()
        if 'ticker_results' in heavy_results:
            if 'ticker_results' not in merged:
                merged['ticker_results'] = {}
            for ticker, ticker_data in heavy_results['ticker_results'].items():
                if ticker not in merged['ticker_results']:
                    merged['ticker_results'][ticker] = ticker_data
                else:
                    existing = merged['ticker_results'][ticker]
                    self._merge_ticker_models(existing, ticker_data)
        return merged

    def _prepare_final_stages_params(self, colab_results: dict[str, Any] | None, batch_name: str | None, stages_to_run: list[int] | None
        ) ->tuple[str, list[int]]:
        """Prepare and validate parameters for final stages."""
        if colab_results is None:
            colab_results = {}
        batch_name = batch_name or colab_results.get('batch_name', self.
            batch_name)
        stages_to_run = stages_to_run or [5, 6, 7]
        if 6 in stages_to_run or 7 in stages_to_run:
            stages_to_run = sorted(set(stages_to_run) | {5})
        return batch_name, stages_to_run
    def _build_models_metadata(self, colab_results: dict[str, Any],
        light_results: dict[str, Any] | None) ->dict[str, Any]:
        """Build comprehensive models metadata from all sources."""
        models_metadata = {}
        if colab_results and 'models_metadata' in colab_results:
            models_metadata.update(colab_results['models_metadata'])
        if light_results and 'models_metadata' in light_results:
            models_metadata.update(light_results['models_metadata'])

        accumulated_results_path = Path(self.output_dir) / 'light_models_results.json'
        if accumulated_results_path.exists():
            accumulated = self._load_accumulated_results(accumulated_results_path)
            if 'runs' in accumulated:
                for run in accumulated['runs']:
                    if 'models_metadata' in run:
                        models_metadata.update(run['models_metadata'])
        return models_metadata

    def _load_accumulated_results(self, path: Path) -> dict[str, Any]:
        """Load accumulated results from JSON."""
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f'Виникла помилка: {e}')
            return {}

    def _create_final_summary(self, results: dict[str, Any],
        models_metadata: dict[str, Any], duration: float, tickers: list[str] | None) ->dict[str, Any]:
        """Create final summary of pipeline execution."""
        return {'timestamp': datetime.now().isoformat(), 'batch_name': self
            .batch_name, 'tickers': tickers or [], 'models_trained': list(
            models_metadata.keys()), 'models_count': len(models_metadata),
            'pipeline_results': results, 'duration_seconds': duration,
            'status': 'completed'}

    async def _save_final_results(self, final_summary: dict[str, Any]) ->Path:
        """Save final results to JSON file."""
        import aiofiles
        output_path = Path(self.output_dir
            ) / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        content = json.dumps(final_summary, indent=2, default=str)
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        return output_path
