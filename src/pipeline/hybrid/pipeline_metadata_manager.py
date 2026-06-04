"""
Pipeline Metadata Manager for Hybrid Orchestrator.

Creates, accumulates, and persists pipeline execution metadata.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from src.core.logging.logger import ProjectLogger


class PipelineMetadataManager:
    """
    Manages pipeline execution metadata.

    Handles creation, accumulation, and persistence of pipeline metadata.
    """

    def __init__(self, output_dir: Path, batch_name: str, light_models:
        list[str], heavy_models: list[str]):
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.light_models = light_models
        self.heavy_models = heavy_models
        self.logger = ProjectLogger.get_logger(__name__)
        self.batch_metadata_file = 'batch_metadata.json'

    def _create_pipeline_metadata(self, timestamp: str, tickers: list[str] | None, timeframes: list[str] | None, stages: list[int],
        saved_files: dict[str, str]) ->dict[str, Any]:
        """Create pipeline metadata."""
        return {'timestamp': timestamp, 'tickers': tickers, 'timeframes':
            timeframes, 'stages_completed': stages, 'saved_files':
            saved_files, 'light_models': self.light_models, 'heavy_models':
            self.heavy_models}

    def _save_metadata(self, metadata: dict[str, Any], timestamp: str) ->None:
        """Save metadata to files."""
        metadata_path = (self.output_dir /
            f'{self.batch_name}_metadata_{timestamp}.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        batch_metadata = self._create_batch_metadata_dict(metadata, timestamp)
        batch_metadata_path = self.output_dir / self.batch_metadata_file
        with open(batch_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2)
        self.logger.info(f'📋 Metadata saved: {metadata_path}')

    def _create_batch_metadata_dict(self, metadata: dict[str, Any],
        timestamp: str) ->dict[str, Any]:
        """Create batch metadata dictionary."""
        return {'batch_name': self.batch_name, 'timestamp': timestamp,
            'tickers': metadata['tickers'], 'timeframes': metadata[
            'timeframes'], 'heavy_models': self.heavy_models, 'files':
            metadata['saved_files']}

    def _load_or_create_accumulated_results(self, light_results_path: Path,
        current_run: dict[str, Any]) ->dict[str, Any]:
        """Load existing or create new accumulated results."""
        accumulated_results = {'timestamp': datetime.now().isoformat(),
            'total_runs': 1, 'runs': [current_run]}
        if light_results_path.exists():
            try:
                with open(light_results_path, encoding='utf-8') as f:
                    existing = json.load(f)
                    accumulated_results['total_runs'] = existing.get(
                        'total_runs', 0) + 1
                    accumulated_results['runs'] = existing.get('runs', []) + [
                        current_run]
                    self.logger.info(
                        f"📊 Accumulated {accumulated_results['total_runs']} runs"
                        )
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(f'⚠️ Could not load existing results: {e}')
                raise
        return accumulated_results

    async def _save_light_model_results(self, results: dict[str, Any],
        timestamp: str) ->dict[str, Any]:
        """Save light model results."""
        light_results_path = self.output_dir / 'light_models_results.json'
        current_run = {'timestamp': timestamp, 'models_metadata': results.
            get('models_metadata', {}), 'metrics': results.get('metrics', {})}
        accumulated_results = self._load_or_create_accumulated_results(
            light_results_path, current_run)
        async with aiofiles.open(light_results_path, 'w', encoding='utf-8'
            ) as f:
            await f.write(json.dumps(accumulated_results, indent=2, default
                =str))
        self.logger.info(
            f'✅ Light models results accumulated: {light_results_path}')
        return {'status': 'light_models_complete', 'results': results,
            'saved_path': str(light_results_path), 'timestamp': timestamp,
            'total_runs': accumulated_results['total_runs']}
