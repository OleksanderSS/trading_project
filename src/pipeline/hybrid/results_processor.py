"""
Results Processor: Handles loading, converting, and processing results from Colab and local training.
Extracted from HybridOrchestrator to improve code organization and testability.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger


class ResultsProcessor:
    """Processes training results from various sources."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def load_colab_results(self, batch_name: str, output_dir: Path) ->dict[
        str, Any]:
        """Loads training results from Colab."""
        batch_dir = self._find_batch_directory(batch_name, output_dir)
        if not batch_dir:
            self.logger.warning(
                f'⚠️ Batch directory not found for {batch_name}')
            return {'status': 'error', 'message': 'Batch directory not found'}
        results_path = self._find_results_file(batch_dir)
        if not results_path:
            self.logger.warning(f'⚠️ Results file not found in {batch_dir}')
            return {'status': 'error', 'message': 'Results file not found'}
        results = self._load_results_json(results_path)
        results = self._convert_model_paths(results, batch_dir)
        self.logger.info(f'✅ Loaded Colab results from {results_path}')
        return results

    def _find_batch_directory(self, batch_name: str, output_dir: Path
        ) ->Path | None:
        """Find the batch directory, trying similar names if exact match not found."""
        eff_batch_name = batch_name.replace('target_target_', 'target_')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f'🔍 Searching for batch directory: batch_name={batch_name}, eff_batch_name={eff_batch_name}, output_dir={output_dir}'
                )
        if output_dir.name == batch_name or output_dir.name == eff_batch_name:
            if output_dir.exists():
                return output_dir
        batch_dir = output_dir / batch_name
        if batch_dir.exists():
            return batch_dir
        batch_dir = output_dir / eff_batch_name
        if batch_dir.exists():
            return batch_dir
        search_root = output_dir
        if output_dir.name == batch_name or output_dir.name == eff_batch_name:
            search_root = output_dir.parent
        try:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f'🔍 Searching for similar directories in {search_root}')
            similar = [d for d in search_root.iterdir() if d.is_dir() and (
                eff_batch_name in d.name or d.name in eff_batch_name)]
            if similar:
                chosen = max(similar, key=lambda p: p.stat().st_mtime)
                self.logger.info(f'✅ Found similar batch directory: {chosen}')
                return chosen
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Error searching for batch directory: {e}')
            raise
        return None

    def _find_results_file(self, batch_dir: Path) ->Path | None:
        """Find the results file, trying summary first then regular results."""
        files_to_try = [batch_dir / 'colab_results_summary.json', batch_dir /
            'colab_results.json', batch_dir / 'results.json']
        for file_path in files_to_try:
            if file_path.exists():
                return file_path
        return None

    def _load_results_json(self, results_path: Path) ->dict[str, Any]:
        """Load results from JSON file."""
        try:
            with open(results_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(
                f'❌ Error loading results from {results_path}: {e}')
            raise

    def _convert_model_paths(self, obj: Any, batch_dir: Path) ->Any:
        """Convert model paths to local paths recursively."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'model_path' and isinstance(v, str):
                    obj[k] = self._convert_single_model_path(v, batch_dir)
                else:
                    obj[k] = self._convert_model_paths(v, batch_dir)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._convert_model_paths(item, batch_dir)
        return obj

    def _convert_single_model_path(self, model_path: str, batch_dir: Path
        ) ->str:
        """Convert a single model path to local path."""
        fname = model_path.split('/')[-1]
        models_dir = batch_dir / 'models'
        if models_dir.exists():
            local_path = models_dir / fname
            if local_path.exists():
                return str(local_path)
        return str(batch_dir / fname)

    def build_models_metadata(self, colab_results: dict[str, Any],
        light_results: dict[str, Any] | None) ->dict[str, Any]:
        """Build models metadata from colab and light results."""
        models_metadata = {}
        if 'models_metadata' in colab_results:
            models_metadata.update(colab_results['models_metadata'])
            self._update_selected_features_from_ticker_results(models_metadata,
                colab_results)
        if light_results and 'models_metadata' in light_results:
            models_metadata.update(light_results['models_metadata'])
        self.logger.info(f'✅ Built metadata for {len(models_metadata)} models')
        return models_metadata

    def _update_selected_features_from_ticker_results(self, models_metadata:
        dict[str, Any], colab_results: dict[str, Any]) ->None:
        """Update selected features from ticker results in colab results."""
        if 'ticker_results' not in colab_results:
            return
        ticker_results = colab_results['ticker_results']
        for ticker, ticker_data in ticker_results.items():
            for _timeframe, timeframe_data in ticker_data.get('timeframes', {}
                ).items():
                for target, target_data in timeframe_data.get('results', {}
                    ).items():
                    for model, model_data in target_data.get('models', {}
                        ).items():
                        self._update_model_selected_features(models_metadata,
                            ticker, target, model, model_data)

    def _update_model_selected_features(self, models_metadata: dict[str,
        Any], ticker: str, target: str, model: str, model_data: dict[str, Any]
        ) ->None:
        """Update selected features for a specific model."""
        key = f'{ticker}_{target}_{model}'
        if key in models_metadata:
            models_metadata[key]['selected_features'] = model_data.get(
                'selected_features', [])
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'✅ Updated selected features for {key}')

    def extract_batch_name_from_path(self, path_str: str) ->str | None:
        """Extract batch name from path."""
        parts = Path(path_str.replace('/', os.sep)).parts
        if 'accumulated' in parts:
            idx = parts.index('accumulated')
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return None
