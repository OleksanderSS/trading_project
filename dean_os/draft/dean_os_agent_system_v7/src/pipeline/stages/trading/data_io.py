from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger


class TradingDataIO:

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.output_dir = Path(self.config_manager.get(
            'system.accumulation.output_dir', 'data/colab/accumulated'))

    async def load_predictions(self, kwargs: dict[str, Any]) ->tuple[list[
        dict[str, Any]] | None, dict[str, float] | None, dict[str, Any]]:
        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')
        if not predictions:
            predictions, current_prices, kwargs = (await self.
                _load_predictions_from_disk(kwargs))
        return predictions, current_prices, kwargs

    async def _load_predictions_from_disk(self, kwargs: dict[str, Any]
        ) ->tuple[list[dict[str, Any]] | None, dict[str, float] | None,
        dict[str, Any]]:
        self.logger.warning(
            "⚠️ No 'predictions' found in kwargs. Attempting to load from disk..."
            )
        batch_name = kwargs.get('batch_name')
        if not batch_name:
            if (self.output_dir / 'main_database').exists():
                batch_name = 'main_database'
            else:
                batch_dirs = list(self.output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().
                        st_mtime).name
            if batch_name:
                self.logger.info(f'🔍 Using batch: {batch_name}')
        if not batch_name:
            self.logger.warning('⚠️ Could not find batch_name')
            return None, None, kwargs
        return await self._process_batch_file(batch_name, kwargs)

    async def _process_batch_file(self, batch_name: str, kwargs: dict[str, Any]
        ) ->tuple[list[dict[str, Any]] | None, dict[str, float] | None,
        dict[str, Any]]:
        batch_dir = self.output_dir / batch_name
        stage_5_file = batch_dir / 'stage_5_results.json'
        if not stage_5_file.exists():
            self.logger.warning(f'⚠️ File not found: {stage_5_file}')
            return None, None, kwargs
        try:
            content = await self._read_file_async(stage_5_file)
            stage_5_results = json.loads(content)
            predictions = stage_5_results.get('predictions', [])
            current_prices = stage_5_results.get('current_prices', {})
            if 'models_metadata' not in kwargs:
                models_metadata = stage_5_results.get('models_metadata', {})
                if models_metadata:
                    kwargs['models_metadata'] = models_metadata
                    self.logger.info(
                        f'✅ Loaded {len(models_metadata)} models with metadata'
                        )
            self.logger.info(
                f'✅ Loaded {len(predictions)} forecasts from {stage_5_file.name}'
                )
            self.logger.info(
                f'✅ Loaded prices for {len(current_prices)} tickers')
            return predictions, current_prices, kwargs
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'❌ Error loading {stage_5_file}: {e}',
                exc_info=True)
            return None, None, kwargs

    async def _read_file_async(self, file_path: Path) ->str:
        async with aiofiles.open(file_path, encoding='utf-8') as f:
            return await f.read()

    def save_stage_6_results(self, results_bundle: dict[str, Any], kwargs:
        dict[str, Any]) ->None:
        try:
            batch_name = kwargs.get('batch_name'
                ) or self._find_latest_batch_name()
            if not batch_name:
                return
            batch_dir = self.output_dir / batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)
            stage_6_results = {'timestamp': datetime.now().isoformat(),
                'batch_name': batch_name, **results_bundle, 'total_trades':
                len(results_bundle.get('trade_history', [])),
                'portfolio_value': results_bundle.get('portfolio_summary',
                {}).get('total_value', 0)}
            results_file = batch_dir / 'stage_6_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(stage_6_results, f, indent=2, default=str)
            self.logger.info(f'✅ Stage 6 results saved: {results_file.name}')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Error saving Stage 6 results: {e}',
                exc_info=True)
            raise

    def _find_latest_batch_name(self) ->(str | None):
        batch_dirs = list(self.output_dir.glob('test_ticker_*'))
        if not batch_dirs:
            return None
        return max(batch_dirs, key=lambda p: p.stat().st_mtime).name
