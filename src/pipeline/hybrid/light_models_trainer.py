"""
Light Models Trainer - Handles light model training logic
"""
import time
import copy
import aiofiles
import pandas as pd
from typing import Dict, List, Any, Optional, cast
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
logger = ProjectLogger.get_logger(__name__)


@dataclass
class LightModelTrainingConfig:
    """Configuration for light model training with selected features."""
    test_ticker: str
    test_target: str
    test_model: str
    light_models_to_train: List[str]
    target_cols: List[str]
    selected_features_files: List[str]


class LightModelsTrainer:
    """Handles light model training and result accumulation."""

    def __init__(self, trainer_config: Dict[str, Any]):
        self.config_manager = trainer_config['config_manager']
        self.output_dir = trainer_config['output_dir']
        self.batch_name = trainer_config['batch_name']
        self.light_models = trainer_config['light_models']
        self.models_config = trainer_config['models_config']
        self.logger = ProjectLogger.get_logger(__name__)

    async def run_light_models(self, features_df: pd.DataFrame, targets_df:
        pd.DataFrame, tickers: Optional[List[str]]=None) ->Dict[str, Any]:
        """Trains light models locally and accumulates results."""
        self.logger.info('Launching light model training...')
        original_config = self.config_manager.merged_config.get('models')
        self._set_temp_light_models_config()
        try:
            orchestrator = PipelineOrchestrator(config_manager=self.
                config_manager, stages_to_run=[4])
            results = await orchestrator.run(enriched_data=features_df,
                targets_df=targets_df, tickers=tickers, run_mode='train',
                stages_to_run=[4])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return await self._save_light_model_results(results, timestamp)
        finally:
            self.config_manager.merged_config['models'] = original_config

    def _set_temp_light_models_config(self) ->None:
        """Set temporary light models configuration."""
        if 'modeling' not in self.config_manager.merged_config:
            self.config_manager.merged_config['modeling'] = {}
        modeling_config = self.config_manager.merged_config['modeling']
        if not isinstance(modeling_config, dict):
            self.config_manager.merged_config['modeling'] = {}
            modeling_config = self.config_manager.merged_config['modeling']
        modeling_config['strategy'] = 'batch'
        models_dict = self.models_config.as_dict() if hasattr(self.
            models_config, 'as_dict') else self.models_config
        temp_config_dict = copy.deepcopy(models_dict)
        temp_config_dict['categories'] = {'light': self.light_models}
        self.config_manager.merged_config['models'] = temp_config_dict

    async def _save_light_model_results(self, results: Dict[str, Any],
        timestamp: str) ->Dict[str, Any]:
        """Save light model results."""
        from pathlib import Path
        light_results_path = self.output_dir / 'light_models_results.json'
        current_run = {'timestamp': timestamp, 'models_metadata': results.
            get('models_metadata', {}), 'metrics': results.get('metrics', {})}
        accumulated_results = self._load_or_create_accumulated_results(
            light_results_path, current_run)
        async with aiofiles.open(light_results_path, 'w', encoding='utf-8'
            ) as f:
            import json
            await f.write(json.dumps(accumulated_results, indent=2, default
                =str))
        self.logger.info(f'Light models results saved to {light_results_path}')
        return {'status': 'light_models_complete', 'models_metadata':
            results.get('models_metadata', {}), 'metrics': results.get(
            'metrics', {}), 'timestamp': timestamp,
            'accumulated_results_path': str(light_results_path)}

    def _load_or_create_accumulated_results(self, results_path: Path,
        current_run: Dict[str, Any]) ->Dict[str, Any]:
        """Load existing results or create new accumulation structure."""
        import json
        if results_path.exists():
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    accumulated = cast(Dict[str, Any], json.load(f))
                if 'runs' not in accumulated:
                    accumulated['runs'] = []
                accumulated['runs'].append(current_run)
                accumulated['last_updated'] = current_run['timestamp']
                return accumulated
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(f'Could not load existing results: {e}')
                raise
        return {'batch_name': self.batch_name, 'created': current_run[
            'timestamp'], 'last_updated': current_run['timestamp'], 'runs':
            [current_run]}

    async def run_light_models_with_selected_features(self, features_df: pd
        .DataFrame, targets_df: pd.DataFrame, config: LightModelTrainingConfig
        ) ->Dict[str, Any]:
        """Run light models with pre-selected features."""
        self.logger.info(
            f'Training light models with pre-selected features for {len(config.light_models_to_train)} models...'
            )
        selected_features = self._load_selected_features(config.
            selected_features_files, config.test_model)
        if not selected_features:
            self.logger.warning(
                'No selected features found, using all features')
            selected_features = [col for col in features_df.columns if not
                col.startswith('target_')]
        filtered_features = features_df[selected_features + ['ticker',
            'timeframe', 'datetime'] + config.target_cols]
        return await self.run_light_models(filtered_features, targets_df)

    def _load_selected_features(self, selected_features_files: List[str],
        model_name: str) ->List[str]:
        """Load selected features for a specific model."""
        import json
        from pathlib import Path
        for features_file in selected_features_files:
            features_path = Path(features_file)
            if features_path.exists():
                try:
                    with open(features_path, 'r', encoding='utf-8') as f:
                        features_data = json.load(f)
                    if features_data.get('model_name') == model_name:
                        return cast(List[str], features_data.get(
                            'selected_features', []))
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(
                        f'Could not load features from {features_file}: {e}')
                    raise
        return []
