"""
Selected Features Processor for Hybrid Orchestrator.

Discovers, validates, loads, and processes selected features files into training contexts.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiofiles
from src.core.logging.logger import ProjectLogger
from .context_builder import ContextBuilder
from .feature_selection_validator import MockFeaturesRequest, FeatureSelectionValidator
from dataclasses import dataclass


@dataclass
class DiscoverSelectedFeaturesFilesConfig:
    batch_dir: Path
    test_ticker: Optional[str]
    test_target: Optional[str]
    light_models_to_train: List[str]
    features_df: Any


@dataclass
class ProcessSelectedFeaturesFileConfig:
    file_path: Path
    test_ticker: Optional[str]
    test_target: Optional[str]
    test_model: Optional[str]
    light_models_to_train: List[str]
    target_cols: List[str]


@dataclass
class ProcessFilesToContextsConfig:
    selected_features_files: List[Path]
    test_ticker: Optional[str]
    test_target: Optional[str]
    test_model: Optional[str]
    light_models_to_train: List[str]
    target_cols: List[str]


@dataclass
class FeatureLoadRequest:
    """Request for loading selected features files."""
    batch_dir: Path
    test_ticker: Optional[str]
    test_target: Optional[str]
    test_model: Optional[str]
    light_models_to_train: List[str]
    target_cols: List[str]
    features_df: Any


class SelectedFeaturesProcessor:
    """
    Processes selected features files into training contexts.
    
    Handles file discovery, validation, loading, and context creation.
    """

    def __init__(self, context_builder: ContextBuilder,
        feature_selection_validator: FeatureSelectionValidator):
        self.context_builder = context_builder
        self.feature_selection_validator = feature_selection_validator
        self.logger = ProjectLogger.get_logger(__name__)
        self.selected_features_pattern = 'selected_features_*.json'

    def _discover_selected_features_files(self, config:
        DiscoverSelectedFeaturesFilesConfig) ->List[Path]:
        """Discover and collect selected features files."""
        selected_features_files = list(config.batch_dir.glob(self.
            selected_features_pattern))
        if config.test_ticker:
            mock_request = MockFeaturesRequest(batch_dir=config.batch_dir,
                test_ticker=config.test_ticker, test_target=config.
                test_target, light_models=config.light_models_to_train,
                features_df=config.features_df)
            selected_features_files.extend(self.feature_selection_validator
                ._create_mock_selected_features_for_test(mock_request))
        return selected_features_files

    def _validate_files_exist(self, selected_features_files: List[Path]
        ) ->bool:
        """Validate that selected features files exist."""
        if not selected_features_files:
            self.logger.error('No selected_features_*.json files found.')
            return False
        return True

    async def _process_selected_features_file(self, config:
        ProcessSelectedFeaturesFileConfig) ->Optional[Dict[str, Any]]:
        """Process a single selected features file and return context data or None."""
        try:
            async with aiofiles.open(config.file_path, 'r', encoding='utf-8'
                ) as f:
                content = await f.read()
                data = json.loads(content)
            return self.context_builder._validate_and_create_context(data,
                config.test_ticker, config.test_target, config.test_model,
                config.light_models_to_train, config.target_cols, config.
                file_path)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Error loading {config.file_path.name}: {e}')
            raise RuntimeError(
                f"Failed to process selected features file: {config.file_path}"
            ) from e

    def _extract_context_info(self, context_data: Dict[str, Any]) ->tuple[
        str, str, str]:
        """Extract context information from processed data."""
        model_name = context_data['model_name']
        context_ticker = context_data['ticker']
        context_target = context_data['targets'][0] if context_data['targets'
            ] else 'ALL'
        return model_name, context_ticker, context_target

    async def _process_files_to_contexts(self, config:
        ProcessFilesToContextsConfig) ->Dict[str, Dict[str, Any]]:
        """Process selected features files into contexts."""
        selected_feature_contexts = {}
        for file_path in config.selected_features_files:
            file_config = ProcessSelectedFeaturesFileConfig(file_path=
                file_path, test_ticker=config.test_ticker, test_target=
                config.test_target, test_model=config.test_model,
                light_models_to_train=config.light_models_to_train,
                target_cols=config.target_cols)
            context_data = await self._process_selected_features_file(
                file_config)
            if context_data:
                model_name, context_ticker, context_target = (self.
                    _extract_context_info(context_data))
                context_id = self.context_builder._create_context_id(
                    context_ticker, config.test_ticker, context_target,
                    model_name)
                selected_feature_contexts[context_id] = context_data
        return selected_feature_contexts

    async def _load_selected_features_files(self, request: FeatureLoadRequest
        ) ->Dict[str, Dict[str, Any]]:
        """Load and process selected features files."""
        discovery_config = DiscoverSelectedFeaturesFilesConfig(batch_dir=
            request.batch_dir, test_ticker=request.test_ticker, test_target
            =request.test_target, light_models_to_train=request.
            light_models_to_train, features_df=request.features_df)
        selected_features_files = self._discover_selected_features_files(
            discovery_config)
        if not self._validate_files_exist(selected_features_files):
            return {}
        process_config = ProcessFilesToContextsConfig(selected_features_files
            =selected_features_files, test_ticker=request.test_ticker,
            test_target=request.test_target, test_model=request.test_model,
            light_models_to_train=request.light_models_to_train,
            target_cols=request.target_cols)
        return await self._process_files_to_contexts(process_config)
