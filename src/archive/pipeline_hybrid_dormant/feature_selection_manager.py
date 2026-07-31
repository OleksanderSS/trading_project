# audit-ignore: ARCHITECTURAL_USAGE
"""
Feature Selection Manager for Hybrid Orchestrator.
Handles all feature selection logic and validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.hybrid.feature_selection_validator import MockFeatureFilesRequest

logger = ProjectLogger.get_logger(__name__)


@dataclass
class MockSelectionParams:
    """Parameters for mock feature selection."""
    batch_dir: Path
    test_ticker: str
    test_target: str
    light_models: list[str]
    features_df: pd.DataFrame


class FeatureSelectionManager:
    """Manages feature selection operations."""

    def __init__(self, config):
        self.config = config
        self.logger = ProjectLogger.get_logger(__name__)

    def create_mock_feature_selection(self, params: MockSelectionParams) -> list[Path]:
        """Create mock feature selection for testing."""
        valid_features = self._extract_valid_features(params.features_df)
        mock_selected_features = valid_features[:min(50, len(valid_features))]

        self.logger.info(f"Selected {len(mock_selected_features)} features")

        # Create mock files request
        mock_files_request = MockFeatureFilesRequest(
            batch_dir=params.batch_dir,
            test_ticker=params.test_ticker,
            test_target=params.test_target,
            light_models=params.light_models,
            selected_features=mock_selected_features
        )

        return self._write_mock_feature_files(mock_files_request)

    def _extract_valid_features(self, features_df: pd.DataFrame) -> list[str]:
        """Extract valid numeric features from dataframe."""
        feature_cols = [col for col in features_df.columns
                       if not col.startswith('target_') and col not in ['ticker', 'timeframe', 'date', 'datetime']]

        valid_features = []
        for col in feature_cols:
            if self._is_valid_feature(features_df, col):
                valid_features.append(col)
        return valid_features

    def _is_valid_feature(self, features_df: pd.DataFrame, col: str) -> bool:
        """Check if feature is valid."""
        notna_count = features_df[col].notna().sum()
        if notna_count == 0:
            return False

        dtype = features_df[col].dtype
        return dtype in ['float64', 'int64', 'float32', 'int32', 'bool']

    def _write_mock_feature_files(self, request: MockFeatureFilesRequest) -> list[Path]:
        """Write mock feature files."""
        created_files = []

        for model in request.light_models:
            features_data = {
                'model_name': model,
                'features': request.selected_features,
                'selection_method': 'mock_selection',
                'test_ticker': request.test_ticker,
                'test_target': request.test_target,
                'selection_score': 0.8,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            feature_file = request.batch_dir / f"selected_features_{model}_{request.test_target}.json"
            with open(feature_file, 'w') as f:
                json.dump(features_data, f, indent=2, default=str)

            created_files.append(feature_file)
            self.logger.info(f"Created mock feature file: {feature_file}")

        return created_files
