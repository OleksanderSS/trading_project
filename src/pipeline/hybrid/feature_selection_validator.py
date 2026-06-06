# audit-ignore: ARCHITECTURAL_USAGE
"""
Feature Selection Validator: Checks if feature selection is needed and creates mock selections for testing.
Extracted from HybridOrchestrator to improve code organization and testability.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

FEATURES_FILE = 'features.parquet'
TARGETS_FILE = 'targets.parquet'
SELECTED_FEATURES_PATTERN = 'selected_features_*.json'


@dataclass
class MockFeatureFilesRequest:
    """Request for writing mock feature files."""
    batch_dir: Path
    test_ticker: str
    test_target: str
    light_models: list[str]
    selected_features: list[str]


@dataclass
class MockFeaturesRequest:
    """Request for creating mock selected_features."""
    batch_dir: Path
    test_ticker: str
    test_target: str
    light_models: list[str]
    features_df: pd.DataFrame


class FeatureSelectionValidator:
    """Validates and creates feature selections."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def check_if_feature_selection_needed(self, batch_dir: Path,
        new_rows_count: int, force: bool=False) ->dict[str, Any]:
        """Check if new feature selection is required."""
        result = self._check_initial_or_forced_selection(batch_dir, force)
        if result:
            return result
        result = self._check_time_based_selection(batch_dir)
        if result:
            return result
        return self._check_data_change_percentage(batch_dir, new_rows_count)

    def _check_initial_or_forced_selection(self, batch_dir: Path, force: bool
        ) ->dict[str, Any] | None:
        """Check if selection is needed due to initial run or force."""
        files = list(batch_dir.glob(SELECTED_FEATURES_PATTERN))
        if not files:
            return {'needed': True, 'reason':
                'No selected_features files found (initial run)'}
        if force:
            return {'needed': True, 'reason':
                'Forced feature selection requested'}
        return None

    def _check_time_based_selection(self, batch_dir: Path) ->dict[str, Any] | None:
        """Check if selection is needed based on time elapsed."""
        files = list(batch_dir.glob(SELECTED_FEATURES_PATTERN))
        if not files:
            return None
        try:
            with open(files[0], encoding='utf-8') as f:
                data = json.load(f)
                last_ts = data.get('timestamp')
            if last_ts:
                days = (datetime.now() - datetime.fromisoformat(last_ts)).days
                if days > 7:
                    return {'needed': True, 'reason':
                        f'{days} days passed (> 7 days)'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error checking time-based selection: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Could not check time-based selection: {e}')
        return None

    def _check_data_change_percentage(self, batch_dir: Path, new_rows_count:
        int) ->dict[str, Any]:
        """Check if selection is needed based on data change percentage."""
        f_path = batch_dir / FEATURES_FILE
        if not f_path.exists():
            return {'needed': True, 'reason': 'No features file found'}
        try:
            old_rows = len(pd.read_parquet(f_path))
            if old_rows > 0:
                pct = new_rows_count / old_rows * 100
                if pct > 10:
                    return {'needed': True, 'reason':
                        f'Data changed by {pct:.1f}% (> 10%)'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error checking data change: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Could not check data change: {e}')
        return {'needed': False, 'reason': 'No feature selection needed'}

    def create_mock_selected_features_for_test(self, request:
        MockFeaturesRequest) ->list[Path]:
        """Create mock selected_features files for testing when Colab results are not available."""
        self.logger.info(
            f'Creating mock selected_features files: {request.test_ticker}, {request.test_target}'
            )
        valid_features = self._extract_valid_features(request.features_df)
        mock_selected_features = valid_features[:20] if len(valid_features
            ) >= 20 else valid_features
        self.logger.info(f'Selected {len(mock_selected_features)} features')
        mock_files_request = MockFeatureFilesRequest(batch_dir=request.
            batch_dir, test_ticker=request.test_ticker, test_target=request
            .test_target, light_models=request.light_models,
            selected_features=mock_selected_features)
        return self._write_mock_feature_files(mock_files_request)

    def _extract_valid_features(self, features_df: pd.DataFrame) ->list[str]:
        """Extract valid numeric features from dataframe."""
        feature_cols = [col for col in features_df.columns if not col.
            startswith('target_') and col not in ['ticker', 'timeframe',
            'date', 'datetime']]
        valid_features = []
        for col in feature_cols:
            if self._is_valid_feature(features_df, col):
                valid_features.append(col)
        return valid_features

    def _is_valid_feature(self, features_df: pd.DataFrame, col: str) ->bool:
        """Check if feature is valid."""
        try:
            notna_count = features_df[col].notna().sum()
            if notna_count == 0:
                return False
            dtype = features_df[col].dtype
            return dtype in ['float64', 'int64', 'float32', 'int32', 'bool']
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error validating feature {col}: {e}', exc_info=True)
            return False

    def _write_mock_feature_files(self, request: MockFeatureFilesRequest
        ) ->list[Path]:
        """Write mock feature files."""
        created_files = []
        request.batch_dir.mkdir(parents=True, exist_ok=True)
        for model_name in request.light_models:
            mock_data = {'ticker': request.test_ticker, 'target': request.
                test_target, 'model_type': model_name, 'selected_features':
                request.selected_features, 'feature_count': len(request.
                selected_features), 'max_features': 20, 'timestamp':
                datetime.now().isoformat()}
            file_name = (
                f'selected_features_{model_name}_{request.test_ticker}_{request.test_target}.json'
                )
            file_path = request.batch_dir / file_name
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(mock_data, f, indent=2)
            created_files.append(file_path)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Created mock file: {file_path}')
        return created_files
