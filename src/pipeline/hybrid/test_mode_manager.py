# src/pipeline/hybrid/test_mode_manager.py
"""
Test Mode Manager for Hybrid Orchestrator.

Handles test mode configuration loading and data filtering based on test criteria.
"""

import json
from pathlib import Path

import aiofiles
import pandas as pd

from src.core.logging.logger import ProjectLogger


class TestModeManager:
    """
    Manages test mode configuration and filtering.

    Handles loading test parameters from runtime_params.json
    and applying filters to data and contexts.
    """

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    async def _load_test_mode_config(self, batch_dir: Path) -> tuple[str | None, str | None, str | None]:
        """Load test mode configuration from runtime_params.json."""
        runtime_params_path = batch_dir / "runtime_params.json"
        test_ticker = test_target = test_model = None

        self.logger.info(f"🔍 Looking for runtime_params.json at: {runtime_params_path}")
        self.logger.info(f"🔍 File exists: {runtime_params_path.exists()}")

        if runtime_params_path.exists():
            try:
                async with aiofiles.open(runtime_params_path, encoding='utf-8') as f:
                    content = await f.read()
                    runtime_params = json.loads(content)
                self.logger.info(f"🔍 Loaded runtime_params: {runtime_params}")
                test_mode = runtime_params.get('test_mode', {})
                self.logger.info(f"🔍 Test mode section: {test_mode}")
                if test_mode.get('test_ticker') is not None or test_mode.get('test_target') is not None or test_mode.get('test_model') is not None:
                    test_ticker = test_mode.get('test_ticker')
                    test_target = test_mode.get('test_target')
                    test_model = test_mode.get('test_model')
                    self.logger.info(f"Test mode loaded from file: {test_ticker}|{test_target}|{test_model}")
                else:
                    self.logger.info("🔍 No test mode parameters found, using None values")
            except Exception as e:
                self.logger.warning(f"Could not read runtime_params.json: {e}")
        else:
            self.logger.warning(f"🔍 runtime_params.json not found at {runtime_params_path}")

        return test_ticker, test_target, test_model

    def _filter_data_for_test_mode(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                   test_ticker: str | None, test_target: str | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]:
        """Filter data for test mode."""
        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        if test_target and test_target in target_cols:
            target_cols = [test_target]

        ticker_col = next((c for c in ['ticker', 'symbol', 'asset'] if c in features_df.columns), None)

        # Only filter if test_ticker is explicitly set (not None or empty)
        if test_ticker and ticker_col:
            mask = features_df[ticker_col].str.upper() == test_ticker.upper()
            features_df, targets_df = features_df[mask].copy(), targets_df[mask].copy()
        elif test_ticker is None:
            # No filtering when test_ticker is None - return all data
            pass

        return features_df, targets_df, target_cols, ticker_col

    def _should_skip_model(self, model_name: str, test_model: str | None) -> bool:
        """Check if model should be skipped based on test filter."""
        return test_model is not None and model_name != test_model

    def _should_skip_ticker(self, context_ticker: str, test_ticker: str | None) -> bool:
        """Check if context should be skipped based on ticker filter."""
        return bool(test_ticker is not None and context_ticker and context_ticker != test_ticker.upper())

    def _should_skip_target(self, context_target: str, test_target: str | None) -> bool:
        """Check if context should be skipped based on target filter."""
        return bool(test_target is not None and context_target and context_target != test_target)

    def _should_skip_features(self, selected_features: list[str], model_name: str,
                             light_models_to_train: list[str]) -> bool:
        """Check if context should be skipped based on features and model availability."""
        return not selected_features or model_name not in light_models_to_train
