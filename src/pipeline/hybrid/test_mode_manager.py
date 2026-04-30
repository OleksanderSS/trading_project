# src/pipeline/hybrid/test_mode_manager.py
"""
Test Mode Manager for Hybrid Orchestrator.

Handles test mode configuration loading and data filtering based on test criteria.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import aiofiles

from src.core.logging.logger import ProjectLogger


class TestModeManager:
    """
    Manages test mode configuration and filtering.
    
    Handles loading test parameters from runtime_params.json
    and applying filters to data and contexts.
    """
    
    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)
    
    async def _load_test_mode_config(self, batch_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Load test mode configuration from runtime_params.json."""
        runtime_params_path = batch_dir / "runtime_params.json"
        test_ticker = test_target = test_model = None
        
        if runtime_params_path.exists():
            try:
                async with aiofiles.open(runtime_params_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    runtime_params = json.loads(content)
                test_mode = runtime_params.get('test_mode', {})
                if test_mode.get('enabled'):
                    test_ticker = test_mode.get('test_ticker')
                    test_target = test_mode.get('test_target')
                    test_model = test_mode.get('test_model')
                    self.logger.info(f"Test mode: {test_ticker}|{test_target}|{test_model}")
            except Exception as e:
                self.logger.warning(f"Could not read runtime_params.json: {e}")
        
        return test_ticker, test_target, test_model
    
    def _filter_data_for_test_mode(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                                   test_ticker: Optional[str], test_target: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[str]]:
        """Filter data for test mode."""
        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        if test_target and test_target in target_cols:
            target_cols = [test_target]

        ticker_col = next((c for c in ['ticker', 'symbol', 'asset'] if c in features_df.columns), None)
        if test_ticker and ticker_col:
            mask = features_df[ticker_col].str.upper() == test_ticker.upper()
            features_df, targets_df = features_df[mask].copy(), targets_df[mask].copy()
        
        return features_df, targets_df, target_cols, ticker_col
    
    def _should_skip_model(self, model_name: str, test_model: Optional[str]) -> bool:
        """Check if model should be skipped based on test filter."""
        return test_model is not None and model_name != test_model
    
    def _should_skip_ticker(self, context_ticker: str, test_ticker: Optional[str]) -> bool:
        """Check if context should be skipped based on ticker filter."""
        return test_ticker is not None and context_ticker and context_ticker != test_ticker.upper()
    
    def _should_skip_target(self, context_target: str, test_target: Optional[str]) -> bool:
        """Check if context should be skipped based on target filter."""
        return test_target is not None and context_target and context_target != test_target
    
    def _should_skip_features(self, selected_features: List[str], model_name: str, 
                             light_models_to_train: List[str]) -> bool:
        """Check if context should be skipped based on features and model availability."""
        return not selected_features or model_name not in light_models_to_train