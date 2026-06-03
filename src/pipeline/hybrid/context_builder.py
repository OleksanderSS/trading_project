# src/pipeline/hybrid/context_builder.py
"""
Context Builder for Hybrid Orchestrator.

Builds and validates training contexts from raw data.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from src.core.logging.logger import ProjectLogger

from .test_mode_manager import TestModeManager


class ContextBuilder:
    """
    Builds training contexts from selected features data.
    
    Creates context IDs and validates context data against test filters.
    """
    
    def __init__(self, test_mode_manager: TestModeManager):
        self.test_mode_manager = test_mode_manager
        self.logger = ProjectLogger.get_logger(__name__)
    
    def _create_context_id(self, context_ticker: str, test_ticker: Optional[str], 
                          context_target: str, model_name: str) -> str:
        """Create unique context identifier."""
        ticker = context_ticker or test_ticker or 'ALL'
        target = context_target or 'ALL'
        return f"{ticker}::{target}::{model_name}"
    
    def _create_context_data(self, model_name: str, context_ticker: str, context_target: str, 
                            selected_features: List[str], target_cols: List[str], file_path: Path) -> Dict[str, Any]:
        """Create context data dictionary."""
        return {
            'model_name': model_name,
            'ticker': context_ticker,
            'targets': [context_target] if context_target else list(target_cols),
            'selected_features': selected_features,
            'source_file': file_path.name
        }
    
    def _validate_and_create_context(self, data: Dict[str, Any], test_ticker: Optional[str], 
                                     test_target: Optional[str], test_model: Optional[str],
                                     light_models_to_train: List[str], target_cols: List[str],
                                     file_path: Path) -> Optional[Dict[str, Any]]:
        """Validate data and create context if valid."""
        model_name = data.get('model_type', data.get('model_name'))
        
        if self.test_mode_manager._should_skip_model(model_name, test_model):
            return None
        
        context_ticker = str(data.get('ticker', '')).upper()
        context_target = data.get('target')
        
        if self.test_mode_manager._should_skip_ticker(context_ticker, test_ticker) or \
           self.test_mode_manager._should_skip_target(context_target, test_target):
            return None
        
        selected_features = data.get('selected_features', [])
        if self.test_mode_manager._should_skip_features(selected_features, model_name, light_models_to_train):
            return None
        
        return self._create_context_data(model_name, context_ticker, context_target, 
                                        selected_features, target_cols, file_path)