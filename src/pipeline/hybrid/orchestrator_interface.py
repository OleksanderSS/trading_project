"""
Orchestrator Interface - Public API methods.
Contains all public methods that the main orchestrator delegates to.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.core.logging.logger import ProjectLogger
from .pipeline_config import PipelineParams, FinalStagesParams, ColabBatchParams

logger = ProjectLogger.get_logger(__name__)


class OrchestratorInterface:
    """Public interface methods for Hybrid Orchestrator."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = ProjectLogger.get_logger(__name__)
    
    def check_if_feature_selection_needed(self, batch_dir: Path, new_rows_count: int, force: bool = False) -> Dict[str, Any]:
        """Checks if new feature selection is required."""
        # Check for initial run or forced selection
        initial_check = self._check_initial_or_forced_selection(batch_dir, force)
        if initial_check['needed']:
            return initial_check
        
        # Check if we have enough new data
        threshold = self.orchestrator.config.models_config.get('feature_selection', {}).get('new_data_threshold', 1000)
        if new_rows_count >= threshold:
            return {
                'needed': True,
                'reason': f'New data threshold reached: {new_rows_count} >= {threshold}',
                'threshold': threshold,
                'new_rows': new_rows_count
            }
        
        return {
            'needed': False,
            'reason': f'Insufficient new data: {new_rows_count} < {threshold}',
            'threshold': threshold,
            'new_rows': new_rows_count
        }
    
    def _check_initial_or_forced_selection(self, batch_dir: Path, force: bool = False) -> Dict[str, Any]:
        """Check for initial run or forced selection."""
        if force:
            return {
                'needed': True,
                'reason': 'Feature selection forced by user',
                'forced': True
            }
        
        # Check if this is the first run
        feature_files = list(batch_dir.glob("selected_features_*.json"))
        if not feature_files:
            return {
                'needed': True,
                'reason': 'Initial feature selection required',
                'initial': True
            }
        
        return {
            'needed': False,
            'reason': 'Feature selection already exists',
            'existing_files': len(feature_files)
        }
    
        
    async def prepare_colab_data(self, features_df: pd.DataFrame = None, 
                                  targets_df: pd.DataFrame = None,
                                  tickers: List[str] = None, 
                                  timeframes: List[str] = None,
                                  test_ticker: Optional[str] = None,
                                  test_target: Optional[str] = None,
                                  test_model: Optional[str] = None,
                                  epochs: Optional[int] = None,
                                  max_iterations: Optional[int] = None,
                                  batch_name: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Prepare data for Colab training.
        
        Args:
            features_df: Computed features DataFrame from local pipeline
            targets_df: Computed targets DataFrame from local pipeline
            tickers: List of ticker symbols
            timeframes: List of timeframes
            test_ticker: Optional test ticker for test mode
            test_target: Optional test target for test mode
            test_model: Optional test model for test mode
            epochs: Optional epochs for test mode
            max_iterations: Optional max iterations for test mode
            **kwargs: Additional parameters for batch preparation
        
        Returns:
            Dictionary with batch preparation results including batch directory and metadata
        """
        self.logger.info("   Delegating to colab_manager to save batch data...")
        
        # Use provided DataFrames or create empty ones
        features_df = features_df if features_df is not None else pd.DataFrame()
        targets_df = targets_df if targets_df is not None else pd.DataFrame()
        
        # Create BatchPreparationConfig
        from src.pipeline.hybrid.colab_manager import BatchPreparationConfig
        
        config = BatchPreparationConfig(
            tickers=tickers,
            timeframes=timeframes,
            batch_name=batch_name,
            accumulate=kwargs.get('accumulate', True),
            force_feature_selection=kwargs.get('force_feature_selection', False),
            # Test mode parameters
            test_ticker=test_ticker,
            test_target=test_target,
            test_model=test_model,
            epochs=epochs,
            max_iterations=max_iterations
        )
        
        # Delegate to colab_manager to package and save the data
        return self.orchestrator.colab_manager.prepare_colab_batch(
            features_df=features_df,
            targets_df=targets_df,
            config=config
        )
    
    def load_colab_results(self, batch_name: str) -> Dict[str, Any]:
        """Loads training results from Colab."""
        return self.orchestrator.colab_manager.load_colab_results(batch_name)
    
    def extract_batch_name_from_path(self, path_str: str) -> Optional[str]:
        """Extract batch name from path."""
        parts = Path(path_str.replace('/', '\\')).parts
        if 'accumulated' in parts:
            idx = parts.index('accumulated')
            if len(parts) > idx + 1: 
                return parts[idx + 1]
        return None
    
    async def run_full_hybrid_pipeline(self, tickers: Optional[List[str]] = None, 
                                      timeframes: Optional[List[str]] = None, 
                                      accumulate: bool = True, force_training: bool = False, 
                                      skip_colab: bool = False, 
                                      force_feature_selection: bool = False) -> Dict[str, Any]:
        """Run full hybrid pipeline with all parameters."""
        params = PipelineParams(
            tickers=tickers,
            timeframes=timeframes,
            accumulate=accumulate,
            force_training=force_training,
            skip_colab=skip_colab,
            force_feature_selection=force_feature_selection
        )
        
        return await self.orchestrator.pipeline_manager.run_full_hybrid_pipeline(params)
    
    async def run_final_stages(self, features_df: Optional[pd.DataFrame], targets_df: Optional[pd.DataFrame], 
                              colab_results: Optional[Dict[str, Any]] = None, 
                              light_results: Optional[Dict[str, Any]] = None, 
                              tickers: Optional[List[str]] = None, 
                              timeframes: Optional[List[str]] = None, 
                              batch_name: Optional[str] = None) -> Dict[str, Any]:
        """Run final stages of the pipeline."""
        params = FinalStagesParams(
            features_df=features_df,
            targets_df=targets_df,
            colab_results=colab_results,
            light_results=light_results,
            tickers=tickers,
            timeframes=timeframes,
            batch_name=batch_name
        )
        
        return await self.orchestrator.pipeline_manager.run_final_stages(params)
