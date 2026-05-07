"""
Pipeline Manager for Hybrid Orchestrator.
Handles pipeline execution and coordination.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, cast
import pandas as pd

from src.core.logging.logger import ProjectLogger
from .pipeline_config import PipelineParams, FinalStagesParams, ColabBatchParams

logger = ProjectLogger.get_logger(__name__)


class PipelineManager:
    """Manages pipeline execution for hybrid orchestrator."""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = ProjectLogger.get_logger(__name__)
    
    async def run_full_hybrid_pipeline(self, params: Optional[PipelineParams] = None) -> Dict[str, Any]:
        """Full hybrid pipeline with smart caching logic."""
        if params is None:
            params = PipelineParams()
            
        self.logger.info(f"Launching full hybrid pipeline for batch: {self.orchestrator.batch_name}")
        
        # Step 1: Collect local data
        local_res = await self._collect_local_data(params.tickers, params.timeframes)
        if local_res['status'] != 'local_complete':
            return local_res
            
        # Step 2: Check cache and handle data
        n_f, n_t = self._handle_data_caching(local_res, params.force_training)
        if n_f is None:
            return {'status': 'no_data', 'message': 'No data collected'}
            
        # Step 3: Prepare Colab package
        self.logger.info("Preparing Colab package...")
        b_info = self.orchestrator.colab_manager.prepare_colab_batch(
            n_f, n_t, params.tickers or [], params.timeframes or [], 
            accumulate=params.accumulate, force_feature_selection=params.force_feature_selection
        )
        
        # Step 4: Handle Colab or skip path
        if params.skip_colab:
            return await self._handle_skip_colab_path(b_info, n_f, params.tickers, params.timeframes)
        else:
            return self._handle_colab_path(b_info)
    
    async def run_final_stages(self, params: Optional[FinalStagesParams] = None) -> Dict[str, Any]:
        """Run final stages 4-7 of pipeline."""
        if params is None:
            params = FinalStagesParams()
            
        # Delegate to final_stages_orchestrator for real execution
        from src.pipeline.hybrid_orchestrator import HybridFinalStagesRequest
        
        request = HybridFinalStagesRequest(
            features_df=params.features_df,
            targets_df=params.targets_df,
            colab_results=params.colab_results,
            light_results=params.light_results,
            tickers=params.tickers,
            timeframes=params.timeframes,
            batch_name=params.batch_name or self.orchestrator.batch_name
        )
        
        return cast(Dict[str, Any], await self.orchestrator.final_stages_orchestrator.run_final_stages(request))
    
    async def _collect_local_data(self, tickers: Optional[List[str]], 
                                  timeframes: Optional[List[str]]) -> Dict[str, Any]:
        """Collect local pipeline data."""
        self.logger.info("Collecting new data...")
        return cast(Dict[str, Any], await self.orchestrator.run_local_pipeline(tickers, timeframes))
    
    def _handle_data_caching(self, local_res: Dict[str, Any], force_training: bool) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Handle data caching logic."""
        return cast(Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]], self.orchestrator.cache_manager.handle_data_caching(
            local_res, force_training, self.orchestrator.batch_name, self.orchestrator.output_dir
        ))
    
    async def _handle_skip_colab_path(self, b_info: Dict[str, Any], n_f: pd.DataFrame, 
                                     tickers: Optional[List[str]], 
                                     timeframes: Optional[List[str]]) -> Dict[str, Any]:
        """Handle skip Colab path."""
        self._create_fallback_selected_features(b_info, n_f)
        final_results = await self.run_final_stages()
        return {'status': 'completed_without_colab', 'final_results': final_results}
    
    def _handle_colab_path(self, b_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Colab training path."""
        instr = self._generate_colab_instructions(b_info)
        self.logger.info(f"PAUSED: Colab training required.\n{instr}")
        return {'status': 'paused_for_colab', 'colab_batch': b_info, 'colab_instructions': instr}
    
    def _create_fallback_selected_features(self, batch_info: Dict[str, Any], features_df: pd.DataFrame):
        """Create fallback selected features when skipping Colab."""
        batch_dir = Path(batch_info['batch_dir'])
        selected_features = list(features_df.columns)
        
        # Save fallback features
        import json
        features_file = batch_dir / 'selected_features_fallback.json'
        with open(features_file, 'w') as f:
            json.dump({'features': selected_features, 'method': 'fallback'}, f, indent=2)
    
    def _generate_colab_instructions(self, batch_info: Dict[str, Any]) -> str:
        """Generates instructions for running in Colab."""
        name = batch_info['batch_name']
        return f"""
COLAB INSTRUCTIONS:
1. Transfer the batch folder '{name}' to your Google Drive.
2. Run the Colab notebook and mount your drive.
3. Perform feature selection and heavy model training.
4. Once finished, run: python run_hybrid_pipeline.py --mode continue --batch-name {name}
"""
    
    async def _run_light_models(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                               tickers: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """Run light models training."""
        self.logger.info("Training light models...")
        
        # This would integrate with the actual light model training
        # For now, return placeholder results
        return {
            'models_trained': ['light_model_1', 'light_model_2'],
            'performance': {'accuracy': 0.85, 'f1_score': 0.82},
            'status': 'completed'
        }
