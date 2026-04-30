"""
Final Stages Executor - Handles final stages execution
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator

logger = ProjectLogger.get_logger(__name__)


class FinalStagesExecutor:
    """Handles execution of final pipeline stages."""
    
    def __init__(self, config_manager, output_dir: str, batch_name: str):
        self.config_manager = config_manager
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(__name__)
    
    async def run_final_stages(self, features_df, targets_df, colab_results: Optional[Dict[str, Any]] = None,
                              light_results: Optional[Dict[str, Any]] = None, tickers: Optional[List[str]] = None,
                              timeframes: Optional[List[str]] = None, batch_name: Optional[str] = None) -> Dict[str, Any]:
        """Runs final stages 4-7 after Colab results are loaded."""
        batch_name, stages_to_run = self._prepare_final_stages_params(colab_results, batch_name, [5, 6, 7])
        
        self.logger.info(f"Running final stages {stages_to_run} for batch: {batch_name}")
        
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=stages_to_run
        )
        
        start_time = time.time()
        
        results = await orchestrator.run(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            timeframes=timeframes,
            run_mode='train',
            colab_results=colab_results,
            light_results=light_results
        )
        
        duration = time.time() - start_time
        
        # Build models metadata
        models_metadata = self._build_models_metadata(colab_results, light_results)
        
        # Create final summary
        final_summary = self._create_final_summary(results, models_metadata, duration, tickers)
        
        # Save final results
        final_results_path = await self._save_final_results(final_summary)
        
        self.logger.info(f"Final stages completed in {duration:.1f}s")
        
        return {
            'status': 'final_stages_complete',
            'results': results,
            'models_metadata': models_metadata,
            'final_summary': final_summary,
            'final_results_path': str(final_results_path),
            'duration_seconds': duration
        }
    
    def _prepare_final_stages_params(self, colab_results: Optional[Dict[str, Any]], batch_name: Optional[str],
                                    stages_to_run: Optional[List[int]]) -> Tuple[str, List[int]]:
        """Prepare and validate parameters for final stages."""
        if colab_results is None:
            colab_results = {}
        
        batch_name = batch_name or colab_results.get('batch_name', self.batch_name)
        stages_to_run = stages_to_run or [5, 6, 7]
        
        # Ensure stage 5 is included if stages 6 or 7 are requested
        if 6 in stages_to_run or 7 in stages_to_run:
            stages_to_run = sorted(set(stages_to_run) | {5})
        
        return batch_name, stages_to_run
    
    def _build_models_metadata(self, colab_results: Dict[str, Any],
                              light_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive models metadata from all sources."""
        models_metadata = {}
        
        # Add Colab heavy models metadata
        if colab_results and 'models_metadata' in colab_results:
            models_metadata.update(colab_results['models_metadata'])
        
        # Add light models metadata
        if light_results and 'models_metadata' in light_results:
            models_metadata.update(light_results['models_metadata'])
        
        # Add metadata from accumulated results
        accumulated_results_path = self.output_dir / "light_models_results.json"
        if accumulated_results_path.exists():
            try:
                with open(accumulated_results_path, 'r', encoding='utf-8') as f:
                    accumulated = json.load(f)
                
                if 'runs' in accumulated:
                    for run in accumulated['runs']:
                        if 'models_metadata' in run:
                            models_metadata.update(run['models_metadata'])
            except Exception as e:
                self.logger.warning(f"Could not load accumulated results: {e}")
        
        return models_metadata
    
    def _create_final_summary(self, results: Dict[str, Any], models_metadata: Dict[str, Any],
                             duration: float, tickers: Optional[List[str]]) -> Dict[str, Any]:
        """Create final summary of pipeline execution."""
        return {
            'timestamp': datetime.now().isoformat(),
            'batch_name': self.batch_name,
            'tickers': tickers or [],
            'models_trained': list(models_metadata.keys()),
            'models_count': len(models_metadata),
            'pipeline_results': results,
            'duration_seconds': duration,
            'status': 'completed'
        }
    
    async def _save_final_results(self, final_summary: Dict[str, Any]) -> Path:
        """Save final results to JSON file."""
        import aiofiles
        
        output_path = self.output_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        content = json.dumps(final_summary, indent=2, default=str)
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        return output_path
