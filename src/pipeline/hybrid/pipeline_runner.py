"""
Pipeline Runner - Handles pipeline execution logic
"""

import time
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.pipeline.hybrid.metadata_manager import MetadataManager, MetadataParams

logger = ProjectLogger.get_logger(__name__)


class PipelineRunner:
    """Handles pipeline execution and result management."""
    
    def __init__(self, config_manager, output_dir: str, batch_name: str, 
                 feature_processor, metadata_manager: MetadataManager):
        self.config_manager = config_manager
        self.output_dir = Path(output_dir)
        self.batch_name = batch_name
        self.feature_processor = feature_processor
        self.metadata_manager = metadata_manager
        self.logger = ProjectLogger.get_logger(__name__)
    
    async def run_local_pipeline(self, tickers: Optional[List[str]] = None,
                                 timeframes: Optional[List[str]] = None,
                                 stages_to_run: Optional[List[int]] = None) -> Dict[str, Any]:
        """Executes the local part of the pipeline (stages 0-3 + light models)."""
        start_time = time.time()
        self.logger.info("Launch local pipeline...")
        
        local_stages = stages_to_run or [0, 1, 2, 3]  # Include Stage 3 for full pipeline
        results = await self._run_pipeline_stages(tickers, timeframes, local_stages)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = self._save_pipeline_results(results, timestamp)
        
        metadata_params = MetadataParams(
            timestamp=timestamp,
            tickers=tickers,
            timeframes=timeframes,
            stages=local_stages,
            saved_files=saved_files,
            batch_name=self.batch_name
        )
        metadata = self.metadata_manager.create_pipeline_metadata(metadata_params)
        self._save_metadata(metadata, timestamp)
        
        total_duration = time.time() - start_time
        self._log_pipeline_completion(total_duration)
        
        return {
            'status': 'local_complete',
            'results': results,
            'saved_files': saved_files,
            'metadata_path': str(self.output_dir / f"{self.batch_name}_metadata_{timestamp}.json"),
            'timestamp': timestamp,
            'duration_seconds': total_duration
        }
    
    async def _run_pipeline_stages(self, tickers: Optional[List[str]], 
                                   timeframes: Optional[List[str]], 
                                   stages: List[int]) -> Dict[str, Any]:
        """Runs pipeline stages."""
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=stages
        )
        
        stage_start = time.time()
        results = await orchestrator.run(
            tickers=tickers,
            timeframes=timeframes,
            run_mode='train'
        )
        stage_duration = time.time() - stage_start
        self.logger.info(f"Stages {stages} completed in {stage_duration:.1f}s")
        
        return results
    
    def _save_pipeline_results(self, results: Dict[str, Any], timestamp: str) -> Dict[str, str]:
        """Saves results from pipeline stages."""
        saved_files = {}
        
        if not results:
            return saved_files
            
        if 'raw_data' in results:
            saved_files['raw_data'] = str(self._save_stage_result(
                results['raw_data'], f"{self.batch_name}_stage1_raw_data_{timestamp}.parquet"
            ))
        
        if 'cleaned_data' in results:
            saved_files['cleaned_data'] = str(self._save_stage_result(
                results['cleaned_data'], f"{self.batch_name}_stage2_cleaned_data_{timestamp}.parquet"
            ))
        
        if 'enriched_data' in results:
            enriched_result = self._process_enriched_data(results['enriched_data'])
            if enriched_result:
                proc_data = enriched_result['data']
                results['enriched_data'] = proc_data['data']
                results['features_df'] = proc_data['features']
                results['targets_df'] = proc_data['targets']
                saved_files['features'] = enriched_result['paths']['features']
        
        return saved_files
    
    def _save_stage_result(self, data: Any, filename: str):
        """Save stage result to file."""
        from pathlib import Path
        import pickle
        
        path = self.output_dir / filename
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, compression='snappy')
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        return path
    
    def _process_enriched_data(self, enriched_data: Any) -> Optional[Dict[str, Any]]:
        """Handle enriched data processing."""
        processed_data = self.feature_processor.process_enriched_data(enriched_data)
        if processed_data is None:
            return None
        
        return {
            'data': processed_data,
            'paths': self._save_enriched_data(processed_data)
        }
    
    def _save_enriched_data(self, processed_data: Any) -> Dict[str, str]:
        """Save enriched data to files."""
        from pathlib import Path
        
        # ✅ FIX: output_dir already includes batch_name!
        batch_dir = self.output_dir
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        save_result = self.feature_processor.save_enriched_data(processed_data, batch_dir)
        
        return {
            'features': str(save_result['features_path']),
            'targets': str(save_result['targets_path'])
        }
    
    def _save_metadata(self, metadata: Dict[str, Any], timestamp: str) -> None:
        """Save metadata to files."""
        from pathlib import Path
        
        metadata_path = self.output_dir / f"{self.batch_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        # Save batch metadata
        batch_metadata_path = self.output_dir / "batch_metadata.json"
        batch_metadata = self._create_batch_metadata_dict(metadata, timestamp)
        with open(batch_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(batch_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved: {metadata_path}")
    
    def _create_batch_metadata_dict(self, metadata: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """Create batch metadata dictionary."""
        return {
            'batch_name': self.batch_name,
            'timestamp': timestamp,
            'stages_completed': metadata.get('stages', []),
            'files': metadata.get('saved_files', [])
        }
    
    def _log_pipeline_completion(self, total_duration: float) -> None:
        """Log pipeline completion."""
        self.logger.info(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
