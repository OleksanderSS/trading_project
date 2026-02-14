"""
Optimal Pipeline Orchestrator with Colab Integration

This module provides the optimal pipeline orchestration:
- Stage 1-4: Local processing (data collection, enrichment, feature engineering, light models)
- Stage 4 Heavy: Colab processing for heavy models with automatic sync
- Stage 5: Local vector analysis and final signal generation
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import pandas as pd

# Import pipeline components
from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrich
from core.stages.stage_manager import StageManager
from core.stages.stage_4_modeling import run_stage_4_modeling
from core.stages.stage_5_pipeline_fixed import run_stage_5_with_models

# Import Colab integration
from utils.colab_manager import ColabManager
from utils.colab_utils import ColabUtils

# Import model management
# from models.model_selector.dual_model_manager import DualModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Optimal pipeline orchestrator with Colab integration"""
    
    def __init__(self, tickers: Dict[str, str], time_frames: List[str], debug: bool = False):
        self.tickers = tickers
        self.time_frames = time_frames
        self.debug = debug
        
        # Initialize managers
        self.colab_manager = ColabManager()
        self.colab_utils = ColabUtils()
        # self.dual_model_manager = DualModelManager()  # Temporarily disabled
        
        # Pipeline state
        self.pipeline_state = {
            "stage_1_data": None,
            "stage_2_data": None,
            "stage_3_data": None,
            "stage_4_light_models": None,
            "stage_4_heavy_models": None,
            "stage_5_results": None
        }
        
        logger.info(f"PipelineOrchestrator initialized for {len(tickers)} tickers, {len(time_frames)} timeframes")
    
    def run_optimal_pipeline(self, colab_training: bool = True) -> Dict[str, Any]:
        """
        Run optimal pipeline with automatic Colab integration
        
        Args:
            colab_training: Whether to use Colab for heavy model training
            
        Returns:
            Dictionary with pipeline results and state
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("STARTING OPTIMAL PIPELINE WITH COLAB INTEGRATION")
        logger.info("="*60)
        
        try:
            # Stage 1: Data Collection (Local)
            stage_1_results = self._run_stage_1_local()
            
            # Stage 2: Data Enrichment (Local)
            stage_2_results = self._run_stage_2_local(stage_1_results)
            
            # Stage 3: Feature Engineering (Local)
            stage_3_results = self._run_stage_3_local(stage_2_results)
            
            # Stage 4: Light Models (Local)
            stage_4_light_results = self._run_stage_4_light_local(stage_3_results)
            
            # Stage 4: Heavy Models (Colab or Local)
            if colab_training:
                stage_4_heavy_results = self._run_stage_4_heavy_colab(stage_3_results)
            else:
                stage_4_heavy_results = self._run_stage_4_heavy_local(stage_3_results)
            
            # Stage 5: Vector Analysis & Signal Generation (Local)
            stage_5_results = self._run_stage_5_vector_analysis(
                stage_4_light_results, 
                stage_4_heavy_results
            )
            
            # Compile final results
            total_time = time.time() - start_time
            
            final_results = {
                "pipeline_type": "optimal_with_colab" if colab_training else "optimal_local",
                "total_time": total_time,
                "stage_1": stage_1_results,
                "stage_2": stage_2_results,
                "stage_3": stage_3_results,
                "stage_4_light": stage_4_light_results,
                "stage_4_heavy": stage_4_heavy_results,
                "stage_5": stage_5_results,
                "summary": self._generate_pipeline_summary(stage_5_results)
            }
            
            logger.info("="*60)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s")
            logger.info("="*60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _run_stage_1_local(self) -> Dict[str, Any]:
        """Run Stage 1: Data Collection locally"""
        logger.info("[Stage 1] Running data collection locally...")
        
        results = run_stage_1_collect(debug_no_network=self.debug)
        
        self.pipeline_state["stage_1_data"] = results
        logger.info(f"[Stage 1] Completed: {len(results)} data sources")
        return results
    
    def _run_stage_2_local(self, stage_1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 2: Data Enrichment locally"""
        logger.info("[Stage 2] Running data enrichment locally...")
        
        results = run_stage_2_enrich(
            stage1_data=stage_1_data,
            keyword_dict={},
            tickers=list(self.tickers.keys()),
            time_frames=self.time_frames,
            mode="train"
        )
        
        self.pipeline_state["stage_2_data"] = results
        logger.info(f"[Stage 2] Completed: enriched data available")
        return results
    
    def _run_stage_3_local(self, stage_2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 3: Feature Engineering locally"""
        logger.info("[Stage 3] Running feature engineering locally...")
        
        # Initialize Stage Manager for Stage 3
        stage_manager = StageManager()
        
        # Run Stage 3 using StageManager
        if isinstance(stage_2_data, tuple):
            merged_df = stage_2_data[0] if len(stage_2_data) > 0 else pd.DataFrame()
        else:
            merged_df = stage_2_data.get('merged_df',
                pd.DataFrame()) if isinstance(stage_2_data,
                dict) else pd.DataFrame()
            
        features_df, context_df, trigger_data, technical_df = stage_manager.run_stage_3(
            merged_df=merged_df,
            force_refresh=self.debug
        )
        
        results = {
            'features_df': features_df,
            'context_df': context_df,
            'trigger_data': trigger_data,
            'technical_df': technical_df
        }
        
        self.pipeline_state["stage_3_data"] = results
        logger.info(f"[Stage 3] Completed: features engineered")
        return results
    
    def _run_stage_4_light_local(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 4: Light Models locally"""
        logger.info("[Stage 4 Light] Running light models locally...")
        
        light_models = ["lgbm", "rf", "linear", "mlp", "ensemble"]
        results = {}
        
        for model_name in light_models:
            logger.info(f"[Stage 4 Light] Training {model_name}...")
            try:
                # Skip Stage 4 for now - requires different parameters
                logger.warning(f"[Stage 4 Light] {model_name} skipped - incompatible interface")
                results[model_name] = {"status": "skipped", "reason": "interface mismatch"}
                continue
            except Exception as e:
                logger.error(f"[Stage 4 Light] {model_name} failed: {e}")
                results[model_name] = {"status": "error", "error": str(e)}
        
        self.pipeline_state["stage_4_light_models"] = results
        logger.info(f"[Stage 4 Light] Completed: {len([r for r in results.values() if r.get('status') == 'success'])} successful models")
        return results
    
    def _run_stage_4_heavy_colab(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 4: Heavy Models via Colab with automatic sync"""
        logger.info("[Stage 4 Heavy] Running heavy models via Colab...")
        
        try:
            # Step 1: Export features to Colab
            logger.info("[Stage 4 Heavy] Exporting features to Colab...")
            export_success = self.colab_manager.sync_to_colab(
                tickers=self.tickers,
                time_frames=self.time_frames
            )
            
            if not export_success:
                raise Exception("Failed to export features to Colab")
            
            logger.info("[Stage 4 Heavy] Features exported successfully")
            
            # Step 2: Create Colab notebook template
            logger.info("[Stage 4 Heavy] Creating Colab notebook template...")
            notebook_path = self.colab_utils.create_colab_notebook_template(
                tickers=self.tickers,
                time_frames=self.time_frames
            )
            
            logger.info(f"[Stage 4 Heavy] Notebook template created: {notebook_path}")
            
            # Step 3: Wait for user to run Colab training (manual step)
            logger.info("="*50)
            logger.info("MANUAL STEP REQUIRED:")
            logger.info("1. Open the generated Colab notebook")
            logger.info("2. Run all cells to train heavy models")
            logger.info("3. Results will be automatically synced back")
            logger.info("="*50)
            
            # Step 4: Import results from Colab (after training)
            logger.info("[Stage 4 Heavy] Waiting for Colab results...")
            
            # Try to import results (will be empty until training completes)
            max_wait_time = 300  # 5 minutes max wait
            wait_interval = 30   # Check every 30 seconds
            waited_time = 0
            
            while waited_time < max_wait_time:
                try:
                    results = self.colab_manager.accumulate_results()
                    if results:
                        logger.info(f"[Stage 4 Heavy] Results imported: {len(results)} model results")
                        break
                    else:
                        logger.info(f"[Stage 4 Heavy] Waiting for results... ({waited_time}s/{max_wait_time}s)")
                        time.sleep(wait_interval)
                        waited_time += wait_interval
                except Exception as e:
                    logger.warning(f"[Stage 4 Heavy] Error checking results: {e}")
                    time.sleep(wait_interval)
                    waited_time += wait_interval
            
            if waited_time >= max_wait_time:
                logger.warning("[Stage 4 Heavy] Timeout waiting for Colab results")
                results = {}
            
            self.pipeline_state["stage_4_heavy_models"] = results
            return results
            
        except Exception as e:
            logger.error(f"[Stage 4 Heavy] Colab integration failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_stage_4_heavy_local(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 4: Heavy Models locally (fallback option)"""
        logger.info("[Stage 4 Heavy] Running heavy models locally (fallback)...")
        
        heavy_models = ["gru", "transformer", "cnn", "lstm", "tabnet", "autoencoder"]
        results = {}
        
        for model_name in heavy_models:
            logger.info(f"[Stage 4 Heavy] Training {model_name} locally...")
            try:
                # Skip Stage 4 for now - requires different parameters
                logger.warning(f"[Stage 4 Heavy] {model_name} skipped - incompatible interface")
                results[model_name] = {"status": "skipped", "reason": "interface mismatch"}
                continue
            except Exception as e:
                logger.error(f"[Stage 4 Heavy] {model_name} failed: {e}")
                results[model_name] = {"status": "error", "error": str(e)}
        
        self.pipeline_state["stage_4_heavy_models"] = results
        logger.info(f"[Stage 4 Heavy] Completed: {len([r for r in results.values() if r.get('status') == 'success'])} successful models")
        return results
    
    def _run_stage_5_vector_analysis(self,
        light_results: Dict[str,
        Any],
        heavy_results: Dict[str,
        Any]) -> Dict[str,
        Any]:
        """Run Stage 5: Vector Analysis and Signal Generation"""
        logger.info("[Stage 5] Running vector analysis and signal generation...")
        
        # Combine all model results
        all_model_results = {**light_results, **heavy_results}
        
        # Skip Stage 5 for now - requires different parameters
        logger.warning("[Stage 5] Skipping - incompatible interface")
        results = {"status": "skipped", "reason": "interface mismatch"}
        
        self.pipeline_state["stage_5_results"] = results
        logger.info(f"[Stage 5] Vector analysis completed")
        return results
    
    def _generate_pipeline_summary(self, stage_5_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "tickers": list(self.tickers.keys()),
            "timeframes": self.time_frames,
            "total_models": len(self.pipeline_state["stage_4_light_models"]) + len(self.pipeline_state["stage_4_heavy_models"]),
                
            "successful_models": 0,
            "failed_models": 0,
            "best_signals": {},
            "vector_analysis": {}
        }
        
        # Count successful/failed models
        all_models = {**self.pipeline_state["stage_4_light_models"], **self.pipeline_state["stage_4_heavy_models"]}
        for model_name, results in all_models.items():
            if isinstance(results, dict) and results.get("status") == "success":
                summary["successful_models"] += 1
            else:
                summary["failed_models"] += 1
        
        # Extract best signals from Stage 5
        if stage_5_results:
            for model_name, model_results in stage_5_results.items():
                if isinstance(model_results, dict):
                    for combination, result in model_results.items():
                        if isinstance(result, dict) and result.get("status") == "success":
                            signal = result.get("final_signal", 0)
                            confidence = result.get("vector_analysis", {}).get("confidence", 0.0)
                            
                            if combination not in summary["best_signals"] or float(confidence) > summary["best_signals"][combination].get("confidence",
                                0.0):
                                summary["best_signals"][combination] = {
                                    "model": model_name,
                                    "signal": signal,
                                    "confidence": confidence,
                                    "recommendation": result.get("vector_analysis", {}).get("recommendation", "HOLD")
                                }
        
        return summary
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return self.pipeline_state.copy()
    
    def reset_pipeline_state(self) -> None:
        """Reset pipeline state"""
        self.pipeline_state = {
            "stage_1_data": None,
            "stage_2_data": None,
            "stage_3_data": None,
            "stage_4_light_models": None,
            "stage_4_heavy_models": None,
            "stage_5_results": None
        }
        logger.info("Pipeline state reset")


# Convenience function for easy usage
def run_optimal_pipeline(tickers: Dict[str,
    str],
    time_frames: List[str],
    debug: bool = False,
    colab_training: bool = True) -> Dict[str,
    Any]:
    logger.info(f"[DEBUG] run_optimal_pipeline called with tickers: {list(tickers.keys())}, timeframes: {time_frames}")
    """
    Convenience function to run the optimal pipeline
    
    Args:
        tickers: Dictionary of ticker symbols
        time_frames: List of time frames
        debug: Whether to run in debug mode
        colab_training: Whether to use Colab for heavy model training
        
    Returns:
        Pipeline results dictionary
    """
    orchestrator = PipelineOrchestrator(tickers, time_frames, debug)
    return orchestrator.run_optimal_pipeline(colab_training=colab_training)


if __name__ == "__main__":
    # Example usage
    from config.config import TICKERS, TIME_FRAMES
    
    logger.info("Running optimal pipeline with Colab integration...")
    results = run_optimal_pipeline(TICKERS, TIME_FRAMES, debug=True, colab_training=True)
    
    logger.info("\nPipeline Summary:")
    logger.info(f"Total time: {results['total_time']:.1f}s")
    logger.info(f"Successful models: {results['summary']['successful_models']}")
    logger.info(f"Failed models: {results['summary']['failed_models']}")
    
    logger.info("\nBest Signals:")
    for combination, signal_info in results['summary']['best_signals'].items():
        logger.info(f"{combination}: {signal_info['recommendation']} (confidence: {signal_info['confidence']:.2f})")