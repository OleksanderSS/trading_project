# core/stages/stage_4_unified.py - Unified Stage 4 for light + heavy models

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from core.stages.stage_manager import StageManager
from utils.colab_utils import ColabUtils
from config.config import TICKERS, TIME_FRAMES

logger = logging.getLogger(__name__)


class UnifiedStage4:
    """
    Unified Stage 4 that combines light models (local) and heavy models (Colab)
    """
    
    def __init__(self):
        self.stage_manager = StageManager()
        self.colab_utils = ColabUtils()
        
        # Model definitions
        self.light_models = ['lgbm', 'rf', 'linear', 'mlp']
        self.heavy_models = ['gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder']
    
    def run_unified_stage_4(self, features_df: pd.DataFrame, tickers: Dict, timeframes: list) -> Dict[str, Any]:
        """
        Run unified Stage 4 with both light and heavy models
        
        Args:
            features_df: Features from Stage 3
            tickers: Dictionary of tickers
            timeframes: List of timeframes
            
        Returns:
            Dictionary with all model results
        """
        logger.info("[START] Unified Stage 4: Light + Heavy Models")
        logger.info(f"[DATA] Tickers: {list(tickers.keys())}")
        logger.info(f" Таймфрейми: {timeframes}")
        
        results = {
            'light_models': {},
            'heavy_models': {},
            'all_models': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'tickers': list(tickers.keys()),
                'timeframes': timeframes,
                'total_combinations': len(tickers) * len(timeframes) * (len(self.light_models) + len(self.heavy_models))
            }
        }
        
        # 1. Train light models locally
        logger.info("\n Stage 4a: Light Models (Local)")
        light_results = self._train_light_models(features_df, tickers, timeframes)
        results['light_models'] = light_results
        
        # 2. Load heavy models from Colab
        logger.info("\n[BRAIN] Stage 4b: Heavy Models (Colab)")
        heavy_results = self._load_heavy_models()
        results['heavy_models'] = heavy_results
        
        # 3. Combine all models
        logger.info("\n[DATA] Stage 4c: Combining Results")
        all_models = {**light_results, **heavy_results}
        results['all_models'] = all_models
        
        # 4. Statistics
        light_count = len(light_results)
        heavy_count = len(heavy_results)
        total_count = len(all_models)
        
        logger.info(f"\n[UP] Stage 4 Statistics:")
        logger.info(f"   Light models: {light_count}")
        logger.info(f"  [BRAIN] Heavy models: {heavy_count}")
        logger.info(f"  [DATA] Total models: {total_count}")
        
        # 5. Save unified results
        self._save_unified_results(results)
        
        return results
    
    def _train_light_models(self, features_df: pd.DataFrame, tickers: Dict, timeframes: list) -> Dict[str, Any]:
        """Train light models locally"""
        light_results = {}
        
        logger.info("  [REFRESH] Training light models...")
        
        # Import light models module
        try:
            from core.models.light_models import LightModelTrainer
            trainer = LightModelTrainer()
        except ImportError:
            logger.info("  [ERROR] Light models module not found - skipping light models")
            return light_results
        
        for ticker in tickers.keys():
            for timeframe in timeframes:
                for model_type in self.light_models:
                    try:
                        logger.info(f"    [REFRESH] {model_type.upper()} - {ticker} - {timeframe}")
                        
                        model_results = trainer.train_light_model(
                            features_df=features_df,
                            model_type=model_type,
                            ticker=ticker,
                            timeframe=timeframe
                        )
                        
                        if model_results:
                            key = f"{model_type}_{ticker}_{timeframe}"
                            light_results[key] = {
                                'model_type': 'light',
                                'model_name': model_type,
                                'ticker': ticker,
                                'timeframe': timeframe,
                                'results': model_results,
                                'accuracy': getattr(model_results, 'accuracy', 0),
                                'timestamp': datetime.now().isoformat()
                            }
                            logger.info(f"    [OK] {model_type.upper()} {ticker} {timeframe} completed")
                        else:
                            logger.info(f"    [ERROR] {model_type.upper()} {ticker} {timeframe} failed")
                            
                    except Exception as e:
                        logger.info(f"    [ERROR] Error {model_type} {ticker} {timeframe}: {e}")
        
        return light_results
    
    def _load_heavy_models(self) -> Dict[str, Any]:
        """Load heavy models from Colab results"""
        heavy_results = {}
        
        try:
            # Try to load from Colab results directory
            results_dir = "data/colab/results"
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.endswith('.parquet')]
                
                if files:
                    logger.info(f"   Found {len(files)} Colab result files")
                    
                    for file in files:
                        try:
                            file_path = os.path.join(results_dir, file)
                            df = pd.read_parquet(file_path)
                            
                            # Convert DataFrame to dictionary format
                            for _, row in df.iterrows():
                                key = f"{row['model']}_{row['ticker']}_{row['timeframe']}"
                                heavy_results[key] = {
                                    'model_type': 'heavy',
                                    'model_name': row['model'],
                                    'ticker': row['ticker'],
                                    'timeframe': row['timeframe'],
                                    'results': row.to_dict(),
                                    'accuracy': row.get('accuracy', row.get('mse', 0)),
                                    'timestamp': row.get('timestamp', datetime.now().isoformat())
                                }
                            
                            logger.info(f"  [OK] Loaded {len(df)} heavy model results")
                            
                        except Exception as e:
                            logger.info(f"  [ERROR] Error loading {file}: {e}")
                else:
                    logger.info("  [WARN] No Colab result files found")
            else:
                logger.info("  [WARN] Colab results directory not found")
                
        except Exception as e:
            logger.info(f"  [ERROR] Error loading heavy models: {e}")
        
        return heavy_results
    
    def _save_unified_results(self, results: Dict[str, Any]):
        """Save unified Stage 4 results"""
        try:
            # Create output directory
            output_dir = "data/unified_stage_4"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save all models as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert to serializable format
            serializable_results = {}
            for key, value in results['all_models'].items():
                if hasattr(value.get('results', {}), 'to_dict'):
                    results_dict = value['results'].to_dict()
                else:
                    results_dict = value.get('results', {})
                
                serializable_results[key] = {
                    'model_type': value['model_type'],
                    'model_name': value['model_name'],
                    'ticker': value['ticker'],
                    'timeframe': value['timeframe'],
                    'results': results_dict,
                    'accuracy': value['accuracy'],
                    'timestamp': value['timestamp']
                }
            
            # Save JSON
            json_path = os.path.join(output_dir, f"unified_models_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Save metadata
            metadata = {
                **results['metadata'],
                'light_model_count': len(results['light_models']),
                'heavy_model_count': len(results['heavy_models']),
                'total_model_count': len(results['all_models']),
                'output_path': json_path
            }
            
            metadata_path = os.path.join(output_dir, f"unified_models_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\n Unified Stage 4 results saved:")
            logger.info(f"   Models: {json_path}")
            logger.info(f"   Metadata: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving unified results: {e}")
    
    def get_model_comparison(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Get comparison of all models"""
        comparison_data = []
        
        for key, model_info in results['all_models'].items():
            comparison_data.append({
                'model_key': key,
                'model_type': model_info['model_type'],
                'model_name': model_info['model_name'],
                'ticker': model_info['ticker'],
                'timeframe': model_info['timeframe'],
                'accuracy': model_info['accuracy'],
                'timestamp': model_info['timestamp']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_models(self, results: Dict[str, Any], top_n: int = 5) -> pd.DataFrame:
        """Get top N best models"""
        comparison_df = self.get_model_comparison(results)
        
        # Sort by accuracy (descending)
        best_models = comparison_df.sort_values('accuracy', ascending=False).head(top_n)
        
        return best_models


# Global instance for backward compatibility
unified_stage_4 = UnifiedStage4()


def run_unified_stage_4(features_df: pd.DataFrame, tickers: Dict = None, timeframes: list = None) -> Dict[str, Any]:
    """
    Run unified Stage 4 with both light and heavy models
    
    Args:
        features_df: Features from Stage 3
        tickers: Dictionary of tickers (default from config)
        timeframes: List of timeframes (default from config)
        
    Returns:
        Dictionary with all model results
    """
    if tickers is None:
        tickers = TICKERS
    if timeframes is None:
        timeframes = TIME_FRAMES
    
    return unified_stage_4.run_unified_stage_4(features_df, tickers, timeframes)


if __name__ == "__main__":
    # Test function
    logger.info(" Testing Unified Stage 4...")
    
    # Load test data
    try:
        from utils.data_storage import load_from_storage
        features_df = load_from_storage("data/stages/merged_full.parquet")
        
        if not features_df.empty:
            results = run_unified_stage_4(features_df)
            logger.info("[OK] Unified Stage 4 test completed")
        else:
            logger.info("[ERROR] No test data found")
            
    except Exception as e:
        logger.info(f"[ERROR] Test error: {e}")