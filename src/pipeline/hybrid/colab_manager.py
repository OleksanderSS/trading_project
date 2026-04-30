"""
Colab Manager for Hybrid Orchestrator.
Handles all Colab-related operations including batch preparation and result loading.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Constants
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"
BATCH_METADATA_FILE = "batch_metadata.json"
SELECTED_FEATURES_PATTERN = "selected_features_*.json"


@dataclass
class BatchPreparationConfig:
    """Configuration for preparing a Colab batch to avoid excessive arguments."""
    tickers: List[str]
    timeframes: List[str]
    batch_name: Optional[str] = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False
    
    # Test mode parameters (optional - only for test mode)
    test_ticker: Optional[str] = None
    test_target: Optional[str] = None
    test_model: Optional[str] = None
    epochs: Optional[int] = None
    max_iterations: Optional[int] = None


class ColabManager:
    """Manages Colab-related operations for hybrid pipeline."""
    
    def __init__(self, output_dir: Path, batch_name: str):
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(__name__)
    
    def prepare_colab_batch(self, 
                            features_df: pd.DataFrame, 
                            targets_df: pd.DataFrame, 
                            config: BatchPreparationConfig) -> Dict[str, Any]:
        """Prepare data package for Colab training."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resolve batch name for metadata (but don't use for path!)
        base_name = config.batch_name or self.batch_name
        eff_batch_name = base_name.replace('target_target_', 'target_')
        
        # ✅ FIX: output_dir already includes batch_name!
        # Don't add batch_name again
        batch_dir = self.output_dir
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data files with accumulation logic
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        
        # Accumulate data if files exist and accumulate=True
        if config.accumulate and features_path.exists() and targets_path.exists():
            # Load existing data
            existing_features = pd.read_parquet(features_path)
            existing_targets = pd.read_parquet(targets_path)
            
            # Combine with new data
            combined_features = pd.concat([existing_features, features_df], ignore_index=True)
            combined_targets = pd.concat([existing_targets, targets_df], ignore_index=True)
            
            # Remove duplicates based on index columns
            if 'datetime' in combined_features.columns:
                combined_features = combined_features.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
            if 'datetime' in combined_targets.columns:
                combined_targets = combined_targets.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
            
            self.logger.info(f"Accumulated data: {len(existing_features)}→{len(combined_features)} features, {len(existing_targets)}→{len(combined_targets)} targets")
            
            # ✅ FIX: Ensure datetime is preserved as a column before saving
            if 'datetime' not in combined_features.columns and isinstance(combined_features.index, pd.DatetimeIndex):
                combined_features = combined_features.reset_index()
                if 'index' in combined_features.columns:
                    combined_features = combined_features.rename(columns={'index': 'datetime'})
            
            if 'datetime' not in combined_targets.columns and isinstance(combined_targets.index, pd.DatetimeIndex):
                combined_targets = combined_targets.reset_index()
                if 'index' in combined_targets.columns:
                    combined_targets = combined_targets.rename(columns={'index': 'datetime'})
            
            # Save combined data
            combined_features.to_parquet(features_path, index=False)
            combined_targets.to_parquet(targets_path, index=False)
        else:
            # ✅ FIX: Ensure datetime is preserved as a column before saving
            if 'datetime' not in features_df.columns and isinstance(features_df.index, pd.DatetimeIndex):
                features_df = features_df.reset_index()
                if 'index' in features_df.columns:
                    features_df = features_df.rename(columns={'index': 'datetime'})
            
            if 'datetime' not in targets_df.columns and isinstance(targets_df.index, pd.DatetimeIndex):
                targets_df = targets_df.reset_index()
                if 'index' in targets_df.columns:
                    targets_df = targets_df.rename(columns={'index': 'datetime'})
            
            # Save new data
            features_df.to_parquet(features_path, index=False)
            targets_df.to_parquet(targets_path, index=False)
            self.logger.info(f"Created new batch: {len(features_df)} features, {len(targets_df)} targets")
        
        # Get final data sizes after accumulation
        final_features = pd.read_parquet(features_path) if features_path.exists() else features_df
        final_targets = pd.read_parquet(targets_path) if targets_path.exists() else targets_df
        
        # Create config.json ONLY for test mode
        config_path = None
        if self._is_test_mode(config):
            config_path = self._create_test_config(batch_dir, config, timestamp, eff_batch_name)
        else:
            self.logger.info("📊 Full mode: NOT creating config.json (Colab will use all data)")
            # ✅ CRITICAL: Remove old config.json from previous test runs
            old_config = batch_dir / "config.json"
            if old_config.exists():
                self.logger.warning(f"🗑️ Removing old config.json from previous test run: {old_config}")
                old_config.unlink()
                self.logger.info("✅ Old config.json removed - full mode will process ALL tickers")
        
        # Create batch metadata
        batch_metadata = {
            'batch_name': eff_batch_name,
            'timestamp': timestamp,
            'tickers': config.tickers,
            'timeframes': config.timeframes,
            'features_shape': final_features.shape,
            'targets_shape': final_targets.shape,
            'accumulated': config.accumulate and features_path.exists(),
            'test_mode': self._is_test_mode(config),
            'files': {
                'features': str(features_path),
                'targets': str(targets_path),
                'config': str(config_path) if config_path else None
            }
        }
        
        # Save metadata
        metadata_path = batch_dir / BATCH_METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        # Check feature selection using config parameters
        fs_check = self._check_feature_selection(
            batch_dir, 
            features_df, 
            config.check_feature_selection, 
            config.force_feature_selection
        )
        
        result = {
            'batch_dir': str(batch_dir), 
            'batch_name': eff_batch_name, 
            'metadata_path': str(metadata_path), 
            'files': batch_metadata['files'], 
            'feature_selection_check': fs_check,
            'test_mode': self._is_test_mode(config)
        }
        
        if config_path:
            result['config_path'] = str(config_path)
        
        return result
    
    def load_colab_results(self, batch_name: str) -> Dict[str, Any]:
        """Loads training results from Colab."""
        batch_name = batch_name.replace('target_target_', 'target_')
        batch_dir = self._find_batch_directory(batch_name)
        
        if not batch_dir.exists():
            self.logger.error(f"Batch directory not found: {batch_dir}")
            return {'error': f'Batch directory not found: {batch_dir}'}
        
        results = {}
        
        # Mapping of filenames to result keys
        files_to_load = {
            SELECTED_FEATURES_PATTERN: 'selected_features',
            'trained_models_metadata.json': 'models_metadata',
            'colab_results.json': 'models_metadata',  # Fallback/Alternative name from Colab
            'evaluation_results.json': 'evaluation_results'
        }
        
        for pattern, key in files_to_load.items():
            if "*" in pattern:
                found_files = list(batch_dir.glob(pattern))
                for file_path in found_files:
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if key not in results:
                                results[key] = data
                            elif isinstance(results[key], dict) and isinstance(data, dict):
                                results[key].update(data)
            else:
                file_path = batch_dir / pattern
                
            if file_path and file_path.exists() and "*" not in pattern:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if key == 'models_metadata' and 'models_metadata' in data:
                        # Extract nested models_metadata if present
                        results[key] = data['models_metadata']
                    else:
                        results[key] = data
        
        return results
    
    def _find_batch_directory(self, batch_name: str) -> Path:
        """Find the batch directory by name."""
        direct_path = self.output_dir / batch_name
        if direct_path.exists():
            return direct_path
        
        for batch_dir in self.output_dir.glob(f"*{batch_name}*"):
            if batch_dir.is_dir():
                return batch_dir
        
        return direct_path
    
    def _check_feature_selection(self, batch_dir: Path, features_df: pd.DataFrame, 
                                 check_selection: bool, force_selection: bool) -> Dict[str, Any]:
        """Check if feature selection is needed."""
        if not check_selection:
            return {'needed': False, 'reason': 'Feature selection check disabled'}
        
        selected_features_files = list(batch_dir.glob(SELECTED_FEATURES_PATTERN))
        
        if force_selection or not selected_features_files:
            reason = 'Forced selection' if force_selection else 'No existing selection'
            return {'needed': True, 'reason': reason}
        
        if len(features_df) < 1000:
            return {'needed': False, 'reason': 'Dataset too small for feature selection'}
        
        return {'needed': False, 'reason': 'Existing feature selection found'}
    
    def _is_test_mode(self, config: BatchPreparationConfig) -> bool:
        """Check if this is test mode based on config parameters."""
        return bool(config.test_ticker or config.test_target or config.test_model)
    
    def _create_test_config(self, batch_dir: Path, config: BatchPreparationConfig, 
                           timestamp: str, batch_name: str) -> Path:
        """Create config.json for test mode."""
        config_data = {
            'test_mode': {
                'enabled': True,
                'test_ticker': config.test_ticker,
                'test_target': config.test_target,
                'test_model': config.test_model,
                'epochs': config.epochs or 5,
                'max_iterations': config.max_iterations or 5
            },
            'batch_name': batch_name,
            'created_at': timestamp
        }
        
        config_path = batch_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"🧪 Test mode config created: {config.test_ticker} | {config.test_target} | epochs={config.epochs}")
        return config_path