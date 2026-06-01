"""
Scaler Service for Stage 5 Prediction.

Handles target scaler loading and validation.
Extracted from stage_5_prediction.py to reduce coupling.
"""
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from src.core.logging.logger import ProjectLogger


class ScalerService:
    """
    Service for loading and validating target scalers.
    
    Responsibilities:
    - Load target scaler from disk
    - Validate scaler structure
    - Handle scaler path resolution
    """

    ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
    DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger('ScalerService')

    def load_target_scaler(self, meta: Dict[str, Any]) -> Optional[Any]:
        """
        Load target scaler for denormalization.
        
        Args:
            meta: Model metadata containing model_path, ticker, target
            
        Returns:
            Scaler object if valid, None otherwise
        """
        try:
            ticker = meta.get('ticker', '')
            target_col = meta.get('target', '')
            model_path_str = meta.get('model_path', '')
            
            if not model_path_str:
                return None
            
            # Normalize path separators
            model_path_str = model_path_str.replace('/', '\\')
            parts = model_path_str.split('\\')
            
            if 'models' not in parts:
                return None
            
            models_idx = parts.index('models')
            if models_idx <= 0:
                return None
            
            batch_name = parts[models_idx - 1]
            
            # Build scaler path
            base_dir = Path(self.config_manager.get(
                self.ACCUMULATION_OUTPUT_DIR_CONFIG,
                self.DEFAULT_ACCUMULATION_DIR
            ))
            scaler_path = base_dir / batch_name / f'scaler_{ticker}_{target_col}.pkl'
            
            if not scaler_path.exists():
                if self.logger.isEnabledFor(20):  # DEBUG level
                    self.logger.debug(f'⚠️ No target scaler found at {scaler_path}')
                return None
            
            # Load and validate scaler
            target_scaler = joblib.load(scaler_path)
            
            if hasattr(target_scaler, 'scale_'):
                if target_scaler.scale_.shape[0] == 1:
                    self.logger.info(f'✅ Loaded target scaler from {scaler_path}')
                    return target_scaler
                else:
                    self.logger.error(
                        f'❌ INVALID scaler! Has {target_scaler.scale_.shape[0]} features instead of 1'
                    )
            else:
                self.logger.warning('⚠️ Scaler has no scale_ attribute')
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error loading target scaler: {e}", exc_info=True)
            return None
