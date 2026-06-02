import logging
# src/training/light_model_trainer.py

import pandas as pd
from typing import Dict, Any, Optional
import joblib
import uuid
from pathlib import Path

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.factories.model_factory import ModelFactory
from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path

logger = ProjectLogger.get_logger("LightModelTrainer")

class LightModelTrainer:
    """
    Lightweight model trainer using ModelFactory.
    
    Trains and manages in-memory model instances using the centralized
    ModelFactory for consistent model creation and graceful error handling.
    """
    
    def __init__(self):
        self.config_manager = get_current_config()
        self.factory = ModelFactory()
        self.models_in_memory: Dict[str, Any] = {}
        logger.info("LightModelTrainer initialized with ModelFactory")

    def train_light_model(
        self,
        features_df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a light model using ModelFactory.
        
        Args:
            features_df: DataFrame with features and target column
            config: Training configuration dictionary containing:
                - model_type: Type of model (e.g., 'linear', 'rf', 'xgb', 'lgbm')
                - ticker: Ticker symbol
                - timeframe: Timeframe identifier
                - target_col: Name of target column
                - task_type: 'regression' or 'classification' (default: 'regression')
                - params: Optional hyperparameters (default: None)
        
        Returns:
            Dictionary with status, model_key, and metrics
        """
        try:
            # Extract configuration parameters
            model_type = config['model_type']
            ticker = config['ticker']
            timeframe = config['timeframe']
            target_col = config['target_col']
            task_type = config.get('task_type', 'regression')
            params = config.get('params')
            
            # Prepare data
            # Drop all target columns and metadata to prevent leakage
            metadata_cols = ['ticker', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']
            drop_cols = [c for c in features_df.columns if c.startswith('target_') or c in metadata_cols]
            
            # Ensure target_col is removed from drop_cols so we can extract it for y
            # but it MUST be in drop_cols when creating X
            X = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
            y = features_df[target_col]
            
            # Check if target_col was actually in features_df and not dropped
            # (target_col is usually one of the columns starting with 'target_')
            
            # Get model from factory (consistent with batch/progressive trainers)
            is_classification = task_type == 'classification'
            model_params = params or self.config_manager.get_config(f"models.{model_type}", {})
            
            model = self.factory.create_model(
                model_type,
                config=model_params,
                task_type=task_type,
                is_classification=is_classification
            )
            
            if model is None:
                return {
                    "status": "failed",
                    "reason": f"ModelFactory could not create {model_type}",
                    "model_key": None,
                    "metrics": {}
                }
            
            # Train model
            model.train(X, y)
            
            # Generate unique key and store
            model_key = f"{model_type}-{ticker}-{timeframe}-{uuid.uuid4()}"
            self.models_in_memory[model_key] = model
            
            logger.info(f"✅ Trained {model_type} model for {ticker} (key: {model_key})")
            
            return {
                "status": "success",
                "model_key": model_key,
                "metrics": {"model_type": model_type, "ticker": ticker}
            }
        
        except Exception as e:
            logger.error(f"❌ Error training {model_type} for {ticker}: {e}")
            return {
                "status": "failed",
                "reason": str(e),
                "model_key": None,
                "metrics": {}
            }

    def predict(self, model_key: str, features_df: pd.DataFrame) -> pd.Series:
        """
        Make predictions using a stored model.
        
        Args:
            model_key: Key to retrieve model from memory
            features_df: Features for prediction
        
        Returns:
            Predictions
        """
        model = self.models_in_memory.get(model_key)
        if model is None:
            raise ValueError(f"Model '{model_key}' not found in memory.")
        
        return model.predict(features_df)

    def save_model_to_disk(self, model_key: str, path: str) -> bool:
        """
        Save a model from memory to disk.
        
        Args:
            model_key: Key to retrieve model from memory
            path: File path to save to
        
        Returns:
            True if successful, False otherwise
        """
        model = self.models_in_memory.get(model_key)
        if model is None:
            logger.warning(f"Model '{model_key}' not found in memory")
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            logger.info(f"✅ Saved model {model_key} to {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving model {model_key}: {e}")
            return False

    def load_model_from_disk(self, model_key: str, path: str) -> bool:
        """
        Load a model from disk into memory.
        
        Args:
            model_key: Key to store model under in memory
            path: File path to load from
        
        Returns:
            True if successful, False otherwise
        """
        try:
            trusted_path = resolve_trusted_artifact_path(
                path,
                allowed_suffixes={'.joblib', '.pkl', '.pickle'},
                must_exist=True,
            )
            self.models_in_memory[model_key] = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            logger.info(f"✅ Loaded model from {path} (key: {model_key})")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading model from {path}: {e}")
            return False

    def remove_model(self, model_key: str) -> bool:
        """
        Remove a model from memory.
        
        Args:
            model_key: Key of model to remove
        
        Returns:
            True if model was removed, False if not found
        """
        if model_key in self.models_in_memory:
            del self.models_in_memory[model_key]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Removed model {model_key} from memory")
            return True
        logger.warning(f"Model '{model_key}' not found in memory")
        return False

    def clear_memory(self):
        """Clear all models from memory."""
        self.models_in_memory.clear()
        logger.info("Cleared all models from memory")
