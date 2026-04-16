# src/training/light_model_trainer.py

import pandas as pd
from typing import Dict, Any, Optional
import joblib
import uuid
from pathlib import Path

from src.config.unified_config_manager import UnifiedConfigManager
from src.factories.model_factory import ModelFactory
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("LightModelTrainer")

class LightModelTrainer:
    """
    Lightweight model trainer using ModelFactory.
    
    Trains and manages in-memory model instances using the centralized
    ModelFactory for consistent model creation and graceful error handling.
    """
    
    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.factory = ModelFactory()
        self.models_in_memory: Dict[str, Any] = {}
        logger.info("LightModelTrainer initialized with ModelFactory")

    def train_light_model(
        self,
        features_df: pd.DataFrame,
        model_type: str,
        ticker: str,
        timeframe: str,
        target_col: str,
        task_type: str = "regression",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a light model using ModelFactory.
        
        Args:
            features_df: DataFrame with features and target column
            model_type: Type of model (e.g., 'linear', 'rf', 'xgb', 'lgbm')
            ticker: Ticker symbol
            timeframe: Timeframe identifier
            target_col: Name of target column
            task_type: 'regression' or 'classification'
            params: Optional hyperparameters
        
        Returns:
            Dictionary with status, model_key, and metrics
        """
        try:
            # Prepare data
            X = features_df.drop(columns=[target_col])
            y = features_df[target_col]
            
            # Get model from factory (consistent with batch/progressive trainers)
            is_classification = task_type == 'classification'
            model_params = params or self.config_manager.get_config(f"models.{model_type}", {})
            
            model = self.factory.create_model(
                model_type=model_type,
                is_classification=is_classification,
                params=model_params
            )
            
            if model is None:
                return {
                    "status": "failed",
                    "reason": f"ModelFactory could not create {model_type}",
                    "model_key": None,
                    "metrics": {}
                }
            
            # Train model
            model.fit(X, y)
            
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
            self.models_in_memory[model_key] = joblib.load(path)
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
            logger.debug(f"Removed model {model_key} from memory")
            return True
        logger.warning(f"Model '{model_key}' not found in memory")
        return False

    def clear_memory(self):
        """Clear all models from memory."""
        self.models_in_memory.clear()
        logger.info("Cleared all models from memory")

    def train_light_model(self, features_df: pd.DataFrame, model_type: str, ticker: str, timeframe: str, target_col: str, task_type: str) -> Dict[str, Any]:
        """Trains a model and stores it in memory."""
        
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        model_config = self.models_config.get(model_type, {})
        default_params = model_config.get('default_params', {})
        
        model = self._get_model_instance(model_type, task_type, default_params)
        
        model.fit(X, y)
        
        model_key = f"{model_type}-{ticker}-{timeframe}-{uuid.uuid4()}"
        self._models_in_memory[model_key] = model
        
        # In a real scenario, you'd calculate metrics here
        metrics = {"placeholder_metric": 1.0}
        
        return {"status": "success", "model_key": model_key, "metrics": metrics}

    def predict(self, model_key: str, features_df: pd.DataFrame) -> pd.Series:
        """Makes predictions using a model stored in memory."""
        model = self._models_in_memory.get(model_key)
        if model is None:
            raise ValueError(f"Model with key '{model_key}' not found in memory.")
        
        return model.predict(features_df)

    def save_model_to_disk(self, model_key: str, path: str) -> bool:
        """Saves a model from memory to a file."""
        model = self._models_in_memory.get(model_key)
        if model is None:
            return False
        try:
            joblib.dump(model, path)
            return True
        except Exception:
            return False

    def load_model_from_disk(self, model_key: str, path: str) -> bool:
        """Loads a model from a file into memory."""
        try:
            self._models_in_memory[model_key] = joblib.load(path)
            return True
        except Exception:
            return False
