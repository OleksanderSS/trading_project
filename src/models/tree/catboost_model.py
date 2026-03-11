# src/models/tree/catboost_model.py

import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union
from catboost import CatBoostRegressor, CatBoostClassifier
from src.models.interfaces import BaseModel
from src.core.logging.logger import ProjectLogger

class CatBoostModel(BaseModel):
    """
    CatBoost model implementation for trading tasks, following the unified BaseModel interface.
    """
    
    def __init__(self, task_type: str = "classification", iterations: int = 200, depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        super().__init__(model_type="catboost", task_type=task_type)
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.logger = ProjectLogger.get_logger("CatBoostModel")

    @property
    def name(self) -> str:
        """Unique identifier for the model."""
        return "catboost"

    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Cleans column names and handles duplicates for CatBoost compatibility."""
        if isinstance(X, pd.DataFrame):
            X = X.loc[:, ~X.columns.duplicated()].copy()
            # Replace problematic characters in column names
            X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                       for col in X.columns]
        return X

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Trains the CatBoost model."""
        try:
            X_clean = self._prepare_data(X)
            
            if isinstance(X_clean, pd.DataFrame):
                self.feature_cols = X_clean.columns.tolist()

            if self.task_type == "classification":
                self.model = CatBoostClassifier(
                    iterations=self.iterations,
                    depth=self.depth,
                    learning_rate=self.learning_rate,
                    random_seed=self.random_state,
                    verbose=False,
                    **kwargs
                )
            else:
                self.model = CatBoostRegressor(
                    iterations=self.iterations,
                    depth=self.depth,
                    learning_rate=self.learning_rate,
                    random_seed=self.random_state,
                    verbose=False,
                    **kwargs
                )
            
            self.model.fit(X_clean, y)
            self.is_trained = True
            
            self.logger.info(f"CatBoost trained successfully ({self.task_type}, iterations={self.iterations}, depth={self.depth})")
            return self.get_model_info()
            
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            raise

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Makes predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            X_clean = self._prepare_data(X)
            return self.model.predict(X_clean)
        except Exception as e:
            self.logger.error(f"CatBoost prediction failed: {e}")
            raise

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities (classification only)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        try:
            X_clean = self._prepare_data(X)
            return self.model.predict_proba(X_clean)
        except Exception as e:
            self.logger.error(f"CatBoost probability prediction failed: {e}")
            raise

    def save_model(self, path: str) -> bool:
        """Saves the model using CatBoost's native format or joblib."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Using native catboost format for better compatibility
            self.model.save_model(path)
            self.logger.info(f"CatBoost model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save CatBoost model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Loads the model from a file."""
        try:
            if self.task_type == "classification":
                self.model = CatBoostClassifier()
            else:
                self.model = CatBoostRegressor()
                
            self.model.load_model(path)
            self.is_trained = True
            self.logger.info(f"CatBoost model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load CatBoost model: {e}")
            return False

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Returns feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            return self.model.get_feature_importance()
        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return None