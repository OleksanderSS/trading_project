# src/models/light_model_trainer.py

import pandas as pd
from typing import Dict, Any
import joblib
import uuid

# Model imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from src.config.unified_config_manager import get_current_config

class LightModelTrainer:
    """Manages the training, prediction, and storage of lightweight models."""
    
    _models_in_memory: Dict[str, Any] = {}

    def __init__(self):
        config = get_current_config()
        self.models_config = config.get('models', {})

    def _get_model_instance(self, model_type: str, task_type: str, params: Dict[str, Any]) -> Any:
        """Creates a model instance based on the model type and task type."""
        model_map = {
            "regression": {
                "linear": LinearRegression,
                "random_forest": RandomForestRegressor,
                "svm": SVR,
                "knn": KNeighborsRegressor,
                "xgboost": XGBRegressor,
                "lightgbm": LGBMRegressor,
                "catboost": CatBoostRegressor
            },
            "classification": {
                "linear": LogisticRegression,
                "random_forest": RandomForestClassifier,
                "svm": SVC,
                "knn": KNeighborsClassifier,
                "xgboost": XGBClassifier,
                "lightgbm": LGBMClassifier,
                "catboost": CatBoostClassifier
            }
        }
        
        try:
            model_class = model_map[task_type][model_type]
            return model_class(**params)
        except KeyError:
            raise ValueError(f"Unsupported model type '{model_type}' for task '{task_type}'.")

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
