# core/models/model_interface.py - Єдиний andнтерфейс for allх моwhereлей

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Баwithовий andнтерфейс for allх моwhereлей"""
    
    def __init__(self, model_type: str, task_type: str = "regression"):
        self.model_type = model_type
        self.task_type = task_type
        self.is_trained = False
        self.feature_cols = None
        self.metrics = {}
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Тренування моwhereлand"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Прогноwithування"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оцandнка якостand моwhereлand"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Збереження моwhereлand"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Заванandження моwhereлand"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Отримати andнформацandю про model"""
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "is_trained": self.is_trained,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics
        }

class LightModelInterface(BaseModel):
    """Інтерфейс for легких моwhereлей"""
    
    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        from .light_models import LightModelTrainer
        self.trainer = LightModelTrainer()
    
    def train(self, X: np.ndarray, y: np.ndarray, ticker: str = "DEFAULT", timeframe: str = "1d") -> Dict[str, Any]:
        """Тренування легкої моwhereлand"""
        # Створення DataFrame with data
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        
        # Тренування череwith LightModelTrainer
        result = self.trainer.train_light_model(
            features_df=df,
            model_type=self.model_type,
            ticker=ticker,
            timeframe=timeframe,
            target_col="target",
            task_type=self.task_type
        )
        
        if result["status"] == "success":
            self.is_trained = True
            self.feature_cols = feature_cols
            self.metrics = result["metrics"]
            self.model_key = result["model_key"]
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Прогноwithування"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Створення DataFrame
        df = pd.DataFrame(X, columns=self.feature_cols)
        
        # Прогноwithування череwith треnotр
        return self.trainer.predict(self.model_key, df)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оцandнка якостand моwhereлand"""
        predictions = self.predict(X)
        
        if self.task_type == "regression":
            return {
                "mse": mean_squared_error(y, predictions),
                "r2": r2_score(y, predictions),
                "mae": mean_absolute_error(y, predictions)
            }
        else:  # classification
            return {
                "accuracy": accuracy_score(y, predictions),
                "f1": f1_score(y, predictions, average='weighted')
            }
    
    def save_model(self, path: str) -> bool:
        """Збереження моwhereлand"""
        try:
            import joblib
            model_data = {
                "model_type": self.model_type,
                "task_type": self.task_type,
                "is_trained": self.is_trained,
                "feature_cols": self.feature_cols,
                "metrics": self.metrics,
                "model_key": self.model_key if hasattr(self, 'model_key') else None
            }
            joblib.dump(model_data, path)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Заванandження моwhereлand"""
        try:
            import joblib
            model_data = joblib.load(path)
            
            self.model_type = model_data["model_type"]
            self.task_type = model_data["task_type"]
            self.is_trained = model_data["is_trained"]
            self.feature_cols = model_data["feature_cols"]
            self.metrics = model_data["metrics"]
            
            if model_data.get("model_key"):
                self.model_key = model_data["model_key"]
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class HeavyModelInterface(BaseModel):
    """Інтерфейс for важких моwhereлей (Colab)"""
    
    def __init__(self, model_type: str, task_type: str = "regression"):
        super().__init__(model_type, task_type)
        self.colab_manager = None
    
    def train(self, X: np.ndarray, y: np.ndarray, ticker: str = "DEFAULT", timeframe: str = "1d") -> Dict[str, Any]:
        """Тренування важкої моwhereлand в Colab"""
        try:
            from utils.colab_manager import ColabManager
            self.colab_manager = ColabManager()
            
            # Пandдготовка data for Colab
            data = {
                "X": X.tolist(),
                "y": y.tolist(),
                "model_type": self.model_type,
                "task_type": self.task_type,
                "ticker": ticker,
                "timeframe": timeframe
            }
            
            # Вandдправка в Colab for тренування
            result = self.colab_manager.train_heavy_model(data)
            
            if result.get("success"):
                self.is_trained = True
                self.metrics = result.get("metrics", {})
                self.model_id = result.get("model_id")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training heavy model: {e}")
            return {"success": False, "error": str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Прогноwithування череwith Colab"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if not self.colab_manager:
            raise ValueError("Colab manager not initialized")
        
        try:
            data = {
                "X": X.tolist(),
                "model_id": self.model_id
            }
            
            result = self.colab_manager.predict_heavy_model(data)
            
            if result.get("success"):
                return np.array(result["predictions"])
            else:
                raise ValueError(f"Prediction failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error predicting with heavy model: {e}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оцandнка якостand моwhereлand"""
        predictions = self.predict(X)
        
        if self.task_type == "regression":
            return {
                "mse": mean_squared_error(y, predictions),
                "r2": r2_score(y, predictions),
                "mae": mean_absolute_error(y, predictions)
            }
        else:  # classification
            return {
                "accuracy": accuracy_score(y, predictions),
                "f1": f1_score(y, predictions, average='weighted')
            }
    
    def save_model(self, path: str) -> bool:
        """Збереження моwhereлand (в Colab)"""
        try:
            if not self.colab_manager or not hasattr(self, 'model_id'):
                return False
            
            result = self.colab_manager.save_heavy_model(self.model_id, path)
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error saving heavy model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Заванandження моwhereлand (with Colab)"""
        try:
            if not self.colab_manager:
                from utils.colab_manager import ColabManager
                self.colab_manager = ColabManager()
            
            result = self.colab_manager.load_heavy_model(path)
            
            if result.get("success"):
                self.is_trained = True
                self.model_id = result.get("model_id")
                self.metrics = result.get("metrics", {})
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading heavy model: {e}")
            return False

class ModelFactory:
    """Фабрика for створення моwhereлей"""
    
    @staticmethod
    def create_model(model_type: str, model_category: str = "light", 
                    task_type: str = "regression") -> BaseModel:
        """Create model вandдповandдного типу"""
        
        if model_category == "light":
            return LightModelInterface(model_type, task_type)
        elif model_category == "heavy":
            return HeavyModelInterface(model_type, task_type)
        else:
            raise ValueError(f"Unknown model category: {model_category}")
    
    @staticmethod
    def get_available_models(model_category: str = None) -> Dict[str, List[str]]:
        """Отримати список доступних моwhereлей"""
        
        light_models = [
            "random_forest", "linear", "svr", "knn", 
            "xgboost", "lightgbm", "catboost"
        ]
        
        heavy_models = [
            "lstm", "gru", "transformer", "cnn", 
            "autoencoder", "tabnet", "deep_ensemble"
        ]
        
        if model_category == "light":
            return {"light": light_models}
        elif model_category == "heavy":
            return {"heavy": heavy_models}
        else:
            return {"light": light_models, "heavy": heavy_models}
