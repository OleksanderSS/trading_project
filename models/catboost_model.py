# models/catboost_model.py

import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from catboost import CatBoostRegressor, CatBoostClassifier
import logging

logger = logging.getLogger(__name__)

class CatBoostModel:
    """CatBoost модель для трейдингу"""
    
    def __init__(self, task="classification", iterations=200, depth=6, learning_rate=0.1, random_state=42):
        self.task = task
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
    def fit(self, X, y, **kwargs):
        """Тренування моделі"""
        try:
            # Унікальні назви колонок
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                # Заміна проблемних символів в назвах колонок
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            # Створення моделі
            if self.task == "classification":
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
            
            # Тренування
            self.model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"OK CatBoost trained ({self.task}, iterations={self.iterations}, depth={self.depth})")
            
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            raise
    
    def predict(self, X):
        """Прогнозування"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Підготовка data
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            return self.model.predict(X)
            
        except Exception as e:
            logger.error(f"CatBoost prediction failed: {e}")
            raise
    
    def predict_proba(self, X):
        """Прогнозування ймовірностей"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        try:
            # Підготовка data
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            return self.model.predict_proba(X)
            
        except Exception as e:
            logger.error(f"CatBoost probability prediction failed: {e}")
            raise
    
    def save(self, path: str):
        """Збереження моделі"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            joblib.dump(self.model, path)
            logger.info(f"CatBoost model saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save CatBoost model: {e}")
            raise
    
    def load(self, path: str):
        """Завантаження моделі"""
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"CatBoost model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
            raise
    
    def get_feature_importance(self):
        """Важливість фіч"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            return self.model.get_feature_importance()
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
        return {
            'task': self.task,
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'is_trained': self.is_trained
        }

# Зберігаємо стару функцію для сумісності
def train_catboost_model(X_train, y_train, task="regression", iterations=200, depth=6,
                         learning_rate=0.1, random_state=42, save_path=None, **kwargs):
    """Стара функція для сумісності"""
    model = CatBoostModel(task=task, iterations=iterations, depth=depth, 
                          learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train, **kwargs)
    
    if save_path:
        model.save(save_path)
    
    return model