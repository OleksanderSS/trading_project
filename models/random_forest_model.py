# models/random_forest_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RandomForestModel:
    """RandomForest модель"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, classification: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classification = classification
        self.is_trained = False
        self.model = None
        
    def _create_model(self):
        """Створити модель"""
        try:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            ) if self.classification else RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("OK Created RandomForest model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create RandomForest model: {e}")
            return False
    
    def fit(self, X, y):
        """Тренування моделі"""
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Створення моделі
            if self.model is None:
                if not self._create_model():
                    raise RuntimeError("Cannot create model")
            
            # Тренування
            self.model.fit(X, y)
            self.is_trained = True
            
            logger.info(f"OK RandomForest model trained (classification: {self.classification})")
            
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")
            raise
    
    def predict(self, X):
        """Прогнозування"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            return self.model.predict(X)
                
        except Exception as e:
            logger.error(f"RandomForest prediction failed: {e}")
            raise
    
    def predict_proba(self, X):
        """Прогнозування ймовірностей"""
        if not self.classification:
            raise ValueError("predict_proba only available for classification")
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            return self.model.predict_proba(X)
                
        except Exception as e:
            logger.error(f"RandomForest probability prediction failed: {e}")
            raise
    
    def get_feature_importance(self):
        """Отримати важливість фіч"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            else:
                logger.warning("Feature importance not available for this model")
                return None
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'classification': self.classification,
            'is_trained': self.is_trained,
            'has_model': self.model is not None
        }
