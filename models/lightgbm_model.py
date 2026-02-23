# models/lightgbm_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LightGBMModel:
    """LightGBM модель з fallback на RandomForest"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1,
                 num_leaves: int = 31, classification: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.classification = classification
        self.is_trained = False
        self.model = None
        
        # Fallback модель
        self.fallback_model = None
        
    def _create_fallback_model(self):
        """Створити fallback модель на основі RandomForest"""
        try:
            self.fallback_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            ) if self.classification else RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            
            logger.info("OK Created RandomForest fallback model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
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
            
            # Спроба використати LightGBM
            try:
                self._fit_lightgbm(X, y)
            except ImportError:
                logger.warning("LightGBM not available, using fallback model")
                self._fit_fallback(X, y)
                
            self.is_trained = True
            logger.info(f"OK LightGBM model trained (classification: {self.classification})")
            
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            # Fallback до простої моделі
            try:
                self._fit_fallback(X, y)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise
    
    def _fit_lightgbm(self, X, y):
        """Тренування LightGBM"""
        import lightgbm as lgb
        
        # Створення моделі
        if self.classification:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=42,
                verbose=-1
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=42,
                verbose=-1
            )
        
        # Тренування
        self.model.fit(X, y)
        logger.info("OK LightGBM model trained successfully")
    
    def _fit_fallback(self, X, y):
        """Тренування fallback моделі"""
        if self.fallback_model is None:
            if not self._create_fallback_model():
                raise RuntimeError("Cannot create fallback model")
        
        # Використовуємо останні значення для тренування
        if len(X) > 10:
            X_train = X[-min(len(X), 100):]  # Останні 100 точок
            y_train = y[-min(len(y), 100):]
        else:
            X_train = X
            y_train = y
        
        self.fallback_model.fit(X_train, y_train)
        logger.info("OK Fallback model trained")
    
    def predict(self, X):
        """Прогнозування"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба LightGBM
            if self.model is not None:
                return self._predict_lightgbm(X)
            else:
                return self._predict_fallback(X)
                
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
    
    def _predict_lightgbm(self, X):
        """Прогнозування LightGBM"""
        return self.model.predict(X)
    
    def _predict_fallback(self, X):
        """Прогнозування fallback"""
        if self.fallback_model is None:
            raise RuntimeError("No fallback model available")
        
        # Використовуємо останні значення для прогнозу
        if len(X.shape) == 2:
            return self.fallback_model.predict(X[-1:].reshape(1, -1))
        else:
            return self.fallback_model.predict(X.reshape(1, -1))
    
    def predict_proba(self, X):
        """Прогнозування ймовірностей"""
        if not self.classification:
            raise ValueError("predict_proba only available for classification")
        
        predictions = self.predict(X)
        
        # Конвертація в ймовірності
        if len(predictions.shape) == 1:
            probas = np.zeros((len(predictions), 2))
            probas[:, 1] = predictions
            probas[:, 0] = 1 - predictions
            return probas
        else:
            return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'classification': self.classification,
            'is_trained': self.is_trained,
            'has_lightgbm_model': self.model is not None,
            'has_fallback_model': self.fallback_model is not None
        }
