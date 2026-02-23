# models/svm_model.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from sklearn.svm import SVR, SVC
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SVMModel:
    """SVM модель з fallback на RandomForest"""
    
    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", classification: bool = True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.classification = classification
        self.is_trained = False
        self.model = None
        self.imputer = SimpleImputer(strategy="mean")
        
        # Fallback модель
        self.fallback_model = None
        
    def _create_fallback_model(self):
        """Створити fallback модель на основі RandomForest"""
        try:
            self.fallback_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ) if self.classification else RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
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
            
            # Спроба використати SVM
            try:
                self._fit_svm(X, y)
            except Exception as e:
                logger.warning(f"SVM failed: {e}, using fallback model")
                self._fit_fallback(X, y)
                
            self.is_trained = True
            logger.info(f"OK SVM model trained (classification: {self.classification})")
            
        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            # Fallback до простої моделі
            try:
                self._fit_fallback(X, y)
                self.is_trained = True
                logger.info("OK Used fallback model")
            except Exception as e2:
                logger.error(f"Fallback training also failed: {e2}")
                raise
    
    def _fit_svm(self, X, y):
        """Тренування SVM"""
        # Створення моделі
        if self.classification:
            self.model = SVC(
                kernel=self.kernel, 
                C=self.C, 
                gamma=self.gamma, 
                class_weight="balanced", 
                random_state=42
            )
        else:
            self.model = SVR(
                kernel=self.kernel, 
                C=self.C, 
                gamma=self.gamma
            )
        
        # Підготовка data
        X_clean = self.imputer.fit_transform(X)
        
        # Тренування
        self.model.fit(X_clean, y)
        logger.info("OK SVM model trained successfully")
    
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
            
            # Спроба SVM
            if self.model is not None:
                return self._predict_svm(X)
            else:
                return self._predict_fallback(X)
                
        except Exception as e:
            logger.error(f"SVM prediction failed: {e}")
            # Fallback
            try:
                return self._predict_fallback(X)
            except Exception as e2:
                logger.error(f"Fallback prediction also failed: {e2}")
                raise
    
    def _predict_svm(self, X):
        """Прогнозування SVM"""
        X_clean = self.imputer.transform(X)
        return self.model.predict(X_clean)
    
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
        
        try:
            # Конвертація в numpy
            if hasattr(X, 'values'):
                X = X.values
                
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Спроба SVM
            if self.model is not None and hasattr(self.model, 'predict_proba'):
                X_clean = self.imputer.transform(X)
                return self.model.predict_proba(X_clean)
            else:
                # Fallback probability
                predictions = self._predict_fallback(X)
                probas = np.zeros((len(predictions), 2))
                probas[:, 1] = predictions
                probas[:, 0] = 1 - predictions
                return probas
                
        except Exception as e:
            logger.error(f"SVM probability prediction failed: {e}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Параметри моделі"""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'classification': self.classification,
            'is_trained': self.is_trained,
            'has_svm_model': self.model is not None,
            'has_fallback_model': self.fallback_model is not None
        }

# Зберігаємо стару функцію для сумісності
def train_svm_model(X_train, y_train, task="regression", kernel="rbf", C=1.0, gamma="scale", save_path=None, **kwargs):
    """Стара функція для сумісності"""
    try:
        model = SVMModel(kernel=kernel, C=C, gamma=gamma, classification=(task == "classification"))
        model.fit(X_train, y_train)
        logger.info("OK SVM model trained (compatibility function)")
        return model
        
    except Exception as e:
        logger.error(f"Error training SVM model: {e}")
        return None