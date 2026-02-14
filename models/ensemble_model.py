# models/ensemble_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from typing import List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ансамблева модель для комбінування кращих моделей"""
    
    def __init__(self, models: Optional[List[Tuple[str, Any]]] = None, task_type: str = "classification", voting: str = "soft"):
        self.models = models or self._get_default_models(task_type)
        self.task_type = task_type
        self.voting = voting
        self.ensemble = None
        self.is_trained = False
        
    def _get_default_models(self, task_type: str) -> List[Tuple[str, Any]]:
        """Отримати моделі за замовчуванням"""
        if task_type == "classification":
            return [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
        else:
            return [
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('svm', SVR())
            ]
    
    def create_ensemble(self):
        """Створити ансамбль з найкращих моделей"""
        try:
            if self.task_type == "classification":
                self.ensemble = VotingClassifier(
                    estimators=self.models,
                    voting=self.voting
                )
            else:
                self.ensemble = VotingRegressor(estimators=self.models)
                
            logger.info(f"OK Created {self.task_type} ensemble with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}")
            raise
    
    def fit(self, X, y):
        """Тренування ансамблю"""
        try:
            # Підготовка data
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                # Очищення назв колонок
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            if self.ensemble is None:
                self.create_ensemble()
                
            self.ensemble.fit(X, y)
            self.is_trained = True
            
            # Оцінка якості
            try:
                scores = cross_val_score(self.ensemble, X, y, cv=3, scoring='accuracy' if self.task_type == 'classification' else 'r2')
                logger.info(f"OK CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                logger.warning(f"CV scoring failed: {e}")
                
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise
    
    def predict(self, X):
        """Прогноз ансамблю"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Підготовка data
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            return self.ensemble.predict(X)
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    def predict_proba(self, X):
        """Прогноз ймовірностей (тільки для класифікації)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba available only for classification")
        
        try:
            # Підготовка data
            if hasattr(X, "columns"):
                X = X.loc[:, ~X.columns.duplicated()]
                X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') 
                           for col in X.columns]
            
            return self.ensemble.predict_proba(X)
            
        except Exception as e:
            logger.error(f"Ensemble probability prediction failed: {e}")
            raise
    
    def get_feature_importance(self):
        """Отримати важливість фіч (якщо доступно)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Спроба отримати важливість з RandomForest
            for name, model in self.ensemble.estimators_:
                if hasattr(model, 'feature_importances_'):
                    return model.feature_importances_
            
            logger.warning("Feature importance not available for this ensemble")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def add_model(self, name: str, model: Any):
        """Додати модель до ансамблю"""
        if self.is_trained:
            raise ValueError("Cannot add models to trained ensemble")
        
        self.models.append((name, model))
        logger.info(f"Added model {name} to ensemble")
    
    def remove_model(self, name: str):
        """Видалити модель з ансамблю"""
        if self.is_trained:
            raise ValueError("Cannot remove models from trained ensemble")
        
        self.models = [(n, m) for n, m in self.models if n != name]
        logger.info(f"Removed model {name} from ensemble")
    
    def get_model_count(self) -> int:
        """Отримати кількість моделей"""
        return len(self.models)
    
    def get_model_names(self) -> List[str]:
        """Отримати назви моделей"""
        return [name for name, _ in self.models]
    
    def is_classification(self) -> bool:
        """Перевірити чи це класифікація"""
        return self.task_type == "classification"
    
    def get_params(self) -> dict:
        """Отримати параметри ансамблю"""
        return {
            'models': self.get_model_names(),
            'task_type': self.task_type,
            'voting': self.voting,
            'is_trained': self.is_trained,
            'model_count': self.get_model_count()
        }