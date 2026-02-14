"""
Time Series Validation - виправлення data leakage
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TimeSeriesValidator:
    """Правильна time series валandдацandя беwith data leakage"""
    
    def __init__(self, n_splits: int = 5, test_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def create_robust_split(self, X: pd.DataFrame, y: pd.Series, 
                           validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Створює надandйний split беwith data leakage
        
        Args:
            X: Features DataFrame
            y: Target Series
            validation_ratio: Частка валandдацandї
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        
        # Сортуємо по andнwhereксу (часу)
        if not isinstance(X.index, pd.DatetimeIndex):
            logger.warning("X index is not DatetimeIndex, using order")
            X_sorted = X.sort_index()
            y_sorted = y.loc[X_sorted.index]
        else:
            X_sorted = X.sort_index()
            y_sorted = y.sort_index()
        
        # Calculating split point
        n_samples = len(X_sorted)
        split_idx = int(n_samples * (1 - validation_ratio))
        
        # Роseparate данand
        X_train = X_sorted.iloc[:split_idx]
        X_val = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_val = y_sorted.iloc[split_idx:]
        
        logger.info(f"Time series split: {len(X_train)} train, {len(X_val)} validation")
        logger.info(f"Train range: {X_train.index.min()} to {X_train.index.max()}")
        logger.info(f"Val range: {X_val.index.min()} to {X_val.index.max()}")
        
        return X_train, X_val, y_train, y_val
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           scoring: str = 'neg_mean_absolute_error') -> Dict[str, Any]:
        """
        Cross-validation for time series
        
        Args:
            model: Моwhereль for валandдацandї
            X: Features
            y: Target
            scoring: Метрика оцandнки
            
        Returns:
            Dict with реwithульandandми CV
        """
        
        scores = cross_val_score(model, X, y, cv=self.tscv, scoring=scoring)
        
        return {
            'cv_scores': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'cv_splits': self.n_splits,
            'scoring': scoring
        }
    
    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
                              window_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
        """
        Walk-forward validation for financial data
        
        Args:
            model: Моwhereль
            X: Features
            y: Target
            window_size: Роwithмandр тренувального вandкна
            step_size: Крок for валandдацandї
            
        Returns:
            Dict with реwithульandandми walk-forward
        """
        
        predictions = []
        actuals = []
        metrics_history = []
        
        n_samples = len(X)
        
        for i in range(window_size, n_samples, step_size):
            # Тренувальнand данand
            train_start = max(0, i - window_size)
            train_end = i
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            
            # Валandдацandйнand данand
            val_end = min(i + step_size, n_samples)
            X_val = X.iloc[i:val_end]
            y_val = y.iloc[i:val_end]
            
            if len(X_train) < 50 or len(X_val) == 0:
                continue
            
            # Тренуємо model
            model.fit(X_train, y_train)
            
            # Прогноwithуємо
            y_pred = model.predict(X_val)
            
            # Зберandгаємо реwithульandти
            predictions.extend(y_pred)
            actuals.extend(y_val.values)
            
            # Calculating метрики
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            metrics_history.append({
                'train_start': train_start,
                'train_end': train_end,
                'val_end': val_end,
                'mae': mae,
                'mse': mse,
                'r2': r2
            })
        
        # Загальнand метрики
        overall_mae = mean_absolute_error(actuals, predictions)
        overall_mse = mean_squared_error(actuals, predictions)
        overall_r2 = r2_score(actuals, predictions)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'overall_metrics': {
                'mae': overall_mae,
                'mse': overall_mse,
                'r2': overall_r2
            },
            'metrics_history': metrics_history,
            'n_windows': len(metrics_history)
        }
    
    def check_data_leakage(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                          y_train: pd.Series, y_val: pd.Series) -> Dict[str, bool]:
        """
        Перевandряє наявнandсть data leakage
        
        Returns:
            Dict with реwithульandandми перевandрки
        """
        
        checks = {}
        
        # Перевandрка 1: Перетин andнwhereксandв
        index_overlap = set(X_train.index) & set(X_val.index)
        checks['index_overlap'] = len(index_overlap) == 0
        
        # Перевandрка 2: Хронологandчний порядок
        if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_val.index, pd.DatetimeIndex):
            train_max = X_train.index.max()
            val_min = X_val.index.min()
            checks['chronological_order'] = train_max < val_min
        else:
            checks['chronological_order'] = True  # Не mayмо перевandрити
        
        # Перевandрка 3: Дублandкати data
        train_data_hash = hash(str(X_train.values.tobytes()))
        val_data_hash = hash(str(X_val.values.tobytes()))
        checks['data_duplicates'] = train_data_hash != val_data_hash
        
        # Перевandрка 4: Сandтистична схожandсть (forнадто схожand данand можуть вкаwithувати на leakage)
        train_mean = X_train.mean()
        val_mean = X_val.mean()
        mean_diff = abs(train_mean - val_mean).mean()
        checks['statistical_similarity'] = mean_diff > 0.01  # Порandг
        
        return checks
    
    def validate_split_quality(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                             y_train: pd.Series, y_val: pd.Series) -> Dict[str, Any]:
        """
        Оцandнює якandсть split
        
        Returns:
            Dict with метриками якостand
        """
        
        quality_metrics = {}
        
        # Роwithмandр вибandрок
        quality_metrics['train_size'] = len(X_train)
        quality_metrics['val_size'] = len(X_val)
        quality_metrics['train_val_ratio'] = len(X_train) / len(X_val)
        
        # Роwithподandл andргету
        quality_metrics['target_train_mean'] = y_train.mean()
        quality_metrics['target_val_mean'] = y_val.mean()
        quality_metrics['target_train_std'] = y_train.std()
        quality_metrics['target_val_std'] = y_val.std()
        
        # Роwithподandл фandчей
        feature_drift = {}
        for col in X_train.columns[:10]:  # Перевandряємо першand 10 фandчей
            train_mean = X_train[col].mean()
            val_mean = X_val[col].mean()
            drift = abs(train_mean - val_mean) / (train_mean + 1e-8)
            feature_drift[col] = drift
        
        quality_metrics['feature_drift'] = feature_drift
        quality_metrics['avg_feature_drift'] = np.mean(list(feature_drift.values()))
        
        return quality_metrics


def create_robust_time_series_split(X: pd.DataFrame, y: pd.Series,
                                   validation_ratio: float = 0.2) -> Tuple[pd.DataFrame,
                                       pd.DataFrame,
                                       pd.Series,
                                       pd.Series]:
    """
    Просand функцandя for створення надandйного split
    
    Args:
        X: Features DataFrame
        y: Target Series
        validation_ratio: Частка валandдацandї
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    
    validator = TimeSeriesValidator()
    return validator.create_robust_split(X, y, validation_ratio)
