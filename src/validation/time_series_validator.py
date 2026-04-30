"""
Time Series Validation - Robust evaluation for financial data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any, List, Optional
from src.core.logging.logger import ProjectLogger
from src.utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger("TimeSeriesValidator")

class TimeSeriesValidator:
    """
    Provides robust validation protocols for financial time series models.
    Designed as a core pipeline component to ensure data integrity and prevent leakage.
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Initializes the validator.
        
        Args:
            n_splits (int): Number of splits for cross-validation.
        """
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.calendar = TradingCalendar()
        
    def create_robust_split(self, X: pd.DataFrame, y: pd.Series, 
                           validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data chronologically to maintain the temporal order of observations.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            validation_ratio: Proportion of data for validation.
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        x_sorted = X.sort_index()
        y_sorted = y.loc[x_sorted.index]
        
        n_samples = len(x_sorted)
        split_idx = int(n_samples * (1 - validation_ratio))
        
        X_train = x_sorted.iloc[:split_idx]
        x_val = x_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_val = y_sorted.iloc[split_idx:]
        
        logger.info(f"Chronological split successful: {len(X_train)} training and {len(x_val)} validation samples.")
        return X_train, x_val, y_train, y_val
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           scoring: str = 'neg_mean_absolute_error') -> Dict[str, Any]:
        """
        Executes standard TimeSeries cross-validation.
        """
        scores = cross_val_score(model, X, y, cv=self.tscv, scoring=scoring)
        
        return {
            'cv_scores': scores.tolist(),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'n_splits': self.n_splits,
            'scoring': scoring
        }
    
    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
                              window_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
        """
        Executes walk-forward validation (rolling window), mimicking live trading scenarios.
        
        Args:
            model: Model object implementing fit/predict.
            X: Features DataFrame.
            y: Target Series.
            window_size: Size of the rolling training window.
            step_size: Step size for the validation jump.
            
        Returns:
            Dict containing 'fold_metrics' and 'aggregate_metrics'.
        """
        predictions = []
        actuals = []
        fold_metrics = []
        
        n_samples = len(X)
        for i in range(window_size, n_samples, step_size):
            train_end = i
            val_end = min(i + step_size, n_samples)
            
            if val_end <= i: break
            
            X_train, y_train = X.iloc[train_end - window_size : train_end], y.iloc[train_end - window_size : train_end]
            x_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
            
            if len(X_train) < 50: continue
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(x_val)
                
                predictions.extend(y_pred)
                actuals.extend(y_val.values)
                
                fold_metrics.append({
                    'timestamp': str(y_val.index[-1]),
                    'mae': float(mean_absolute_error(y_val, y_pred)),
                    'mse': float(mean_squared_error(y_val, y_pred)),
                    'r2': float(r2_score(y_val, y_pred))
                })
            except Exception as e:
                logger.error(f"Error in walk-forward fold at {y_val.index[0]}: {e}")
        
        if not predictions:
            return {'status': 'error', 'message': 'Insufficient data for validation folds'}

        aggregate_metrics = {
            'mae': float(mean_absolute_error(actuals, predictions)),
            'mse': float(mean_squared_error(actuals, predictions)),
            'r2': float(r2_score(actuals, predictions))
        }

        logger.info(f"Walk-forward validation complete over {len(fold_metrics)} folds.")
        return {
            'fold_metrics': fold_metrics,
            'aggregate_metrics': aggregate_metrics
        }
    
    def validate_time_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates the continuity of the time series using the TradingCalendar.
        
        Args:
            df: DataFrame with a DatetimeIndex.
            
        Returns:
            Report on data integrity and detected gaps.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("Integrity check failed: Index is not a DatetimeIndex.")
            return {'is_valid': False, 'error': 'Index must be DatetimeIndex'}
            
        start_date, end_date = df.index.min(), df.index.max()
        
        # Check trading days specifically
        expected_trading_days = self.calendar.get_trading_days(start=start_date, end=end_date)
        actual_days = df.index.normalize().unique()
        
        missing_trading_days = set(expected_trading_days) - set(actual_days)
        
        report = {
            'is_valid': len(missing_trading_days) == 0,
            'missing_points_count': len(missing_trading_days),
            'missing_dates': sorted(missing_trading_days)[:5],
            'coverage_ratio': 1.0 - (len(missing_trading_days) / len(expected_trading_days)) if len(expected_trading_days) > 0 else 0
        }
        
        if not report['is_valid']:
            logger.warning(f"Data integrity warning: Found {report['missing_points_count']} missing trading days.")
            
        return report

    def check_leakage(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes train and test sets for potential index overlap or temporal violations.
        
        Args:
            X_train: Training feature set.
            X_test: Test/Validation feature set.
            
        Returns:
            Leakage detection summary.
        """
        overlap = X_train.index.intersection(X_test.index)
        
        is_chronological = True
        if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_test.index, pd.DatetimeIndex):
            is_chronological = X_train.index.max() < X_test.index.min()
            
        leakage_detected = len(overlap) > 0 or not is_chronological
        
        if leakage_detected:
            logger.critical(f"Data leakage detected! Overlap: {len(overlap)} samples. Chronological order: {is_chronological}")
            
        return {
            'leakage_detected': leakage_detected,
            'overlap_count': len(overlap),
            'is_chronological': is_chronological,
            'train_max_date': str(X_train.index.max()),
            'test_min_date': str(X_test.index.min())
        }

def create_robust_time_series_split(X: pd.DataFrame, y: pd.Series,
                                   validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Helper function for quick chronological data partitioning."""
    validator = TimeSeriesValidator()
    return validator.create_robust_split(X, y, validation_ratio)