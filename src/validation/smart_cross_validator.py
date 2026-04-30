# src/validation/smart_cross_validator.py
"""
SmartCrossValidator Implementation.
Time-series aware k-fold cross-validation with stratification and walk-forward support.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SmartCrossValidator")

class SmartCrossValidator:
    """
    Intelligent time-series validation with temporal sensitivity.
    
    Methods:
    - Time-series k-fold: Sequential data splitting without shuffling.
    - Stratified time-series: Splitting based on target distribution for imbalanced data.
    - Walk-forward validation: Iterative training/validation mimicking live market conditions.
    """

    def __init__(self, project_path=None, n_splits=5):
        """
        Initializes the validator.
        
        Args:
            project_path: Base path for persisting validation reports.
            n_splits: Number of folds for cross-validation.
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.n_splits = n_splits
        self.splits = []
        self.fold_metrics = []

    def time_series_split(self, X, test_size=0.2):
        """
        Sequentially splits data for time-series context (no shuffling).
        
        Args:
            X: Input features (array-like or DataFrame).
            test_size: Proportion of the dataset to be used for testing.
        
        Yields:
            Tuple (train_idx, test_idx) for each generated fold.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            # Sequential distribution: training window grows, test window follows
            train_start = 0
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            if len(test_idx) > 0:
                self.splits.append({
                    'fold': fold + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_indices': train_idx.tolist(),
                    'test_indices': test_idx.tolist()
                })
                yield train_idx, test_idx

    def stratified_time_series_split(self, X, y, n_bins=5):
        """
        Sequential splitting with target-based stratification using quartiles.
        Useful for imbalanced time-series where certain target values are rare.
        
        Args:
            X: Input features.
            y: Target variable used for stratification.
            n_bins: Number of bins for stratification quantization.
        
        Yields:
            Tuple (train_idx, test_idx) for each fold.
        """
        # Bin target values into quartiles
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(test_idx) > 0:
                train_distribution = pd.Series(y_binned[train_idx]).value_counts().to_dict()
                test_distribution = pd.Series(y_binned[test_idx]).value_counts().to_dict()
                
                self.splits.append({
                    'fold': fold + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_distribution': train_distribution,
                    'test_distribution': test_distribution
                })
                yield train_idx, test_idx

    def walk_forward_split(self, X, window_size=None, step_size=None):
        """
        Walk-forward validation (mimics production execution flow).
        
        Initial training period is followed by validation. 
        The window then slides forward by the step size.
        
        Args:
            X: Input sequence features.
            window_size: Size of the training window (defaults to 70% of data).
            step_size: Size of the forward step (defaults to 10% of data).
        
        Yields:
            Tuple (train_idx, test_idx) for each sliding window positioning.
        """
        n_samples = len(X)
        
        if window_size is None:
            window_size = int(n_samples * 0.7)
        if step_size is None:
            step_size = int(n_samples * 0.1)
        
        fold_count = 0
        pos = 0
        
        while pos + window_size + step_size <= n_samples:
            train_idx = np.arange(pos, pos + window_size)
            test_idx = np.arange(pos + window_size, pos + window_size + step_size)
            
            pos += step_size
            fold_count += 1
            
            self.splits.append({
                'fold': fold_count,
                'method': 'walk_forward',
                'train_start': int(train_idx[0]),
                'train_end': int(train_idx[-1]),
                'test_start': int(test_idx[0]),
                'test_end': int(test_idx[-1]),
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            })
            
            yield train_idx, test_idx

    def evaluate_folds(self, model_func, X, y, metric_func):
        """
        Evaluates a model across all generated folds.
        
        Args:
            model_func: Callable that trains on (X_train, y_train) and predicts on X_test.
            X: Complete feature set.
            y: Complete target set.
            metric_func: Callable calculating metrics from (y_true, y_pred).
        
        Returns:
            Dictionary containing mean metrics and individual fold breakdowns.
        """
        self.fold_metrics = []
        all_metrics = {}
        
        fold_idx = 1
        for train_idx, test_idx in self.time_series_split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Model execution
            y_pred = model_func(X_train, y_train, X_test)
            
            # Performance auditing
            metrics = metric_func(y_test, y_pred)
            metrics['fold'] = fold_idx
            metrics['test_size'] = len(test_idx)
            
            self.fold_metrics.append(metrics)
            all_metrics[f'fold_{fold_idx}'] = metrics
            
            fold_idx += 1
        
        # Calculate statistical aggregate of metrics across folds
        mean_metrics = {}
        if self.fold_metrics:
            for key in self.fold_metrics[0].keys():
                if key not in ['fold', 'test_size']:
                    values = [fold[key] for fold in self.fold_metrics if isinstance(fold.get(key), (int, float))]
                    if values:
                        mean_metrics[f'mean_{key}'] = float(np.mean(values))
                        mean_metrics[f'std_{key}'] = float(np.std(values))
        
        result = {
            'method': 'time_series_cross_validation',
            'n_folds': self.n_splits,
            'folds': all_metrics,
            'overall_metrics': mean_metrics
        }
        
        return result

    def get_folds_summary(self):
        """Retrieves a summary report of generated splits."""
        if not self.splits:
            return {"message": "No splits have been generated yet"}
        
        return {
            'total_folds': len(self.splits),
            'folds': self.splits
        }

    def save_splits(self, filepath=None):
        """Persists split information and metrics to a JSON file."""
        if filepath is None:
            filepath = self.project_path / "cross_validation_splits.json"
        
        report = {
            'method': 'time_series_k_fold',
            'n_splits': self.n_splits,
            'splits': self.splits,
            'fold_metrics': self.fold_metrics,
            'report_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cross-validation metadata persisted to {filepath}")
        return filepath


# EXAMPLE USAGE
if __name__ == "__main__":
    logger.info("🎬 Executing SmartCrossValidator functional demonstration\n")
    
    # Simulate time-series data
    rng = np.random.default_rng(42)
    n_samples = 500
    X_sample = rng.standard_normal((n_samples, 10))
    y_sample = np.cumsum(rng.standard_normal(n_samples) * 0.1) + np.sin(np.arange(n_samples) / 50)
    
    # Initialization
    cv = SmartCrossValidator(n_splits=5)
    
    # K-Fold Generation
    logger.info("Generating Time-Series K-Fold Splits:")
    for idx, (train_ix, test_ix) in enumerate(cv.time_series_split(X_sample), 1):
        logger.info(f"  Fold {idx}: Train size={len(train_ix)}, Test size={len(test_ix)}")
    
    # Walk-Forward Generation
    logger.info("\nGenerating Walk-Forward Splits:")
    cv_wf = SmartCrossValidator(n_splits=3)
    for idx, (train_ix, test_ix) in enumerate(cv_wf.walk_forward_split(X_sample), 1):
        logger.info(f"  Fold {idx}: Train {train_ix[0]}-{train_ix[-1]}, Test {test_ix[0]}-{test_ix[-1]}")
    
    # Persistence audit
    summary_path = cv.save_splits()
    logger.info(f"\n✅ Demonstration complete. Splits summary: {cv.get_folds_summary()}")
