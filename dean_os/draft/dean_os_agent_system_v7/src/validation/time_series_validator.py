"""
Time Series Validation - Robust evaluation for financial data

Consolidated validation system combining:
- TimeSeriesValidator (base functionality)
- ValidationProtocolsEngine (purged/embargo CV)
- SmartCrossValidator (stratified splits)
"""
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.core.logging.logger import ProjectLogger
from src.utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger('TimeSeriesValidator')


class ValidationType(Enum):
    """Validation types for financial ML"""
    WALK_FORWARD = 'walk_forward'
    PURGED_CV = 'purged_cv'
    EMBARGO_CV = 'embargo_cv'
    PURGED_WALK_FORWARD = 'purged_walk_forward'
    STRATIFIED_TS = 'stratified_ts'
    CONSENSUS_STABILITY = 'consensus_stability'


@dataclass
class ValidationResult:
    """Standardized validation output"""
    validation_type: ValidationType
    is_valid: bool
    confidence: float
    performance_metrics: dict[str, float]
    issues_found: list[str]
    recommendations: list[str]
    detailed_results: dict[str, Any]


class PurgedTimeSeriesSplit:
    """
    Advanced Time Series Cross-Validator with Purging and Embargo.
    Prevents data leakage by removing overlapping data points.
    
    Target-horizon aware: Accepts target_horizon to ensure proper temporal separation
    between training features and test targets based on prediction horizon.

    Integrated from ValidationProtocolsEngine.
    """

    def __init__(self, n_splits: int=5, purge_window: int=0, embargo_period:
        int=0, target_horizon: int=1, timestamp_col: str='index', ticker_col: str | None=None):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_period = embargo_period
        self.target_horizon = target_horizon
        self.timestamp_col = timestamp_col
        self.ticker_col = ticker_col

    def split(self, X: pd.DataFrame) ->Generator[tuple[np.ndarray, np.
        ndarray], None, None]:
        """
        Generate train/test splits with purging and embargo based on target horizon.
        
        Args:
            X: Input DataFrame with datetime index or timestamp column
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        # Adjust purge window by target horizon to ensure no leakage
        effective_purge = self.purge_window + self.target_horizon
        effective_embargo = self.embargo_period + self.target_horizon

        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size
            if test_end > n_samples:
                test_end = n_samples
            purged_train_end = train_end - effective_purge
            if purged_train_end <= 0:
                continue
            embargoed_test_start = test_start + effective_embargo
            if embargoed_test_start >= test_end:
                continue
            train_indices = np.arange(0, purged_train_end)
            test_indices = np.arange(embargoed_test_start, test_end)
            yield train_indices, test_indices


class TimeSeriesValidator:
    """
    Provides robust validation protocols for financial time series models.

    Consolidated features:
    - Basic time series validation (original)
    - Purged/Embargo CV (from ValidationProtocolsEngine)
    - Stratified splits (from SmartCrossValidator)
    - Walk-forward validation (enhanced)
    - Data integrity checks
    - Leakage detection
    """

    def __init__(self, n_splits: int=5):
        """
        Initializes the validator.

        Args:
            n_splits (int): Number of splits for cross-validation.
        """
        self.n_splits = n_splits
        self.logger = logger
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        # TimeSeriesSplit is chronological by design and does not support shuffle.
        self.calendar = TradingCalendar()
        self.splits = []
        self.fold_metrics = []

    def create_robust_split(self, X: pd.DataFrame, y: pd.Series,
        validation_ratio: float=0.2) ->tuple[pd.DataFrame, pd.DataFrame, pd
        .Series, pd.Series]:
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
        logger.info(
            f'Chronological split successful: {len(X_train)} training and {len(x_val)} validation samples.'
            )
        return X_train, x_val, y_train, y_val

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
        scoring: str='neg_mean_absolute_error') ->dict[str, Any]:
        """
        Executes standard TimeSeries cross-validation using TimeSeriesSplit.
        """
        # TimeSeriesSplit is chronological by design and does not support shuffle.
        scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=self.n_splits), scoring=scoring)
        return {'cv_scores': scores.tolist(), 'mean_score': float(scores.
            mean()), 'std_score': float(scores.std()), 'n_splits': self.
            n_splits, 'scoring': scoring}

    def walk_forward_validation(self, model, X: pd.DataFrame, y: pd.Series,
        window_size: int=252, step_size: int=21) ->dict[str, Any]:
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
            if val_end <= i:
                break
            X_train, y_train = X.iloc[train_end - window_size:train_end
                ], y.iloc[train_end - window_size:train_end]
            x_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
            if len(X_train) < 50:
                continue
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(x_val)
                predictions.extend(y_pred)
                actuals.extend(y_val.values)
                fold_metrics.append({'timestamp': str(y_val.index[-1]),
                    'mae': float(mean_absolute_error(y_val, y_pred)), 'mse':
                    float(mean_squared_error(y_val, y_pred)), 'r2': float(
                    r2_score(y_val, y_pred))})
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(
                    f'Error in walk-forward fold at {y_val.index[0]}: {e}')
        if not predictions:
            return {'status': 'error', 'message':
                'Insufficient data for validation folds'}
        aggregate_metrics = {'mae': float(mean_absolute_error(actuals,
            predictions)), 'mse': float(mean_squared_error(actuals,
            predictions)), 'r2': float(r2_score(actuals, predictions))}
        logger.info(
            f'Walk-forward validation complete over {len(fold_metrics)} folds.'
            )
        return {'fold_metrics': fold_metrics, 'aggregate_metrics':
            aggregate_metrics}

    def purged_walk_forward_validation(self, model, X: pd.DataFrame, y: pd.
        Series, purge_window: int=5, embargo_period: int=10
        ) ->ValidationResult:
        """
        Purged walk-forward validation with embargo to prevent data leakage.

        Args:
            model: Model object implementing fit/predict.
            X: Features DataFrame.
            y: Target Series.
            purge_window: Number of samples to purge before test set.
            embargo_period: Number of samples to embargo after train set.

        Returns:
            ValidationResult with detailed metrics.
        """
        ps = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_window=
            purge_window, embargo_period=embargo_period)
        metrics, predictions, actuals = [], [], []
        for train_idx, test_idx in ps.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics.append(mean_squared_error(y_test, y_pred))
                predictions.extend(y_pred)
                actuals.extend(y_test)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'Fold failed in purged walk-forward: {e}')
                raise
        is_valid = len(metrics) > 0
        avg_mse = np.mean(metrics) if is_valid else 1.0
        return ValidationResult(validation_type=ValidationType.
            PURGED_WALK_FORWARD, is_valid=is_valid and avg_mse < 0.05,
            confidence=0.9 if is_valid else 0.0, performance_metrics={'mse':
            avg_mse, 'folds': len(metrics)}, issues_found=[
            'Low performance' if avg_mse > 0.05 else ''], recommendations=[
            'Check for remaining leakage' if avg_mse < 0.001 else 'Stable'],
            detailed_results={'fold_errors': metrics})

    def stratified_time_series_split(self, X: pd.DataFrame, y: pd.Series,
        n_bins: int=5) ->Generator[tuple[np.ndarray, np.ndarray], None, None]:
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
                train_distribution = pd.Series(y_binned[train_idx]
                    ).value_counts().to_dict()
                test_distribution = pd.Series(y_binned[test_idx]).value_counts(
                    ).to_dict()
                self.splits.append({'fold': fold + 1, 'train_size': len(
                    train_idx), 'test_size': len(test_idx),
                    'train_distribution': train_distribution,
                    'test_distribution': test_distribution})
                yield train_idx, test_idx

    def run_comprehensive_validation(self, data: pd.DataFrame, features:
        list[str], target: str, model: Any, heavy_model: Any | None=None,
        light_model: Any | None=None) ->dict[str, ValidationResult]:
        """
        Main entry point for Pipeline stages to get a full quality report.

        Runs multiple validation protocols:
        - Purged Walk-Forward
        - Purged CV
        - Embargo CV
        - Consensus Stability (if heavy/light models provided)

        Args:
            data: Complete dataset with features and target.
            features: List of feature column names.
            target: Target column name.
            model: Primary model to validate.
            heavy_model: Optional heavy model for consensus validation.
            light_model: Optional light model for consensus validation.

        Returns:
            Dictionary of ValidationResult objects by protocol name.
        """
        logger.info(f'Executing comprehensive validation for target: {target}')
        results = {}
        results['purged_walk_forward'] = self.purged_walk_forward_validation(
            model, data[features], data[target])
        results['purged_cv'] = self._run_purged_cv(data, features, target,
            model)
        results['embargo_cv'] = self._run_embargo_cv(data, features, target,
            model)
        if heavy_model and light_model:
            results['consensus_stability'] = self._run_consensus_validation(
                data, features, target, heavy_model, light_model)
        return results

    def _run_purged_cv(self, data: pd.DataFrame, features: list[str],
        target: str, model: Any) ->ValidationResult:
        """Runs CV with purging."""
        ps = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_window=5,
            embargo_period=0)
        scores = []
        for train_idx, test_idx in ps.split(data):
            try:
                model.fit(data.iloc[train_idx][features], data.iloc[
                    train_idx][target])
                scores.append(model.score(data.iloc[test_idx][features],
                    data.iloc[test_idx][target]))
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'CV Fold failed for purged_cv: {e}')
                continue
        mean_score = np.mean(scores) if scores else 0.0
        return ValidationResult(validation_type=ValidationType.PURGED_CV,
            is_valid=mean_score > 0.1, confidence=0.8, performance_metrics=
            {'mean_r2_or_acc': mean_score}, issues_found=[],
            recommendations=[], detailed_results={})

    def _run_embargo_cv(self, data: pd.DataFrame, features: list[str],
        target: str, model: Any) ->ValidationResult:
        """Runs CV with embargo."""
        ps = PurgedTimeSeriesSplit(n_splits=self.n_splits, purge_window=0,
            embargo_period=10)
        scores = []
        for train_idx, test_idx in ps.split(data):
            try:
                model.fit(data.iloc[train_idx][features], data.iloc[
                    train_idx][target])
                scores.append(model.score(data.iloc[test_idx][features],
                    data.iloc[test_idx][target]))
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                logger.warning(f'CV Fold failed for embargo_cv: {e}')
                continue
        mean_score = np.mean(scores) if scores else 0.0
        return ValidationResult(validation_type=ValidationType.EMBARGO_CV,
            is_valid=mean_score > 0.1, confidence=0.8, performance_metrics=
            {'mean_r2_or_acc': mean_score}, issues_found=[],
            recommendations=[], detailed_results={})

    def _run_consensus_validation(self, data: pd.DataFrame, features: list[
        str], target: str, heavy_model: Any, light_model: Any
        ) ->ValidationResult:
        """Validates consensus between heavy and light models."""
        min_test_size = 100
        if len(data) < min_test_size:
            return ValidationResult(ValidationType.CONSENSUS_STABILITY,
                False, 0.0, {}, ['Data too small'], [], {})
        split_idx = int(len(data) * 0.8)
        X_train, y_train = data.iloc[:split_idx][features], data.iloc[:
            split_idx][target]
        X_test, _ = data.iloc[split_idx:][features], data.iloc[split_idx:][
            target]
        try:
            heavy_model.fit(X_train, y_train)
            light_model.fit(X_train, y_train)
            h_pred, l_pred = heavy_model.predict(X_test), light_model.predict(
                X_test)
            agreement = np.mean(np.sign(h_pred) == np.sign(l_pred))
            agreement_threshold = 0.7
            return ValidationResult(validation_type=ValidationType.
                CONSENSUS_STABILITY, is_valid=agreement >=
                agreement_threshold, confidence=agreement,
                performance_metrics={'agreement_rate': agreement},
                issues_found=[] if agreement >= agreement_threshold else [
                'Low agreement'], recommendations=[], detailed_results={})
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Consensus validation error: {e}')
            return ValidationResult(ValidationType.CONSENSUS_STABILITY,
                False, 0.0, {}, [str(e)], [], {})

    def validate_time_gaps(self, df: pd.DataFrame) ->dict[str, Any]:
        """
        Validates the continuity of the time series using the TradingCalendar.

        Args:
            df: DataFrame with a DatetimeIndex.

        Returns:
            Report on data integrity and detected gaps.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                'Integrity check failed: Index is not a DatetimeIndex.')
            return {'is_valid': False, 'error': 'Index must be DatetimeIndex'}
        start_date, end_date = df.index.min(), df.index.max()
        expected_trading_days = self.calendar.get_trading_days(start=
            start_date, end=end_date)
        actual_days = df.index.normalize().unique()
        missing_trading_days = set(expected_trading_days) - set(actual_days)
        report = {'is_valid': len(missing_trading_days) == 0,
            'missing_points_count': len(missing_trading_days),
            'missing_dates': sorted(missing_trading_days)[:5],
            'coverage_ratio': 1.0 - len(missing_trading_days) / len(
            expected_trading_days) if len(expected_trading_days) > 0 else 0}
        if not report['is_valid']:
            logger.warning(
                f"Data integrity warning: Found {report['missing_points_count']} missing trading days."
                )
        return report

    def check_leakage(self, X_train: pd.DataFrame, X_test: pd.DataFrame
        ) ->dict[str, Any]:
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
        if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_test
            .index, pd.DatetimeIndex):
            is_chronological = X_train.index.max() < X_test.index.min()
        leakage_detected = len(overlap) > 0 or not is_chronological
        if leakage_detected:
            logger.critical(
                f'Data leakage detected! Overlap: {len(overlap)} samples. Chronological order: {is_chronological}'
                )
        return {'leakage_detected': leakage_detected, 'overlap_count': len(
            overlap), 'is_chronological': is_chronological,
            'train_max_date': str(X_train.index.max()), 'test_min_date':
            str(X_test.index.min())}


def create_robust_time_series_split(X: pd.DataFrame, y: pd.Series,
    validation_ratio: float=0.2) ->tuple[pd.DataFrame, pd.DataFrame, pd.
    Series, pd.Series]:
    """Helper function for quick chronological data partitioning."""
    validator = TimeSeriesValidator()
    return validator.create_robust_split(X, y, validation_ratio)


ValidationProtocolsEngine = TimeSeriesValidator
SmartCrossValidator = TimeSeriesValidator
