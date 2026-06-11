"""
Моніторинг дрифту фіч
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class FeatureDriftMonitor:
    """Моніторинг дрифту фіч"""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_stats: dict[str, dict[str, Any]] = {}
        self.drift_history: dict[str, list[dict[str, Any]]] = {}

    def fit(self, X: pd.DataFrame, feature_names: list[str] | None = None) -> 'FeatureDriftMonitor':
        """Навчання на референсних даних"""
        if feature_names is None:
            feature_names = list(X.columns)

        self.reference_stats = {}

        for feature in feature_names:
            if feature in X.columns:
                feature_data = X[feature].dropna()
                if len(feature_data) > 0:
                    self.reference_stats[feature] = {
                        'mean': float(feature_data.mean()),
                        'std': float(feature_data.std()),
                        'min': float(feature_data.min()),
                        'max': float(feature_data.max()),
                        'median': float(feature_data.median()),
                        'q25': float(feature_data.quantile(0.25)),
                        'q75': float(feature_data.quantile(0.75))
                    }

        return self

    def detect_drift(self, X: pd.DataFrame, feature_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Детекція дрифту"""
        if feature_names is None:
            feature_names = list(X.columns)

        drift_results = {}

        for feature in feature_names:
            if feature in X.columns and feature in self.reference_stats:
                current_data = X[feature].dropna()
                if len(current_data) > 0:
                    drift_info = self._calculate_feature_drift(feature, current_data)
                    drift_results[feature] = drift_info

        return drift_results

    def _calculate_feature_drift(self, feature: str, current_data: pd.Series) -> dict[str, Any]:
        """Розрахунок дрифту для фічі"""
        ref_stats = self.reference_stats[feature]

        # KS test для числових фіч
        if pd.api.types.is_numeric_dtype(current_data):
            # Створення референсної вибірки (припускаємо нормальний розподіл)
            ref_mean = ref_stats['mean']
            ref_std = ref_stats['std']

            if ref_std > 0:
                ref_sample = np.random.normal(ref_mean, ref_std, len(current_data))

                # KS test
                ks_statistic, p_value = stats.ks_2samp(current_data, ref_sample)

                # Population Stability Index (PSI)
                psi = self._calculate_psi(current_data, ref_sample)

                drift_detected = p_value < self.threshold

                return {
                    'drift_detected': drift_detected,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'psi': psi,
                    'current_mean': current_data.mean(),
                    'reference_mean': ref_mean,
                    'mean_diff': current_data.mean() - ref_mean,
                    'current_std': current_data.std(),
                    'reference_std': ref_std
                }

        return {
            'drift_detected': False,
            'ks_statistic': 0.0,
            'p_value': 1.0,
            'psi': 0.0,
            'current_mean': current_data.mean() if len(current_data) > 0 else None,
            'reference_mean': ref_stats['mean'],
            'mean_diff': 0.0
        }

    def _calculate_psi(self, current_data: pd.Series, reference_data: pd.Series, bins: int = 10) -> float:
        """Розрахунок Population Stability Index"""
        # Об'єднання даних для визначення бінів
        combined_data = np.concatenate([current_data, reference_data])

        if len(np.unique(combined_data)) < 2:
            return 0.0

        # Створення бінів
        _, bin_edges = np.histogram(combined_data, bins=bins)

        # Розрахунок частот
        current_hist, _ = np.histogram(current_data, bins=bin_edges)
        ref_hist, _ = np.histogram(reference_data, bins=bin_edges)

        # Нормалізація
        current_percents = current_hist / len(current_data)
        ref_percents = ref_hist / len(reference_data)

        # Додавання невеликих значень для уникнення ділення на 0
        current_percents = np.where(current_percents == 0, 0.0001, current_percents)
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)

        # Розрахунок PSI
        psi = np.sum((current_percents - ref_percents) * np.log(current_percents / ref_percents))

        return psi

    def get_drifted_features(self, X: pd.DataFrame, feature_names: list[str] = None) -> list[str]:
        """Отримати список фіч з дрифтом"""
        drift_results = self.detect_drift(X, feature_names)

        drifted = [feature for feature, result in drift_results.items()
                  if result.get('drift_detected', False)]

        return drifted

    def get_drift_summary(self) -> dict[str, Any]:
        """Отримати підсумок дрифту"""
        if not self.drift_history:
            return {}

        summary = {
            'total_checks': len(self.drift_history),
            'drift_detected_count': sum(1 for result in self.drift_history.values()
                                     if result.get('drift_detected', False)),
            'last_check': max(self.drift_history.keys()) if self.drift_history else None
        }

        return summary

# Глобальний екземпляр
_global_monitor = None

def check_feature_drift(X: pd.DataFrame, feature_names: list[str] = None, threshold: float = 0.05) -> dict[str, dict[str, Any]]:
    """Швидка перевірка дрифту фіч

    Args:
        X: DataFrame з фічами для перевірки
        feature_names: Список назв фіч для перевірки
        threshold: Поріг для детекції дрифту

    Returns:
        Dict з результатами перевірки дрифту
    """
    try:
        monitor = get_feature_drift_monitor(threshold)
        return monitor.detect_drift(X, feature_names)
    except Exception:
        # Якщо щось пішло не так, повертаємо пустий результат
        return {}

def get_feature_drift_monitor(threshold: float = 0.05) -> FeatureDriftMonitor:
    """Отримати глобальний монітор дрифту фіч"""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = FeatureDriftMonitor(threshold)

    return _global_monitor
