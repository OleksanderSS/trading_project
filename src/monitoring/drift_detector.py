"""
Детектор дрифту фіч
"""


import pandas as pd


class DriftDetector:
    """Детектор дрифту фіч"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_stats = {}

    def fit(self, X: pd.DataFrame) -> 'DriftDetector':
        """Навчання на референсних даних"""
        self.reference_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max()
        }
        return self

    def detect_drift(self, X: pd.DataFrame) -> dict[str, float]:
        """Детекція дрифту"""
        drift_scores = {}

        for column in X.columns:
            if column in self.reference_stats['mean'].index:
                # Простий детектор дрифту на основі зміни середнього
                ref_mean = self.reference_stats['mean'][column]
                current_mean = X[column].mean()

                if ref_mean != 0:
                    drift_score = abs(current_mean - ref_mean) / abs(ref_mean)
                else:
                    drift_score = abs(current_mean)

                drift_scores[column] = drift_score

        return drift_scores

    def get_drifted_features(self, X: pd.DataFrame) -> list[str]:
        """Отримати список фіч з дрифтом"""
        drift_scores = self.detect_drift(X)
        drifted = [col for col, score in drift_scores.items()
                  if score > self.threshold]
        return drifted
