"""
AnomalyEngine: anomaly detection and ensemble confidence scoring
extracted from PredictionStage to reduce file size.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger


class AnomalyEngine:
    """Calculates anomaly scores (Z-score / IsoForest / LOF) and ensemble confidence."""

    def __init__(self, diary: Any=None):
        self.logger = ProjectLogger.get_logger('AnomalyEngine')
        self.diary = diary
        self._estimators_cache: dict[str, Any] = {}

    def calculate_anomaly_score(self, X: pd.DataFrame, context_id: str=
        'default') ->float:
        """Return anomaly score in [0, 1]. Higher → more anomalous."""
        try:
            if X.empty or len(X) < 2:
                return 0.5
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.empty or len(X_numeric) < 2:
                return 0.5
            current_row, historical_data = self._prepare_anomaly_data(X_numeric)
            cache_key = f'{context_id}_{X_numeric.shape[1]}'
            z_score = self._calculate_zscore_anomaly(current_row,
                historical_data)
            iso_score = self._calculate_isolation_forest_anomaly(current_row,
                historical_data, cache_key)
            lof_score = self._calculate_lof_anomaly(current_row,
                historical_data, cache_key)
            final = z_score * 0.4 + iso_score * 0.4 + lof_score * 0.2
            return float(np.clip(final, 0, 1))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Anomaly detection failure: {e}')
            return 0.5

    def calculate_ensemble_confidence(self, models: dict[str, Any], X: pd.
        DataFrame, prediction: float, context_id: str) ->dict[str, float]:
        """Return multi-factor confidence dict with key 'score' in [0, 1]."""
        try:
            accuracy_score = self._fetch_diary_accuracy(context_id)
            volatility_factor = self._calc_volatility_factor(X)

            if not models:
                # No models available - use diary accuracy + volatility only
                final = accuracy_score * 0.6 + volatility_factor * 0.4
                return {'score': float(np.clip(final, 0, 1))}

            raw_preds = self._collect_raw_predictions(models, X)
            if not raw_preds:
                final = accuracy_score * 0.6 + volatility_factor * 0.4
                return {'score': float(np.clip(final, 0, 1))}

            consensus_score, dispersion_score = (self.
                _calc_consensus_dispersion(raw_preds, prediction))
            final = (consensus_score * 0.35 + dispersion_score * 0.25 +
                accuracy_score * 0.25 + volatility_factor * 0.15)
            return {'score': float(np.clip(final, 0, 1))}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'⚠️ Confidence calculation failure: {e}')
            return {'score': 0.5}

    def _prepare_anomaly_data(self, X: pd.DataFrame) ->tuple[np.ndarray, np
        .ndarray]:
        current_row = X.iloc[-1:].values
        historical_data = X.iloc[:-1].values if len(X) > 1 else X.values
        return current_row, historical_data

    def _calculate_zscore_anomaly(self, current_row: np.ndarray,
        historical_data: np.ndarray) ->float:
        try:
            mean = np.mean(historical_data, axis=0)
            std = np.std(historical_data, axis=0)
            z_scores = np.abs((current_row - mean) / (std + 1e-06))
            return float(np.clip(float(np.mean(z_scores)) / 3.0, 0, 1))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return 0.5

    def _calculate_isolation_forest_anomaly(self, current_row: np.ndarray,
        historical_data: np.ndarray, cache_key: str) ->float:
        try:
            if len(historical_data) <= 10:
                return 0.5
            iso_key = f'iso_{cache_key}'
            if iso_key not in self._estimators_cache:
                iso_forest = IsolationForest(contamination=0.1, random_state=42
                    )
                iso_forest.fit(historical_data)
                self._estimators_cache[iso_key] = iso_forest
            pred = self._estimators_cache[iso_key].predict(current_row)
            return 1.0 if pred[0] == -1 else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return 0.5

    def _calculate_lof_anomaly(self, current_row: np.ndarray,
        historical_data: np.ndarray, cache_key: str) ->float:
        try:
            if len(historical_data) <= 10:
                return 0.5
            lof_key = f'lof_{cache_key}'
            if lof_key not in self._estimators_cache:
                lof = LocalOutlierFactor(n_neighbors=min(20, len(
                    historical_data) - 1), novelty=True)
                lof.fit(historical_data)
                self._estimators_cache[lof_key] = lof
            pred = self._estimators_cache[lof_key].predict(current_row)
            return 1.0 if pred[0] == -1 else 0.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return 0.5

    def _collect_raw_predictions(self, models: dict[str, Any], X: pd.DataFrame
        ) ->list[float]:
        raw_preds = []
        for m_inst in models.values():
            try:
                p = m_inst.predict(X)
                val = float(p[-1]) if hasattr(p, '__len__') else float(p)
                raw_preds.append(val)
            except (ValueError, TypeError, AttributeError):
                continue
        return raw_preds

    def _calc_consensus_dispersion(self, raw_preds: list[float], prediction:
        float) ->tuple[float, float]:
        if len(raw_preds) <= 1:
            return 0.5, 0.5
        final_dir = prediction > 0
        agreement = sum(1 for p in raw_preds if (p > 0) == final_dir)
        consensus_score = agreement / len(raw_preds)
        variance = np.var(raw_preds)
        dispersion_score = 1.0 / (1.0 + variance * 5)
        return consensus_score, dispersion_score

    def _fetch_diary_accuracy(self, context_id: str) ->float:
        if self.diary is None:
            return 0.5
        try:
            # Try to get recent trades and calculate accuracy from them
            if hasattr(self.diary, 'get_recent_trades'):
                trades_df = self.diary.get_recent_trades(window=20)
                if not trades_df.empty and 'prediction_sign' in trades_df.columns and 'actual_sign' in trades_df.columns:
                    accuracy = (trades_df['prediction_sign'] == trades_df['actual_sign']).mean()
                    return float(accuracy)
            return 0.5
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Diary accuracy fetch failed for {context_id}: {e}', exc_info=True)
            return 0.5

    def _calc_volatility_factor(self, X: pd.DataFrame) ->float:
        try:
            if len(X) > 5:
                vol = np.std(X.iloc[-10:, 0].values)
                return float(1.0 / (1.0 + vol * 20))
            return 1.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Error calculating volatility factor: {e}', exc_info=True)
            raise DataProcessingError(f"Volatility factor calculation failed: {e}") from e
