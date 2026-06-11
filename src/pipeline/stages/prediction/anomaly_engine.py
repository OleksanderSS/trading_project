"""
AnomalyEngine: anomaly detection and ensemble confidence scoring
extracted from PredictionStage to reduce file size.
"""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.core.logging.logger import ProjectLogger


class AnomalyEngine:
    """Calculates anomaly scores (Z-score / IsoForest / LOF) and ensemble confidence."""

    def __init__(self, diary: Any = None):
        self.logger = ProjectLogger.get_logger("AnomalyEngine")
        self.diary = diary
        # Cache to avoid re-fitting IsolationForest/LOF on every call
        self._estimators_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def calculate_anomaly_score(
        self, X: pd.DataFrame, context_id: str = "default"
    ) -> float:
        """Return normalcy score in [0, 1]. Higher → more NORMAL (less anomalous)."""
        try:
            if X.empty or len(X) < 2:
                return 0.8  # Assume normal when insufficient data

            current_row, historical_data = self._prepare_anomaly_data(X)
            cache_key = f"{context_id}_{X.shape[1]}"

            z_score = self._calculate_zscore_anomaly(current_row, historical_data)
            iso_score = self._calculate_isolation_forest_anomaly(current_row, historical_data, cache_key)
            lof_score = self._calculate_lof_anomaly(current_row, historical_data, cache_key)

            # Weighted anomaly amount (0=normal, 1=anomalous)
            anomaly_amount = z_score * 0.4 + iso_score * 0.4 + lof_score * 0.2
            # Invert → normalcy score (1=normal, 0=anomalous)
            normalcy = float(np.clip(1.0 - anomaly_amount, 0, 1))
            return normalcy

        except Exception as e:
            self.logger.warning(f"Anomaly detection failure: {e}")
            return 0.8  # Safe default — assume mostly normal

    def calculate_ensemble_confidence(
        self,
        models: dict[str, Any],
        X: pd.DataFrame,
        prediction: float,
        context_id: str,
        predictions_by_model: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return multi-factor confidence dict with key 'score' in [0, 1]."""
        try:
            # Prefer pre-computed per-model predictions over live model inference
            if predictions_by_model:
                raw_preds = []
                for v in predictions_by_model.values():
                    try:
                        raw_preds.append(float(v))
                    except (TypeError, ValueError):
                        pass
            elif models:
                raw_preds = self._collect_raw_predictions(models, X)
            else:
                raw_preds = []

            if not raw_preds:
                # Fallback: single-model confidence based on volatility + diary only
                accuracy_score = self._fetch_diary_accuracy(context_id)
                volatility_factor = self._calc_volatility_factor(X)
                score = accuracy_score * 0.6 + volatility_factor * 0.4
                return {'score': float(np.clip(score, 0.3, 0.85))}

            consensus_score, dispersion_score = self._calc_consensus_dispersion(raw_preds, prediction)
            accuracy_score = self._fetch_diary_accuracy(context_id)
            volatility_factor = self._calc_volatility_factor(X)

            final = (
                consensus_score * 0.35
                + dispersion_score * 0.25
                + accuracy_score * 0.25
                + volatility_factor * 0.15
            )
            return {'score': float(np.clip(final, 0, 1))}

        except Exception as e:
            self.logger.warning(f"⚠️ Confidence calculation failure: {e}")
            return {'score': 0.5}

    # ------------------------------------------------------------------
    # Private helpers — anomaly
    # ------------------------------------------------------------------

    def _prepare_anomaly_data(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        current_row = X.iloc[-1:].values
        historical_data = X.iloc[:-1].values if len(X) > 1 else X.values
        return current_row, historical_data

    def _calculate_zscore_anomaly(
        self, current_row: np.ndarray, historical_data: np.ndarray
    ) -> float:
        try:
            mean = np.mean(historical_data, axis=0)
            std = np.std(historical_data, axis=0)
            z_scores = np.abs((current_row - mean) / (std + 1e-6))
            return float(np.clip(float(np.mean(z_scores)) / 3.0, 0, 1))
        except Exception:
            return 0.5

    def _calculate_isolation_forest_anomaly(
        self, current_row: np.ndarray, historical_data: np.ndarray, cache_key: str
    ) -> float:
        try:
            if len(historical_data) <= 10:
                return 0.5
            iso_key = f"iso_{cache_key}"
            if iso_key not in self._estimators_cache:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(historical_data)
                self._estimators_cache[iso_key] = iso_forest
            pred = self._estimators_cache[iso_key].predict(current_row)
            return 1.0 if pred[0] == -1 else 0.0
        except Exception:
            return 0.5

    def _calculate_lof_anomaly(
        self, current_row: np.ndarray, historical_data: np.ndarray, cache_key: str
    ) -> float:
        try:
            if len(historical_data) <= 10:
                return 0.5
            lof_key = f"lof_{cache_key}"
            if lof_key not in self._estimators_cache:
                lof = LocalOutlierFactor(
                    n_neighbors=min(20, len(historical_data) - 1), novelty=True
                )
                lof.fit(historical_data)
                self._estimators_cache[lof_key] = lof
            pred = self._estimators_cache[lof_key].predict(current_row)
            return 1.0 if pred[0] == -1 else 0.0
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Private helpers — confidence
    # ------------------------------------------------------------------

    def _collect_raw_predictions(
        self, models: dict[str, Any], X: pd.DataFrame
    ) -> list[float]:
        raw_preds = []
        for m_inst in models.values():
            try:
                p = m_inst.predict(X)
                val = float(p[-1]) if hasattr(p, '__len__') else float(p)
                raw_preds.append(val)
            except (ValueError, TypeError, AttributeError):
                continue
        return raw_preds

    def _calc_consensus_dispersion(
        self, raw_preds: list[float], prediction: float
    ) -> tuple[float, float]:
        if len(raw_preds) <= 1:
            return 0.5, 0.5
        final_dir = prediction > 0
        agreement = sum(1 for p in raw_preds if (p > 0) == final_dir)
        consensus_score = agreement / len(raw_preds)
        variance = np.var(raw_preds)
        dispersion_score = 1.0 / (1.0 + variance * 5)
        return consensus_score, dispersion_score

    def _fetch_diary_accuracy(self, context_id: str) -> float:
        if self.diary is None:
            return 0.5
        try:
            perf = self.diary.get_recent_performance(context=context_id, window=20)
            return float(perf.get('accuracy', 0.5))
        except Exception:
            return 0.5

    def _calc_volatility_factor(self, X: pd.DataFrame) -> float:
        """Higher score = lower volatility = higher confidence."""
        try:
            # Prefer 'close' column for meaningful volatility
            if 'close' in X.columns and len(X) > 5:
                returns = X['close'].pct_change().dropna()
                vol = float(returns.std())
            elif len(X) > 5:
                vol = float(np.std(X.iloc[-10:, 0].values))
            else:
                return 0.6
            # Map: vol=0 → 1.0, vol=0.05 → ~0.5, vol>0.1 → ~0.17
            return float(1.0 / (1.0 + vol * 20))
        except Exception:
            return 0.6
