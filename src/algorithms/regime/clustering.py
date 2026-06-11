import logging
from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import DataProcessingError

logger = logging.getLogger(__name__)

class RegimeClusteringEngine:
    """Виконує ML-кластеризацію для визначення режимів ринку."""

    def __init__(self, n_clusters: int, min_samples: int, seed: int = 42):
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.seed = seed
        self.scaler = StandardScaler()
        self.cluster_model: KMeans | None = None
        self.logger = logger

    def detect_regime_ml(self, returns: np.ndarray, prices: np.ndarray | None, volume: np.ndarray | None, sentiment: np.ndarray | None) -> dict[str, Any]:
        """ML-based regime detection using clustering."""
        try:
            self._ensure_cluster_model_fitted()
            features = self._extract_ml_features(returns, prices, volume, sentiment)
            features_scaled = self._normalize_features(features)

            model = cast(KMeans, self.cluster_model)
            cluster = model.predict(features_scaled)[0]

            regime = self._cluster_to_regime(cluster)
            confidence = self._calculate_ml_confidence(features_scaled)

            return {
                'regime': regime.value,
                'confidence': float(confidence),
                'method': 'ml_clustering',
                'cluster': int(cluster)
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise DataProcessingError("ML regime detection failed") from e

    def _extract_ml_features(self, returns: np.ndarray, prices: np.ndarray | None, volume: np.ndarray | None, sentiment: np.ndarray | None) -> list[float]:
        features = [float(np.mean(returns)), float(np.std(returns)), float(np.min(returns)), float(np.max(returns))]

        if prices is not None and len(prices) > 20:
            trend = np.polyfit(np.arange(len(prices)), prices, 1)[0]
            features.append(float(trend))
        else:
            features.append(0.0)

        if volume is not None and len(volume) > 20:
            volume_sma = np.mean(volume[-20:])
            features.append(float(volume[-1] / volume_sma if volume_sma > 0 else 1.0))
        else:
            features.append(1.0)

        if sentiment is not None and len(sentiment) > 0:
            features.append(float(np.mean(sentiment)))
        else:
            features.append(0.0)

        return features

    def _normalize_features(self, features: list[float]) -> np.ndarray:
        features_array = np.array(features).reshape(1, -1)
        return self.scaler.transform(features_array)

    def _ensure_cluster_model_fitted(self):
        if self.cluster_model is None:
            self._initialize_cluster_centers()

    def _calculate_ml_confidence(self, features_scaled: np.ndarray) -> float:
        if self.cluster_model is None:
            raise DataProcessingError("Cluster model not initialized")
        distances = self.cluster_model.transform(features_scaled)[0]
        min_distance = float(np.min(distances))
        return max(0.5, 1.0 - min_distance)

    def _initialize_cluster_centers(self):
        centers = np.array([
            [0.001, 0.015, -0.03, 0.03, 0.001, 0.5, 0.1],
            [-0.001, 0.015, -0.03, 0.03, -0.001, -0.5, -0.1],
            [0.0, 0.008, -0.01, 0.01, 0.0, 0.0, 0.0],
            [0.0, 0.035, -0.08, 0.08, 0.0, 0.0, 0.0],
            [-0.005, 0.04, -0.15, 0.05, -0.003, -1.0, -0.3],
            [0.0, 0.012, -0.025, 0.025, 0.0, 0.0, 0.0],
            [0.003, 0.025, -0.02, 0.06, 0.002, 1.0, 0.2],
            [0.002, 0.03, -0.04, 0.08, 0.001, 0.8, 0.3]
        ])
        self.scaler.fit(centers)
        self.cluster_model = KMeans(n_clusters=self.n_clusters, init=centers, n_init=1, random_state=self.seed)
        self.cluster_model.fit(centers)

    def _cluster_to_regime(self, cluster: int) -> Any:
        from src.algorithms.regime.types import MarketRegime
        cluster_regime_map = {
            0: MarketRegime.TRENDING_UP, 1: MarketRegime.TRENDING_DOWN,
            2: MarketRegime.RANGING, 3: MarketRegime.VOLATILE,
            4: MarketRegime.CRISIS, 5: MarketRegime.MEAN_REVERSION,
            6: MarketRegime.MOMENTUM, 7: MarketRegime.BREAKOUT
        }
        return cluster_regime_map.get(cluster, MarketRegime.NORMAL)
