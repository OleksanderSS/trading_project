"""
Market Regime Detector - Modular Detection System.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.algorithms.regime.clustering import RegimeClusteringEngine
from src.algorithms.regime.metrics import RegimeMetricsCalculator
from src.algorithms.regime.rules import RegimeRulesEngine
from src.algorithms.regime.types import MarketRegime
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger


@dataclass
class RegimeMetrics:
    """Метрики для визначення режиму ринку"""
    returns: np.ndarray
    prices: np.ndarray | None
    volume: np.ndarray | None
    adx: float
    volatility: float
    mean_return: float

class MarketRegimeDetector:
    """Виявляє режими ринку з використанням ML та статистичних методів."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = ProjectLogger.get_logger('MarketRegimeDetector')
        from src.config.unified_config_manager import get_current_config
        self.config = config or get_current_config().get('logic.regime_detection', {})

        self.use_ml_clustering = self.config.get('use_ml_clustering', True)
        self.min_samples_for_clustering = self.config.get('min_samples_for_clustering', 252)

        self.clustering_engine = RegimeClusteringEngine(
            n_clusters=self.config.get('n_clusters', 8),
            min_samples=self.min_samples_for_clustering
        )
        self.rules_engine = RegimeRulesEngine(self.config)
        self.metrics_calculator = RegimeMetricsCalculator()

        self.logger.info("✅ MarketRegimeDetector initialized")

    def detect_regime(self, returns: np.ndarray, data_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        """Виявляє режим ринку з використанням всіх доступних даних."""
        try:
            if len(returns) < 30:
                return {'regime': MarketRegime.NORMAL.value, 'confidence': 0.5, 'reason': 'insufficient_data'}

            bundle = data_bundle or {}
            volatility, mean_return, adx = self.metrics_calculator.calculate_basic_metrics(returns)

            # 1. Check for extreme crisis conditions
            if mean_return < self.config.get('crisis_threshold', -0.05):
                return {'regime': MarketRegime.CRISIS.value, 'confidence': 0.95, 'reason': 'extreme_negative_returns'}

            # 2. Try ML detection
            if self.use_ml_clustering and len(returns) >= self.min_samples_for_clustering:
                ml_result = self.clustering_engine.detect_regime_ml(returns, bundle.get('prices'), bundle.get('volume'), bundle.get('sentiment_data'))
                if ml_result.get('confidence', 0) > 0.7:
                    return ml_result

            # 3. Rule-based detection
            metrics = RegimeMetrics(
                returns=returns,
                prices=bundle.get('prices'),
                volume=bundle.get('volume'),
                adx=adx,
                volatility=volatility,
                mean_return=mean_return
            )
            rule_regime = self.rules_engine.detect_regime_rules(metrics)

            return rule_regime

        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}", exc_info=True)
            raise DataProcessingError("Regime detection failed") from e
