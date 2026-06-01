from typing import Any, cast
import pandas as pd
import numpy as np
from datetime import datetime
from src.core.logging.logger import ProjectLogger
from src.models.analysis.baseline_dominance_detector import BaselineDominanceDetector
from src.models.analysis.overfitting_detector import OverfittingDetector
from src.models.analysis.regime_winner_analyzer import RegimeWinnerAnalyzer
from src.models.monitoring.prediction_drift_monitor import PredictionDriftMonitor

logger = ProjectLogger.get_logger("ModelHealthAnalyzer")

class ModelHealthAnalyzer:
    """Сервіс для комплексного аналізу здоров'я та продуктивності моделей."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.baseline_detector = BaselineDominanceDetector(self.config.get("baseline_detector", {}))
        self.regime_analyzer = RegimeWinnerAnalyzer(self.config.get("regime_analyzer", {}))
        self.overfitting_detector = OverfittingDetector(self.config.get("overfitting_detector", {}))
        self.drift_monitor = PredictionDriftMonitor(self.config.get("drift_monitor", {}))
        logger.info("✅ ModelHealthAnalyzer components initialized")

    async def analyze(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        market_data: pd.DataFrame | None = None,
        predictions: np.ndarray | None = None,
        actuals: np.ndarray | None = None,
        confidences: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Виконує комплексний аналіз моделі."""
        logger.info(f"🔍 Starting comprehensive analysis for model: {model_name}")

        results = {
            "timestamp": datetime.now(),
            "model_name": model_name,
            "model_type": type(model).__name__,
            "analysis_results": {},
            "overall_health_score": 0.0,
            "recommendations": [],
            "action_required": False,
            "retraining_recommended": False,
        }
        
        analysis_results = results["analysis_results"]

        # 1. Baseline dominance
        analysis_results["baseline"] = await self.baseline_detector.analyze(model, X_train, y_train, X_val, y_val)

        # 2. Regime consistency
        if market_data is not None:
            analysis_results["regime"] = await self.regime_analyzer.analyze(model, market_data, X_train, y_train)
        else:
            analysis_results["regime"] = {"status": "no_market_data"}

        # 3. Overfitting detection
        analysis_results["overfitting"] = await self.overfitting_detector.analyze(model, X_train, y_train, X_val, y_val)

        # 4. Drift monitoring
        if predictions is not None:
            analysis_results["drift"] = await self.drift_monitor.monitor(predictions, actuals, confidences)
        else:
            analysis_results["drift"] = {"status": "no_predictions"}

        # Повертаємо агрегований звіт
        return results
