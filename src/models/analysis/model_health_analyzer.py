from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.models.analysis.model_analyzer import ModelAnalyzer

logger = ProjectLogger.get_logger("ModelHealthAnalyzer")

class ModelHealthAnalyzer:
    """Сервіс для комплексного аналізу здоров'я та продуктивності моделей."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Delegates to ModelAnalyzer, which builds the model_results dict
        # (model/predictions/metrics) each sub-detector's real API actually
        # requires. The direct calls this class used before
        # (baseline_detector.analyze(), regime_analyzer.analyze(),
        # overfitting_detector.analyze(), drift_monitor.monitor()) targeted
        # method names none of those classes have ever defined.
        self.model_analyzer = ModelAnalyzer(self.config)
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
        analysis_results["baseline"] = await self.model_analyzer.perform_baseline_analysis(
            model, X_train, y_train, X_val, y_val
        )

        # 2. Regime consistency
        if market_data is not None:
            analysis_results["regime"] = await self.model_analyzer.perform_regime_analysis(
                model, market_data, X_train, y_train
            )
        else:
            analysis_results["regime"] = {"status": "no_market_data"}

        # 3. Overfitting detection
        analysis_results["overfitting"] = await self.model_analyzer.perform_overfitting_analysis(
            model, X_train, y_train, X_val, y_val
        )

        # 4. Drift monitoring
        if predictions is not None:
            analysis_results["drift"] = await self.model_analyzer.perform_drift_monitoring(
                predictions, actuals, confidences
            )
        else:
            analysis_results["drift"] = {"status": "no_predictions"}

        # Повертаємо агрегований звіт
        return results
