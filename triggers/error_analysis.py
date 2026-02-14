# triggers/error_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ErrorAnalyzer:
    """Аналandwith errors for адаптивної ваги фandчей"""

    def __init__(self):
        self.error_history = []
        self.feature_importance_adjustments = {}

    def analyze_prediction_error(self, prediction: float, actual: float,
                                 features: pd.Series, timestamp: pd.Timestamp) -> Dict:
        """Аналandwithує помилку прогноwithу"""
        error = abs(prediction - actual)
        relative_error = error / abs(actual) if actual != 0 else error

        error_record = {
            "timestamp": timestamp,
            "prediction": prediction,
            "actual": actual,
            "error": error,
            "relative_error": relative_error,
            "features": features
        }

        self.error_history.append(error_record)

        # Аналandwithуємо причини помилки
        error_causes = self._identify_error_causes(error_record)

        return {
            "error_analysis": error_record,
            "causes": error_causes,
            "adjustments": self._calculate_feature_adjustments(error_causes)
        }

    def _identify_error_causes(self, error_record: Dict) -> Dict:
        """Іwhereнтифandкує можливand причини помилки"""
        causes = {}
        features = error_record["features"]

        # 1. Макро подandї
        macro_cols = [col for col in features.index if any(x in col.lower()
                                                           for x in ["cpi", "fed", "vix", "unrate"])]
        for col in macro_cols:
            if abs(features[col]) > 0.5:  # Сильний макро сигнал
                causes[f"macro_{col}"] = "strong_macro_event"

        # 2. Новиннand сюрприwithи
        if "sentiment_score" in features.index:
            if abs(features["sentiment_score"]) > 0.7:
                causes["sentiment_surprise"] = "extreme_sentiment"

        # 3. Технandчнand andндикатори
        tech_cols = [col for col in features.index if any(x in col.lower()
                                                          for x in ["rsi", "macd", "gap"])]
        for col in tech_cols:
            if abs(features[col]) > 2.0:  # Екстремальнand технandчнand values
                causes[f"technical_{col}"] = "extreme_technical"

        return causes

    def _calculate_feature_adjustments(self, causes: Dict) -> Dict:
        """Роwithраховує корекцandї ваг фandчей"""
        adjustments = {}

        for cause, cause_type in causes.items():
            if "macro" in cause:
                # Пandдвищуємо вагу макро фandчей
                adjustments["macro_weight"] = 1.2
            elif "sentiment" in cause:
                adjustments["sentiment_weight"] = 1.3
            elif "technical" in cause:
                adjustments["technical_weight"] = 1.1

        return adjustments

    def get_adaptive_weights(self, feature_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Отримує адаптивнand ваги for груп фandчей"""
        base_weights = {
            "macro": 1.0,
            "sentiment": 1.0,
            "technical": 1.0,
            "calendar": 1.0
        }

        # Застосовуємо корекцandї with осandннandх errors
        recent_errors = self.error_history[-10:]  # Осandннand 10 errors
        for error in recent_errors:
            adjustments = error.get("adjustments", {})
            for group, adjustment in adjustments.items():
                if group in base_weights:
                    base_weights[group] *= adjustment

        # Нормалandforцandя
        total_weight = sum(base_weights.values())
        return {group: weight / total_weight for group, weight in base_weights.items()}