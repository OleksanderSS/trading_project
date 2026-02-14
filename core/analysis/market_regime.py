# core/analysis/market_regime.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class MarketRegimeIndicator:
    """
    Свandтлофор: withелений/жовтий/червоний for оцandнки сandну ринку
    """

    def __init__(self):
        self.regime_thresholds = {
            "vix_high": 25.0,  # VIX вище 25 = червоний
            "vix_low": 15.0,  # VIX нижче 15 = withелений
            "volatility_high": 0.03,  # Волатильнandсть вище 3% = червоний
            "volatility_low": 0.01,  # Волатильнandсть нижче 1% = withелений
            "gap_large": 0.02,  # Геп вище 2% = жовтий/червоний
            "sentiment_extreme": 0.7  # Сентимент вище 70% = обережнandсть
        }

    def analyze_market_regime(self, df: pd.DataFrame) -> Dict:
        """Аналіз режиму ринку"""
        if df.empty:
            return {"regime": "unknown", "score": 0, "signals": {}}
        
        current_row = df.iloc[-1]
        regime_signals = {}
        regime_score = 0

        # 1. VIX сигнал
        if "VIX_SIGNAL" in current_row.index:
            vix_value = current_row["VIX_SIGNAL"]
            if vix_value > self.regime_thresholds["vix_high"]:
                regime_signals["vix"] = "RED"
                regime_score -= 2
            elif vix_value < self.regime_thresholds["vix_low"]:
                regime_signals["vix"] = "GREEN"
                regime_score += 1
            else:
                regime_signals["vix"] = "YELLOW"
                regime_score -= 0.5

        # 2. Волатильність
        volatility_cols = [col for col in current_row.index if "vol" in col.lower()]
        if volatility_cols:
            avg_volatility = current_row[volatility_cols].abs().mean()
            if avg_volatility > self.regime_thresholds["volatility_high"]:
                regime_signals["volatility"] = "RED"
                regime_score -= 2
            elif avg_volatility < self.regime_thresholds["volatility_low"]:
                regime_signals["volatility"] = "GREEN"
                regime_score += 1
            else:
                regime_signals["volatility"] = "YELLOW"
                regime_score -= 0.5

        # 3. Визначення режиму
        if regime_score >= 1:
            regime = "GREEN"
        elif regime_score <= -2:
            regime = "RED"
        else:
            regime = "YELLOW"

        return {
            "regime": regime,
            "score": regime_score,
            "signals": regime_signals
        }
    
    def assess_market_regime(self, df: pd.DataFrame, current_row: pd.Series = None) -> Dict:
        if current_row is None and not df.empty:
            current_row = df.iloc[-1]

        regime_signals = {}
        regime_score = 0

        # 1. VIX сигнал
        if "VIX_SIGNAL" in current_row.index:
            vix_value = current_row["VIX_SIGNAL"]
            if vix_value > self.regime_thresholds["vix_high"]:
                regime_signals["vix"] = "RED"
                regime_score -= 2
            elif vix_value < self.regime_thresholds["vix_low"]:
                regime_signals["vix"] = "GREEN"
                regime_score += 1
            else:
                regime_signals["vix"] = "YELLOW"
                regime_score -= 0.5

        # 2. Волатильнandсть
        volatility_cols = [col for col in current_row.index if "vol" in col.lower()]
        if volatility_cols:
            avg_volatility = current_row[volatility_cols].abs().mean()
            if avg_volatility > self.regime_thresholds["volatility_high"]:
                regime_signals["volatility"] = "RED"
                regime_score -= 2
            elif avg_volatility < self.regime_thresholds["volatility_low"]:
                regime_signals["volatility"] = "GREEN"
                regime_score += 1
            else:
                regime_signals["volatility"] = "YELLOW"
                regime_score -= 0.5

        # 3. Гепи
        gap_cols = [col for col in current_row.index if "gap_percent" in col]
        if gap_cols:
            max_gap = current_row[gap_cols].abs().max()
            if max_gap > self.regime_thresholds["gap_large"]:
                regime_signals["gap"] = "RED"
                regime_score -= 1
            else:
                regime_signals["gap"] = "GREEN"
                regime_score += 0.5

        # 4. Сентимент
        if "sentiment_score" in current_row.index:
            sentiment = abs(current_row["sentiment_score"])
            if sentiment > self.regime_thresholds["sentiment_extreme"]:
                regime_signals["sentiment"] = "YELLOW"
                regime_score -= 0.5
            else:
                regime_signals["sentiment"] = "GREEN"
                regime_score += 0.5

        # Фandнальний вердикт
        if regime_score >= 1.5:
            final_regime = "GREEN"
            confidence = min(1.0, regime_score / 3.0)
        elif regime_score <= -1.5:
            final_regime = "RED"
            confidence = min(1.0, abs(regime_score) / 3.0)
        else:
            final_regime = "YELLOW"
            confidence = 0.5

        return {
            "regime": final_regime,
            "confidence": confidence,
            "score": regime_score,
            "signals": regime_signals,
            "recommendation": self._get_regime_recommendation(final_regime)
        }

    def _get_regime_recommendation(self, regime: str) -> str:
        """Рекомендацandї по режиму"""
        recommendations = {
            "GREEN": "Тренд чandткий, Multi-TF alignment працює. Можна активно торгувати.",
            "YELLOW": "Роwithбandжнandсть мandж новинами and технandкою. Обережнandсть, withменшити поwithицandї.",
            "RED": "Висока волатильнandсть, прогноwithи notсandбandльнand. Зменшити риwithики or утриматися."
        }
        return recommendations.get(regime, "Невandдомий режим")

    def should_trade(self, regime_result: Dict, min_confidence: float = 0.6) -> bool:
        """Чи варто торгувати в поточному режимand"""
        if regime_result["regime"] == "RED":
            return False
        if regime_result["regime"] == "YELLOW" and regime_result["confidence"] < min_confidence:
            return False
        return True