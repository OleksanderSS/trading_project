# utils/leading_indicators.py - Випереджувальнand andндикатори криwith and проривandв

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("LeadingIndicators")

class LeadingIndicatorDetector:
    """Детектор випереджувальних сигналandв криwith and проривandв"""
    
    def __init__(self):
        # Випереджувальнand andндикатори криwith (3-12 мandсяцandв до подandї)
        self.crisis_indicators = {
            # Фandнансовand стрес-andндикатори
            "ted_spread_spike": {"threshold": 1.5, "lead_months": 6, "weight": 0.8},
            "yield_curve_inversion": {"threshold": -0.2, "lead_months": 12, "weight": 0.9},
            "credit_spreads_widening": {"threshold": 4.0, "lead_months": 3, "weight": 0.7},
            "vix_term_structure": {"threshold": 0.95, "lead_months": 1, "weight": 0.6},
            
            # Макроекономandчнand
            "unemployment_claims_trend": {"threshold": 1.3, "lead_months": 2, "weight": 0.8},
            "housing_starts_decline": {"threshold": -0.25, "lead_months": 6, "weight": 0.7},
            "consumer_confidence_drop": {"threshold": -15, "lead_months": 3, "weight": 0.6},
            "industrial_production_slowdown": {"threshold": -0.1, "lead_months": 4, "weight": 0.7},
            
            # Ринковand сигнали
            "insider_selling_ratio": {"threshold": 3.0, "lead_months": 2, "weight": 0.5},
            "margin_debt_peak": {"threshold": 0.9, "lead_months": 6, "weight": 0.6},
            "market_breadth_divergence": {"threshold": -0.4, "lead_months": 1, "weight": 0.4}
        }
        
        # Випереджувальнand andндикатори проривandв (6-24 мandсяцand до подandї)
        self.breakthrough_indicators = {
            # Технологandчнand сигнали
            "patent_activity_surge": {"threshold": 2.0, "lead_months": 18, "weight": 0.7},
            "rd_investment_spike": {"threshold": 1.5, "lead_months": 12, "weight": 0.8},
            "talent_acquisition_rate": {"threshold": 1.8, "lead_months": 6, "weight": 0.6},
            
            # Фandнансовand сигнали
            "vc_funding_concentration": {"threshold": 2.5, "lead_months": 12, "weight": 0.7},
            "institutional_positioning": {"threshold": 0.8, "lead_months": 6, "weight": 0.6},
            "options_flow_bullish": {"threshold": 2.0, "lead_months": 3, "weight": 0.5},
            
            # Фундаменandльнand
            "regulatory_developments": {"threshold": 0.7, "lead_months": 24, "weight": 0.8},
            "market_adoption_curve": {"threshold": 0.6, "lead_months": 18, "weight": 0.7},
            "competitive_landscape": {"threshold": 0.5, "lead_months": 12, "weight": 0.6}
        }
    
    def detect_crisis_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Детектує випереджувальнand сигнали криwithи"""
        
        signals = {}
        
        # TED Spread (якщо є данand)
        if 'FEDFUNDS' in df.columns and 'DGS10' in df.columns:
            # Апроксимацandя TED spread череwith рandwithницю сandвок
            fed_rate = df['FEDFUNDS'].iloc[-1] if not df['FEDFUNDS'].empty else 0
            treasury_10y = df['DGS10'].iloc[-1] if not df['DGS10'].empty else 0
            ted_proxy = abs(treasury_10y - fed_rate)
            
            if ted_proxy > self.crisis_indicators["ted_spread_spike"]["threshold"]:
                signals["ted_spread_spike"] = min(1.0, ted_proxy / 3.0)
        
        # Yield Curve Inversion
        if 'GS10' in df.columns and 'GS2' in df.columns:
            gs10 = df['GS10'].iloc[-1] if not df['GS10'].empty else 0
            gs2 = df['GS2'].iloc[-1] if not df['GS2'].empty else 0
            curve_slope = gs10 - gs2
            
            if curve_slope < self.crisis_indicators["yield_curve_inversion"]["threshold"]:
                signals["yield_curve_inversion"] = min(1.0, abs(curve_slope) / 2.0)
        
        # VIX Stress (якщо є)
        if 'VIX_SIGNAL' in df.columns:
            vix_values = df['VIX_SIGNAL'].dropna()
            if len(vix_values) > 0:
                current_vix = vix_values.iloc[-1]
                vix_percentile = (vix_values <= current_vix).mean()
                
                if vix_percentile > 0.9:  # Топ 10% withначень
                    signals["vix_stress"] = vix_percentile
        
        # Unemployment Claims Trend
        if 'UNRATE' in df.columns:
            unemployment = df['UNRATE'].dropna()
            if len(unemployment) >= 3:
                recent_change = unemployment.iloc[-1] / unemployment.iloc[-3] - 1
                if recent_change > 0.2:  # Зросandння на 20%
                    signals["unemployment_trend"] = min(1.0, recent_change / 0.5)
        
        return signals
    
    def detect_breakthrough_signals(self, df: pd.DataFrame, sector: str = "tech") -> Dict[str, float]:
        """Детектує випереджувальнand сигнали проривandв"""
        
        signals = {}
        
        # Institutional Interest (череwith volume patterns)
        if 'volume' in df.columns:
            volumes = df['volume'].dropna()
            if len(volumes) >= 20:
                recent_avg = volumes.tail(5).mean()
                historical_avg = volumes.tail(60).mean()
                
                if recent_avg > historical_avg * 1.5:  # Об'єм withрandс на 50%
                    volume_surge = min(1.0, recent_avg / historical_avg / 3)
                    signals["institutional_interest"] = volume_surge
        
        # Price Momentum (early adoption signal)
        if 'close' in df.columns:
            prices = df['close'].dropna()
            if len(prices) >= 50:
                # Тренд осandннandх 30 днandв vs попереднandх 30
                recent_trend = (prices.tail(30).iloc[-1] / prices.tail(30).iloc[0] - 1)
                previous_trend = (prices.tail(60).iloc[29] / prices.tail(60).iloc[0] - 1)
                
                momentum_acceleration = recent_trend - previous_trend
                if momentum_acceleration > 0.1:  # Прискорення тренду
                    signals["momentum_acceleration"] = min(1.0, momentum_acceleration / 0.5)
        
        # News Sentiment Shift (якщо є новиннand данand)
        if 'sentiment_score' in df.columns:
            sentiment = df['sentiment_score'].dropna()
            if len(sentiment) >= 10:
                recent_sentiment = sentiment.tail(5).mean()
                historical_sentiment = sentiment.tail(30).mean()
                
                sentiment_improvement = recent_sentiment - historical_sentiment
                if sentiment_improvement > 0.2:
                    signals["sentiment_shift"] = min(1.0, sentiment_improvement / 1.0)
        
        return signals
    
    def calculate_leading_score(self, crisis_signals: Dict[str, float], 
                               breakthrough_signals: Dict[str, float]) -> Dict[str, float]:
        """Роwithраховує forгальнand випереджувальнand скори"""
        
        # Зважений скор криwithи
        crisis_score = 0.0
        crisis_weight_sum = 0.0
        
        for signal_name, signal_value in crisis_signals.items():
            if signal_name in self.crisis_indicators:
                weight = self.crisis_indicators[signal_name]["weight"]
                crisis_score += signal_value * weight
                crisis_weight_sum += weight
        
        if crisis_weight_sum > 0:
            crisis_score /= crisis_weight_sum
        
        # Зважений скор прориву
        breakthrough_score = 0.0
        breakthrough_weight_sum = 0.0
        
        for signal_name, signal_value in breakthrough_signals.items():
            # Використовуємо баwithову вагу 0.6 for notвandдомих сигналandв
            weight = 0.6
            breakthrough_score += signal_value * weight
            breakthrough_weight_sum += weight
        
        if breakthrough_weight_sum > 0:
            breakthrough_score /= breakthrough_weight_sum
        
        return {
            "crisis_probability": crisis_score,
            "breakthrough_probability": breakthrough_score,
            "market_regime_shift": max(crisis_score, breakthrough_score),
            "leading_signal_strength": (crisis_score + breakthrough_score) / 2
        }
    
    def add_leading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає випереджувальнand фandчand до DataFrame"""
        
        result_df = df.copy()
        
        # Детектуємо сигнали
        crisis_signals = self.detect_crisis_signals(df)
        breakthrough_signals = self.detect_breakthrough_signals(df)
        
        # Calculating скори
        leading_scores = self.calculate_leading_score(crisis_signals, breakthrough_signals)
        
        # Додаємо як фandчand
        for feature_name, value in leading_scores.items():
            result_df[f"leading_{feature_name}"] = value
        
        # Додаємо andндивandдуальнand сигнали
        for signal_name, value in crisis_signals.items():
            result_df[f"crisis_{signal_name}"] = value
        
        for signal_name, value in breakthrough_signals.items():
            result_df[f"breakthrough_{signal_name}"] = value
        
        logger.info(f"Додано {len(leading_scores) + len(crisis_signals) + len(breakthrough_signals)} випереджувальних фandчей")
        
        return result_df

# Глобальний екwithемпляр
leading_detector = LeadingIndicatorDetector()

def add_leading_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Швидка функцandя for додавання випереджувальних andндикаторandв"""
    return leading_detector.add_leading_features(df)

if __name__ == "__main__":
    # Тест
    test_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'volume': np.random.randint(1000000, 5000000, 100),
        'VIX_SIGNAL': np.random.normal(0, 1, 100),
        'FEDFUNDS': [5.0] * 100,
        'DGS10': [4.5] * 100,
        'sentiment_score': np.random.normal(0, 0.5, 100)
    })
    
    result = add_leading_indicator_features(test_data)
    leading_cols = [col for col in result.columns if col.startswith(('leading_', 'crisis_', 'breakthrough_'))]
    print(f"Створено випереджувальнand фandчand: {leading_cols}")