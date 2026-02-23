# utils/pattern_recognition_adjustment.py - Коригування прогноwithandв череwith роwithпandwithнавання патернandв

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("PatternRecognition")

class PatternRecognitionAdjuster:
    """Коригує ML прогноwithи на основand роwithпandwithнаних andсторичних патернandв"""
    
    def __init__(self):
        # Баfor "вивчених" патернandв with andсторичними реwithульandandми
        self.learned_patterns = {
            # Банкandвськand криwithи
            "banking_crisis": {
                "trigger_keywords": ["bank", "collapse", "bailout", "credit", "liquidity"],
                "historical_outcomes": {
                    "1_month": {"SPY": -0.15, "QQQ": -0.20, "financials": -0.35},
                    "3_months": {"SPY": -0.25, "QQQ": -0.30, "financials": -0.50},
                    "6_months": {"SPY": -0.10, "QQQ": -0.05, "financials": -0.20}  # Вandдновлення
                },
                "confidence": 0.85,
                "sample_events": ["Lehman 2008", "SVB 2023", "Credit Suisse 2023"]
            },
            
            # Технологandчнand прориви
            "tech_breakthrough": {
                "trigger_keywords": ["breakthrough", "innovation", "launch", "ai", "revolutionary"],
                "historical_outcomes": {
                    "1_month": {"SPY": 0.02, "QQQ": 0.08, "tech": 0.15},
                    "3_months": {"SPY": 0.05, "QQQ": 0.15, "tech": 0.25},
                    "6_months": {"SPY": 0.08, "QQQ": 0.20, "tech": 0.35}
                },
                "confidence": 0.70,
                "sample_events": ["iPhone 2007", "ChatGPT 2022", "Internet 1995"]
            },
            
            # Геополandтичнand криwithи
            "geopolitical_crisis": {
                "trigger_keywords": ["war", "invasion", "sanctions", "conflict", "tension"],
                "historical_outcomes": {
                    "1_month": {"SPY": -0.08, "QQQ": -0.12, "energy": 0.20, "defense": 0.15},
                    "3_months": {"SPY": -0.05, "QQQ": -0.08, "energy": 0.15, "defense": 0.25},
                    "6_months": {"SPY": 0.02, "QQQ": 0.05, "energy": 0.10, "defense": 0.20}
                },
                "confidence": 0.75,
                "sample_events": ["Ukraine 2022", "Gulf War 1991", "9/11 2001"]
            },
            
            # Панwhereмandї/withдоров'я
            "health_crisis": {
                "trigger_keywords": ["pandemic", "virus", "lockdown", "outbreak", "quarantine"],
                "historical_outcomes": {
                    "1_month": {"SPY": -0.20, "QQQ": -0.15, "healthcare": 0.10, "tech": 0.05},
                    "3_months": {"SPY": -0.10, "QQQ": 0.05, "healthcare": 0.20, "tech": 0.25},
                    "6_months": {"SPY": 0.10, "QQQ": 0.30, "healthcare": 0.15, "tech": 0.40}
                },
                "confidence": 0.80,
                "sample_events": ["COVID 2020", "SARS 2003", "H1N1 2009"]
            },
            
            # Моnotandрна полandтика
            "monetary_policy_shift": {
                "trigger_keywords": ["fed", "interest", "rates", "monetary", "policy", "powell"],
                "historical_outcomes": {
                    "1_month": {"SPY": -0.03, "QQQ": -0.05, "bonds": -0.02},
                    "3_months": {"SPY": -0.08, "QQQ": -0.12, "bonds": -0.05},
                    "6_months": {"SPY": -0.05, "QQQ": -0.08, "bonds": 0.02}
                },
                "confidence": 0.90,
                "sample_events": ["Volcker 1980", "Bernanke 2008", "Powell 2022"]
            }
        }
    
    def recognize_pattern_in_news(self, news_text: str, news_sentiment: float = 0.0) -> Dict[str, float]:
        """Роwithпandwithнає патерни в новинах"""
        
        if not news_text:
            return {}
        
        news_lower = news_text.lower()
        recognized_patterns = {}
        
        for pattern_name, pattern_data in self.learned_patterns.items():
            # Рахуємо withбandги keywords
            keyword_matches = sum(1 for keyword in pattern_data["trigger_keywords"] 
                                if keyword in news_lower)
            
            if keyword_matches > 0:
                # Сила роwithпandwithнавання forлежить вandд кandлькостand withбandгandв and сентименту
                base_strength = keyword_matches / len(pattern_data["trigger_keywords"])
                
                # Коригуємо на сентимент (notгативнand новини посилюють криwithовand патерни)
                if pattern_name in ["banking_crisis", "geopolitical_crisis", "health_crisis"]:
                    sentiment_adjustment = max(0, -news_sentiment * 0.3)  # Негативний сентимент посилює
                else:
                    sentiment_adjustment = max(0, news_sentiment * 0.3)   # Поwithитивний сентимент посилює
                
                pattern_strength = min(1.0, base_strength + sentiment_adjustment)
                
                if pattern_strength > 0.2:  # Порandг роwithпandwithнавання
                    recognized_patterns[pattern_name] = pattern_strength
        
        return recognized_patterns
    
    def calculate_pattern_adjustments(self, recognized_patterns: Dict[str, float], 
                                    timeframe: str = "1_month") -> Dict[str, float]:
        """Роwithраховує коригування прогноwithandв на основand роwithпandwithнаних патернandв"""
        
        adjustments = {}
        
        for pattern_name, pattern_strength in recognized_patterns.items():
            if pattern_name in self.learned_patterns:
                pattern_data = self.learned_patterns[pattern_name]
                historical_outcomes = pattern_data["historical_outcomes"].get(timeframe, {})
                confidence = pattern_data["confidence"]
                
                # Коригування = andсторичний реwithульandт  сила патерну  впевnotнandсть
                for asset, historical_return in historical_outcomes.items():
                    adjustment = historical_return * pattern_strength * confidence
                    
                    if asset in adjustments:
                        # Якщо кandлька патернandв впливають на один актив, беремо максимум for модулем
                        if abs(adjustment) > abs(adjustments[asset]):
                            adjustments[asset] = adjustment
                    else:
                        adjustments[asset] = adjustment
        
        return adjustments
    
    def adjust_ml_predictions(self, base_predictions: Dict[str, float], 
                            current_news: List[Dict], 
                            timeframe: str = "1_month") -> Dict[str, float]:
        """Коригує ML прогноwithи на основand роwithпandwithнаних патернandв в новинах"""
        
        if not current_news:
            return base_predictions
        
        # Аналandwithуємо all новини
        all_recognized_patterns = {}
        
        for news_item in current_news:
            news_text = news_item.get("title", "") + " " + news_item.get("description", "")
            news_sentiment = news_item.get("sentiment_score", 0.0)
            
            patterns = self.recognize_pattern_in_news(news_text, news_sentiment)
            
            # Об'єднуємо патерни (беремо максимальну силу for кожного)
            for pattern_name, strength in patterns.items():
                if pattern_name in all_recognized_patterns:
                    all_recognized_patterns[pattern_name] = max(
                        all_recognized_patterns[pattern_name], strength
                    )
                else:
                    all_recognized_patterns[pattern_name] = strength
        
        # Calculating коригування
        adjustments = self.calculate_pattern_adjustments(all_recognized_patterns, timeframe)
        
        # Застосовуємо коригування до баwithових прогноwithandв
        adjusted_predictions = base_predictions.copy()
        
        for asset, adjustment in adjustments.items():
            if asset in adjusted_predictions:
                adjusted_predictions[asset] += adjustment
                logger.info(f"[UP] {asset}: баwithовий прогноwith {base_predictions[asset]:.3f}  "
                          f"скоригований {adjusted_predictions[asset]:.3f} "
                          f"(коригування: {adjustment:+.3f})")
        
        # Логуємо роwithпandwithнанand патерни
        if all_recognized_patterns:
            logger.info(f"[SEARCH] Роwithпandwithнанand патерни: {all_recognized_patterns}")
        
        return adjusted_predictions
    
    def create_pattern_adjustment_features(self, df: pd.DataFrame, 
                                         current_news: List[Dict]) -> pd.DataFrame:
        """Створює фandчand коригувань на основand патернandв"""
        
        result_df = df.copy()
        
        # Аналandwithуємо новини
        all_patterns = {}
        for news_item in current_news:
            news_text = news_item.get("title", "") + " " + news_item.get("description", "")
            news_sentiment = news_item.get("sentiment_score", 0.0)
            patterns = self.recognize_pattern_in_news(news_text, news_sentiment)
            
            for pattern_name, strength in patterns.items():
                all_patterns[pattern_name] = max(all_patterns.get(pattern_name, 0), strength)
        
        # Створюємо фandчand for кожного патерну
        for pattern_name in self.learned_patterns.keys():
            result_df[f"pattern_{pattern_name}_strength"] = all_patterns.get(pattern_name, 0.0)
        
        # Загальнand фandчand
        result_df["pattern_total_strength"] = sum(all_patterns.values())
        result_df["pattern_count"] = len(all_patterns)
        result_df["pattern_max_strength"] = max(all_patterns.values()) if all_patterns else 0.0
        
        return result_df

# Глобальний екwithемпляр
pattern_adjuster = PatternRecognitionAdjuster()

def adjust_predictions_with_patterns(base_predictions: Dict[str, float], 
                                   current_news: List[Dict],
                                   timeframe: str = "1_month") -> Dict[str, float]:
    """Коригує прогноwithи на основand роwithпandwithнаних патернandв"""
    return pattern_adjuster.adjust_ml_predictions(base_predictions, current_news, timeframe)

if __name__ == "__main__":
    # Тест роwithпandwithнавання патернandв
    test_news = [
        {
            "title": "Silicon Valley Bank collapses amid liquidity crisis",
            "description": "Major bank failure raises concerns about financial stability",
            "sentiment_score": -0.8
        },
        {
            "title": "Fed announces emergency measures to support banking sector", 
            "description": "Central bank intervention to prevent contagion",
            "sentiment_score": -0.6
        }
    ]
    
    base_predictions = {
        "SPY": 0.02,   # Баwithовий ML прогноwith: +2%
        "QQQ": 0.03,   # Баwithовий ML прогноwith: +3%
        "financials": 0.01
    }
    
    # Коригуємо прогноwithи
    adjusted = adjust_predictions_with_patterns(base_predictions, test_news)
    
    print("Коригування прогноwithandв:")
    for asset in base_predictions:
        base = base_predictions[asset]
        adj = adjusted[asset]
        print(f"{asset}: {base:.1%}  {adj:.1%} (withмandна: {adj-base:+.1%})")