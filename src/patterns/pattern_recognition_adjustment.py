# src/patterns/pattern_recognition_adjustment.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PatternRecognition")

class PatternRecognitionAdjuster:
    """Adjusts ML predictions based on recognized historical market and news patterns."""
    
    def __init__(self):
        # Database of "learned" patterns with historical outcomes
        self.learned_patterns = {
            # Banking Crises
            "banking_crisis": {
                "trigger_keywords": ["bank", "collapse", "bailout", "credit", "liquidity"],
                "historical_outcomes": {
                    "1_month": {"SPY": -0.15, "QQQ": -0.20, "financials": -0.35},
                    "3_months": {"SPY": -0.25, "QQQ": -0.30, "financials": -0.50},
                    "6_months": {"SPY": -0.10, "QQQ": -0.05, "financials": -0.20}  # Recovery
                },
                "confidence": 0.85,
                "sample_events": ["Lehman 2008", "SVB 2023", "Credit Suisse 2023"]
            },
            
            # Tech Breakthroughs
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
            
            # Geopolitical Crises
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
            
            # Health Crises
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
            
            # Monetary Policy
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
        """Recognizes patterns in news text based on keywords and sentiment."""
        if not news_text:
            return {}
        
        news_lower = news_text.lower()
        recognized_patterns = {}
        
        for pattern_name, pattern_data in self.learned_patterns.items():
            # Count keyword matches
            keyword_matches = sum(1 for keyword in pattern_data["trigger_keywords"] 
                                if keyword in news_lower)
            
            if keyword_matches > 0:
                # Recognition strength depends on keyword count and sentiment
                base_strength = keyword_matches / len(pattern_data["trigger_keywords"])
                
                # Adjust for sentiment (negative news amplifies crisis patterns)
                if pattern_name in ["banking_crisis", "geopolitical_crisis", "health_crisis"]:
                    sentiment_adjustment = max(0, -news_sentiment * 0.3)
                else:
                    sentiment_adjustment = max(0, news_sentiment * 0.3)
                
                pattern_strength = min(1.0, base_strength + sentiment_adjustment)
                
                if pattern_strength > 0.2:  # Threshold for recognition
                    recognized_patterns[pattern_name] = pattern_strength
        
        return recognized_patterns
    
    def calculate_pattern_adjustments(self, recognized_patterns: Dict[str, float], 
                                    timeframe: str = "1_month") -> Dict[str, float]:
        """Calculates prediction adjustments based on recognized patterns."""
        adjustments = {}
        
        for pattern_name, pattern_strength in recognized_patterns.items():
            if pattern_name in self.learned_patterns:
                pattern_data = self.learned_patterns[pattern_name]
                historical_outcomes = pattern_data["historical_outcomes"].get(timeframe, {})
                confidence = pattern_data["confidence"]
                
                # Adjustment = historical result * pattern strength * confidence
                for asset, historical_return in historical_outcomes.items():
                    adjustment = historical_return * pattern_strength * confidence
                    
                    if asset in adjustments:
                        # If multiple patterns affect one asset, take the maximum absolute adjustment
                        if abs(adjustment) > abs(adjustments[asset]):
                            adjustments[asset] = adjustment
                    else:
                        adjustments[asset] = adjustment
        
        return adjustments
    
    def adjust_ml_predictions(self, base_predictions: Dict[str, float], 
                            current_news: List[Dict], 
                            timeframe: str = "1_month") -> Dict[str, float]:
        """Adjusts base ML predictions based on news patterns."""
        if not current_news:
            return base_predictions
        
        # Analyze all news items
        all_recognized_patterns = {}
        
        for news_item in current_news:
            news_text = news_item.get("title", "") + " " + news_item.get("description", "")
            news_sentiment = news_item.get("sentiment_score", 0.0)
            
            patterns = self.recognize_pattern_in_news(news_text, news_sentiment)
            
            # Aggregate patterns (take max strength for each)
            for pattern_name, strength in patterns.items():
                if pattern_name in all_recognized_patterns:
                    all_recognized_patterns[pattern_name] = max(
                        all_recognized_patterns[pattern_name], strength
                    )
                else:
                    all_recognized_patterns[pattern_name] = strength
        
        # Calculate adjustments
        adjustments = self.calculate_pattern_adjustments(all_recognized_patterns, timeframe)
        
        # Apply adjustments to base predictions
        adjusted_predictions = base_predictions.copy()
        
        for asset, adjustment in adjustments.items():
            if asset in adjusted_predictions:
                adjusted_predictions[asset] += adjustment
                logger.info(f"Adjustment for {asset}: base {base_predictions[asset]:.3f} -> "
                          f"adjusted {adjusted_predictions[asset]:.3f} "
                          f"(delta: {adjustment:+.3f})")
        
        if all_recognized_patterns:
            logger.info(f"Recognized patterns: {all_recognized_patterns}")
        
        return adjusted_predictions
    
    def create_pattern_adjustment_features(self, df: pd.DataFrame, 
                                         current_news: List[Dict]) -> pd.DataFrame:
        """Generates adjustment feature columns based on recognized news patterns."""
        result_df = df.copy()
        
        # Analyze news
        all_patterns = {}
        for news_item in current_news:
            news_text = news_item.get("title", "") + " " + news_item.get("description", "")
            news_sentiment = news_item.get("sentiment_score", 0.0)
            patterns = self.recognize_pattern_in_news(news_text, news_sentiment)
            
            for pattern_name, strength in patterns.items():
                all_patterns[pattern_name] = max(all_patterns.get(pattern_name, 0), strength)
        
        # Create features for each pattern
        for pattern_name in self.learned_patterns.keys():
            result_df[f"pattern_{pattern_name}_strength"] = all_patterns.get(pattern_name, 0.0)
        
        # Aggregate features
        result_df["pattern_total_strength"] = sum(all_patterns.values())
        result_df["pattern_count"] = len(all_patterns)
        result_df["pattern_max_strength"] = max(all_patterns.values()) if all_patterns else 0.0
        
        return result_df

# Global instance
pattern_adjuster = PatternRecognitionAdjuster()

def adjust_predictions_with_patterns(base_predictions: Dict[str, float], 
                                   current_news: List[Dict],
                                   timeframe: str = "1_month") -> Dict[str, float]:
    """Adjusts predictions based on recognized patterns (utility function)."""
    return pattern_adjuster.adjust_ml_predictions(base_predictions, current_news, timeframe)

if __name__ == "__main__":
    # Test pattern recognition
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
        "SPY": 0.02,   # Base ML prediction: +2%
        "QQQ": 0.03,   # Base ML prediction: +3%
        "financials": 0.01
    }
    
    # Adjust predictions
    adjusted = adjust_predictions_with_patterns(base_predictions, test_news)
    
    print("Prediction Adjustments:")
    for asset in base_predictions:
        base = base_predictions[asset]
        adj = adjusted[asset]
        print(f"{asset}: {base:.1%} -> {adj:.1%} (Change: {adj-base:+.1%})")