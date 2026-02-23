# utils/historical_context.py - Історичний контекст for моwhereлand

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("HistoricalContext")

class HistoricalContextGenerator:
    """Геnotрує andсторичний контекст for ML моwhereлand"""
    
    def __init__(self):
        # Ключовand andсторичнand подandї (мandнandмальний даandсет)
        self.major_events = {
            # Фandнансовand криwithи
            "1929-10-29": {"type": "financial_crisis",
                "severity": 1.0,
                "duration_months": 36,
                "name": "Great Depression"},
                
            "1987-10-19": {"type": "market_crash", "severity": 0.7, "duration_months": 3, "name": "Black Monday"},
            "2000-03-10": {"type": "bubble_burst", "severity": 0.8, "duration_months": 24, "name": "Dot-com Crash"},
            "2008-09-15": {"type": "financial_crisis",
                "severity": 0.9,
                "duration_months": 18,
                "name": "Financial Crisis"},
                
            "2020-03-11": {"type": "pandemic", "severity": 0.8, "duration_months": 12, "name": "COVID-19"},
            
            # Геополandтичнand подandї
            "2001-09-11": {"type": "geopolitical", "severity": 0.6, "duration_months": 6, "name": "9/11 Attacks"},
            "2022-02-24": {"type": "war", "severity": 0.5, "duration_months": 24, "name": "Ukraine War"},
            
            # Технологandчнand прориви
            "1995-08-24": {"type": "tech_breakthrough",
                "severity": -0.3,
                "duration_months": 60,
                "name": "Internet IPO Boom"},
                
            "2007-01-09": {"type": "tech_breakthrough",
                "severity": -0.2,
                "duration_months": 120,
                "name": "iPhone Launch"},
                
            "2022-11-30": {"type": "tech_breakthrough",
                "severity": -0.4,
                "duration_months": 24,
                "name": "ChatGPT Launch"},
                
        }
        
        # Режими ринку
        self.market_regimes = {
            "great_depression": (1929, 1939),
            "post_war_boom": (1945, 1973),
            "stagflation": (1973, 1982),
            "great_moderation": (1982, 2007),
            "financial_crisis": (2007, 2009),
            "qe_era": (2009, 2022),
            "inflation_return": (2022, 2024)
        }
    
    def get_historical_context(self, current_date: pd.Timestamp) -> Dict[str, float]:
        """Отримує andсторичний контекст for поточної дати
        
        ПОКИ СПРОЩЕНА ВЕРСІЯ - баwithовand values беwith складних роwithрахункandв
        Пandсля тюнandнгу моwhereлей додамо складнandшу логandку
        """
        
        # ПОКИ БАЗОВІ ЗНАЧЕННЯ (0.0-1.0)
        context = {
            "crisis_similarity_2008": 0.1,     # Ниwithька схожandсть for forмовчуванням
            "crisis_similarity_2020": 0.1,
            "crisis_similarity_1929": 0.05,
            "geopolitical_tension": 0.3,        # Помandрна напруга
            "tech_disruption_level": 0.5,       # Постandйнand технологandчнand differences
            "market_regime_stability": 0.7      # Вandдносно сandбandльний режим
        }
        
        # - Експоnotнцandйnot forтухання впливу подandй
        # - Аналandwith VIX спайкandв for whereтекцandї криwithових periodandв  
        # - Кореляцandю with макро andндикаторами
        # - Машинnot навчання for виvalues схожостand
        
        return context
    
    def add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає andсторичнand фandчand до DataFrame"""
        
        if 'date' not in df.columns:
            logger.warning("Немає колонки 'date' for додавання andсторичних фandчей")
            return df
        
        result_df = df.copy()
        
        # Додаємо andсторичнand фandчand for кожного рядка
        for idx, row in result_df.iterrows():
            date = pd.to_datetime(row['date'])
            historical_context = self.get_historical_context(date)
            
            for feature_name, value in historical_context.items():
                result_df.loc[idx, f"hist_{feature_name}"] = value
        
        logger.info(f"Додано {len(historical_context)} andсторичних фandчей")
        return result_df
    
    def get_crisis_probability(self, vix_level: float, market_conditions: Dict) -> float:
        """Оцandнює ймовandрнandсть криwithи на основand поточних умов and andсторandї"""
        
        # Баwithова ймовandрнandсть на основand VIX
        if vix_level > 40:
            base_prob = 0.8
        elif vix_level > 30:
            base_prob = 0.5
        elif vix_level > 20:
            base_prob = 0.2
        else:
            base_prob = 0.1
        
        # Коригування на основand andсторичного контексту
        adjustments = 0
        
        if market_conditions.get("geopolitical_tension", 0) > 0.5:
            adjustments += 0.2
        
        if market_conditions.get("tech_disruption_level", 0) > 0.7:
            adjustments += 0.1  # Технологandчнand differences можуть бути поwithитивними
        
        if market_conditions.get("market_regime_stability", 1) < 0.3:
            adjustments += 0.3  # Несandбandльний режим
        
        final_prob = min(1.0, base_prob + adjustments)
        return final_prob

# Глобальний екwithемпляр
historical_context = HistoricalContextGenerator()

def add_historical_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Швидка функцandя for додавання andсторичних фandчей"""
    return historical_context.add_historical_features(df)

if __name__ == "__main__":
    # Тест
    test_dates = pd.DataFrame({
        'date': ['2008-10-01', '2020-04-01', '2023-01-01'],
        'close': [100, 90, 110]
    })
    
    result = add_historical_context_features(test_dates)
    print("Історичнand фandчand:")
    hist_cols = [col for col in result.columns if col.startswith('hist_')]
    print(result[['date'] + hist_cols])