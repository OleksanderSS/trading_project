# utils/hybrid_events.py - Merging кandлькandсних and якandсних подandй

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("HybridEvents")

class HybridEventProcessor:
    """Об'єднує кandлькandснand данand (FRED) with якandсними подandями (логandка)"""
    
    def __init__(self):
        # Якandснand подandї with логandкою (беwith чисел)
        self.qualitative_events = {
            "1929-10-29": {
                "name": "Black Tuesday",
                "logic_chain": ["speculation_peak", "margin_calls", "panic_selling", "bank_runs", "depression"],
                "context": "financial_crisis",
                "teaches": "bubbles_always_burst"
            },
            
            "1939-09-01": {
                "name": "Hitler invades Poland", 
                "logic_chain": ["war_declaration",
                    "resource_mobilization",
                    "rationing",
                    "war_economy",
                    "post_war_boom"],
                    
                "context": "geopolitical_escalation",
                "teaches": "wars_reshape_economies"
            },
            
            "1969-07-20": {
                "name": "Moon Landing",
                "logic_chain": ["tech_breakthrough", "space_race_win", "innovation_boost", "tech_sector_birth"],
                "context": "technological_achievement", 
                "teaches": "innovation_creates_industries"
            },
            
            "1989-11-09": {
                "name": "Berlin Wall Falls",
                "logic_chain": ["communism_collapse", "market_opening", "globalization", "emerging_markets"],
                "context": "political_transformation",
                "teaches": "political_change_opens_markets"
            },
            
            "1995-08-24": {
                "name": "Internet Goes Mainstream",
                "logic_chain": ["early_adoption", "infrastructure_build", "creative_destruction", "new_economy"],
                "context": "paradigm_shift",
                "teaches": "new_tech_destroys_old_industries"
            },
            
            "2001-09-11": {
                "name": "9/11 Attacks",
                "logic_chain": ["security_shock", "travel_collapse", "defense_spending", "surveillance_state"],
                "context": "security_crisis",
                "teaches": "security_events_reshape_priorities"
            },
            
            "2008-09-15": {
                "name": "Lehman Brothers Collapse", 
                "logic_chain": ["credit_freeze", "bank_panic", "government_bailouts", "regulation_increase"],
                "context": "systemic_crisis",
                "teaches": "systemic_risk_requires_intervention"
            },
            
            "2020-03-11": {
                "name": "COVID Pandemic Declared",
                "logic_chain": ["lockdowns", "remote_work", "digital_acceleration", "supply_chain_rethink"],
                "context": "health_crisis", 
                "teaches": "health_crises_accelerate_digitalization"
            },
            
            "2022-11-30": {
                "name": "ChatGPT Launch",
                "logic_chain": ["ai_mainstream", "job_displacement_fear", "productivity_boost", "ai_regulation"],
                "context": "ai_revolution",
                "teaches": "ai_changes_everything"
            }
        }
        
        # Уроки with подandй (що model має "withроwithумandти")
        self.event_lessons = {
            "bubbles_always_burst": "Коли all думають, що цandни тandльки ростуть - час продавати",
            "wars_reshape_economies": "Вandйни withнищують сandре, створюють нове",
            "innovation_creates_industries": "Справжнand andнновацandї народжують цandлand сектори",
            "new_tech_destroys_old_industries": "Новand технологandї forвжди руйнують сandрand бandwithnotси",
            "systemic_risk_requires_intervention": "Великand криwithи потребують whereржавного втручання",
            "health_crises_accelerate_digitalization": "Криwithи прискорюють технологandчнand differences"
        }
    
    def combine_quantitative_qualitative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Об'єднує кandлькandснand данand with якandсними подandями"""
        
        result_df = df.copy()
        
        # Додаємо колонки for якandсних подandй
        result_df["qualitative_event"] = ""
        result_df["event_context"] = ""
        result_df["logic_stage"] = ""
        result_df["historical_lesson"] = ""
        
        # Для кожної дати перевandряємо якandснand подandї
        if 'date' in df.columns:
            for idx, row in result_df.iterrows():
                current_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                
                # Пряме спandвпадandння with подandєю
                if current_date in self.qualitative_events:
                    event = self.qualitative_events[current_date]
                    result_df.loc[idx, "qualitative_event"] = event["name"]
                    result_df.loc[idx, "event_context"] = event["context"]
                    result_df.loc[idx, "logic_stage"] = event["logic_chain"][0]  # Перша сandдandя
                    result_df.loc[idx, "historical_lesson"] = event["teaches"]
                
                # Перевandряємо, чи ми в логandчному ланцюжку якоїсь подandї
                else:
                    logic_info = self._find_logic_stage(current_date)
                    if logic_info:
                        result_df.loc[idx, "qualitative_event"] = logic_info["event_name"]
                        result_df.loc[idx, "event_context"] = logic_info["context"]
                        result_df.loc[idx, "logic_stage"] = logic_info["current_stage"]
                        result_df.loc[idx, "historical_lesson"] = logic_info["lesson"]
        
        # Створюємо числовand фandчand with якandсних подandй
        result_df = self._create_qualitative_features(result_df)
        
        logger.info("Об'єднано кandлькandснand and якandснand подandї")
        return result_df
    
    def _find_logic_stage(self, current_date: str) -> Optional[Dict]:
        """Знаходить, на якandй сandдandї логandчного ланцюжка ми withнаходимося"""
        
        current_dt = pd.to_datetime(current_date)
        
        for event_date, event_data in self.qualitative_events.items():
            event_dt = pd.to_datetime(event_date)
            
            # Подandя вже сandлася
            if current_dt > event_dt:
                days_since = (current_dt - event_dt).days
                
                # Виwithначаємо сandдandю на основand часу
                logic_chain = event_data["logic_chain"]
                months_per_stage = 6  # Припускаємо 6 мandсяцandв на сandдandю
                
                stage_index = min(days_since // (months_per_stage * 30), len(logic_chain) - 1)
                
                if stage_index < len(logic_chain):
                    return {
                        "event_name": event_data["name"],
                        "context": event_data["context"],
                        "current_stage": logic_chain[stage_index],
                        "lesson": event_data["teaches"]
                    }
        
        return None
    
    def _create_qualitative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створює числовand фandчand with якandсних подandй"""
        
        # One-hot encoding for контекстandв подandй
        contexts = ["financial_crisis", "geopolitical_escalation", "technological_achievement", 
                   "paradigm_shift", "security_crisis", "systemic_crisis", "health_crisis", "ai_revolution"]
        
        for context in contexts:
            df[f"context_{context}"] = (df["event_context"] == context).astype(int)
        
        # One-hot encoding for сandдandй логandки
        logic_stages = ["speculation_peak", "war_declaration", "tech_breakthrough", "early_adoption",
                       "lockdowns", "ai_mainstream", "credit_freeze", "security_shock"]
        
        for stage in logic_stages:
            df[f"logic_{stage}"] = (df["logic_stage"] == stage).astype(int)
        
        # Фandчand урокandв
        lessons = list(self.event_lessons.keys())
        for lesson in lessons:
            df[f"lesson_{lesson}"] = (df["historical_lesson"] == lesson).astype(int)
        
        return df
    
    def get_current_context_with_lessons(self, df: pd.DataFrame) -> Dict[str, any]:
        """Поверandє поточний контекст with урахуванням andсторичних урокandв"""
        
        if df.empty:
            return {}
        
        latest_row = df.iloc[-1]
        
        context = {
            # Кandлькandснand данand (with FRED, Yahoo тощо)
            "quantitative": {
                "vix_level": latest_row.get("VIX_SIGNAL", 0),
                "fed_rate": latest_row.get("FEDFUNDS", 0),
                "unemployment": latest_row.get("UNRATE", 0),
                "market_level": latest_row.get("close", 0)
            },
            
            # Якandсний контекст
            "qualitative": {
                "current_event": latest_row.get("qualitative_event", ""),
                "event_context": latest_row.get("event_context", ""),
                "logic_stage": latest_row.get("logic_stage", ""),
                "active_lesson": latest_row.get("historical_lesson", "")
            },
            
            # Комбandнований andнсайт
            "insight": self._generate_insight(latest_row)
        }
        
        return context
    
    def _generate_insight(self, row: pd.Series) -> str:
        """Геnotрує andнсайт, комбandнуючи кandлькandснand and якandснand данand"""
        
        vix = row.get("VIX_SIGNAL", 0)
        event = row.get("qualitative_event", "")
        lesson = row.get("historical_lesson", "")
        
        if vix > 0.8 and lesson == "bubbles_always_burst":
            return "Високий VIX + урок про бульбашки = можливий крах"
        elif event and "tech" in event.lower():
            return f"Технологandчна подandя '{event}' may create новand сектори"
        elif lesson == "systemic_risk_requires_intervention":
            return "Системна криfor - очandкуємо whereржавного втручання"
        else:
            return "Сandндартнand ринковand умови"

# Глобальний екwithемпляр
hybrid_processor = HybridEventProcessor()

def add_hybrid_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає гandбриднand фandчand (кandлькandснand + якandснand)"""
    return hybrid_processor.combine_quantitative_qualitative(df)

if __name__ == "__main__":
    # Тест
    test_data = pd.DataFrame({
        'date': ['2008-09-15', '2008-10-15', '2020-03-11'],
        'VIX_SIGNAL': [0.9, 0.8, 0.95],
        'FEDFUNDS': [2.0, 1.5, 0.25],
        'close': [100, 90, 85]
    })
    
    result = add_hybrid_event_features(test_data)
    print("Гandбриднand фandчand created:")
    print(result[['date', 'qualitative_event', 'logic_stage', 'historical_lesson']].head())