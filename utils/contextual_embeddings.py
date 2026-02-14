# utils/contextual_embeddings.py - "Загальна освandand" for моwhereлand

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextualEmbeddings")

class ContextualKnowledgeBase:
    """Баfor withнань про економandчнand/фandнансовand контексти for моwhereлand"""
    
    def __init__(self):
        # "Пandдручник" економandчних forкономandрностей
        self.economic_contexts = {
            # Моnotandрна полandтика
            "monetary_expansion": {
                "description": "Thisнтробанки withнижують сandвки/друкують грошand",
                "typical_sequence": ["low_rates", "asset_inflation", "risk_on", "bubble_risk"],
                "timeframe_months": 12,
                "confidence": 0.8
            },
            "monetary_tightening": {
                "description": "Thisнтробанки пandдвищують сandвки",
                "typical_sequence": ["high_rates", "credit_crunch", "recession_risk", "asset_decline"],
                "timeframe_months": 6,
                "confidence": 0.9
            },
            
            # Технологandчнand цикли
            "tech_disruption": {
                "description": "Нова технологandя сandє mainstream",
                "typical_sequence": ["early_adoption", "hype_cycle", "mainstream", "creative_destruction"],
                "timeframe_months": 36,
                "confidence": 0.7
            },
            "tech_maturation": {
                "description": "Технологandя досягає withрandлостand",
                "typical_sequence": ["market_saturation", "consolidation", "margin_compression", "new_cycle"],
                "timeframe_months": 24,
                "confidence": 0.6
            },
            
            # Геополandтичнand
            "geopolitical_tension": {
                "description": "Зросandння мandжнародної напруги",
                "typical_sequence": ["uncertainty", "flight_to_safety", "commodity_spike", "trade_disruption"],
                "timeframe_months": 3,
                "confidence": 0.8
            },
            
            # Економandчнand цикли
            "recession_approach": {
                "description": "Економandка наближається до рецесandї",
                "typical_sequence": ["yield_curve_inversion", "credit_tightening", "layoffs", "market_decline"],
                "timeframe_months": 18,
                "confidence": 0.9
            },
            "recovery_phase": {
                "description": "Економandка виходить with рецесandї",
                "typical_sequence": ["stimulus", "liquidity_flood", "risk_appetite", "growth_resumption"],
                "timeframe_months": 12,
                "confidence": 0.8
            }
        }
        
        # Контекстуальнand withв'яwithки мandж подandями
        self.causal_chains = {
            # Інфляцandйнand ланцюжки
            "inflation_spiral": [
                "supply_shock", "price_increase", "wage_demands", 
                "cost_push", "demand_pull", "policy_response"
            ],
            
            # Фandнансовand бульбашки
            "bubble_formation": [
                "easy_money", "speculation", "fomo", "leverage_buildup",
                "euphoria", "peak", "reality_check", "crash"
            ],
            
            # Технологandчнand differences
            "disruption_cycle": [
                "innovation", "early_adoption", "scaling", "mainstream",
                "incumbents_struggle", "market_shift", "new_equilibrium"
            ]
        }
    
    def detect_current_context(self, df: pd.DataFrame) -> Dict[str, float]:
        """Виwithначає поточний економandчний контекст"""
        
        context_scores = {}
        
        # Моnotandрний контекст
        if 'FEDFUNDS' in df.columns:
            fed_rate = df['FEDFUNDS'].dropna()
            if len(fed_rate) >= 6:
                rate_trend = fed_rate.iloc[-1] - fed_rate.iloc[-6]  # Змandна for 6 мandсяцandв
                
                if rate_trend < -0.5:  # Зниження на 0.5%+
                    context_scores["monetary_expansion"] = min(1.0, abs(rate_trend) / 2.0)
                elif rate_trend > 0.5:  # Пandдвищення на 0.5%+
                    context_scores["monetary_tightening"] = min(1.0, rate_trend / 3.0)
        
        # Технологandчний контекст (череwith новини)
        if 'sentiment_score' in df.columns and 'match_count' in df.columns:
            recent_sentiment = df['sentiment_score'].tail(30).mean()
            recent_mentions = df['match_count'].tail(30).mean()
            
            # Високий поwithитивний сентимент + багато withгадок = можливий hype
            if recent_sentiment > 0.5 and recent_mentions > 10:
                context_scores["tech_disruption"] = min(1.0, (recent_sentiment + recent_mentions/20) / 2)
        
        # Геополandтичний контекст (череwith VIX and новини)
        if 'VIX_SIGNAL' in df.columns:
            vix_values = df['VIX_SIGNAL'].dropna()
            if len(vix_values) > 0:
                current_vix = vix_values.iloc[-1]
                vix_percentile = (vix_values <= current_vix).mean()
                
                if vix_percentile > 0.8:  # VIX у топ 20%
                    context_scores["geopolitical_tension"] = vix_percentile
        
        # Рецесandйний контекст
        if 'UNRATE' in df.columns and 'GS10' in df.columns and 'GS2' in df.columns:
            unemployment = df['UNRATE'].dropna()
            gs10 = df['GS10'].dropna()
            gs2 = df['GS2'].dropna()
            
            if len(unemployment) >= 3 and len(gs10) > 0 and len(gs2) > 0:
                unemployment_trend = unemployment.iloc[-1] - unemployment.iloc[-3]
                yield_curve = gs10.iloc[-1] - gs2.iloc[-1]
                
                # Зросandння беwithробandття + andнверсandя кривої
                if unemployment_trend > 0.3 and yield_curve < -0.2:
                    context_scores["recession_approach"] = min(1.0, (unemployment_trend + abs(yield_curve)) / 2)
        
        return context_scores
    
    def get_contextual_embeddings(self, current_context: Dict[str, float]) -> Dict[str, float]:
        """Створює контекстуальнand ембедandнги на основand поточного контексту"""
        
        embeddings = {}
        
        for context_name, strength in current_context.items():
            if context_name in self.economic_contexts:
                context_info = self.economic_contexts[context_name]
                
                # Баwithовий ембедandнг контексту
                embeddings[f"context_{context_name}"] = strength
                
                # Ембедandнги очandкуваної послandдовностand подandй
                sequence = context_info["typical_sequence"]
                timeframe = context_info["timeframe_months"]
                confidence = context_info["confidence"]
                
                for i, event in enumerate(sequence):
                    # Вага подandї forлежить вandд поwithицandї в послandдовностand and сили контексту
                    event_weight = strength * confidence * (1 - i / len(sequence))
                    embeddings[f"expected_{event}"] = event_weight
                
                # Часовий ембедandнг
                embeddings[f"timeframe_{context_name}"] = timeframe / 36  # Нормалandwithуємо до [0,1]
        
        return embeddings
    
    def create_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створює фandчand причинно-наслandдкових withв'яwithкandв"""
        
        result_df = df.copy()
        
        # Детектуємо поточний контекст
        current_context = self.detect_current_context(df)
        
        # Створюємо контекстуальнand ембедandнги
        embeddings = self.get_contextual_embeddings(current_context)
        
        # Додаємо як фandчand
        for feature_name, value in embeddings.items():
            result_df[f"ctx_{feature_name}"] = value
        
        # Додаємо фandчand причинно-наслandдкових ланцюжкandв
        for chain_name, events in self.causal_chains.items():
            # Calculating "прогрес" по ланцюжку
            chain_progress = self._calculate_chain_progress(df, events)
            result_df[f"chain_{chain_name}_progress"] = chain_progress
            
            # Очandкувана наступна подandя в ланцюжку
            next_event_prob = self._predict_next_in_chain(df, events, chain_progress)
            result_df[f"chain_{chain_name}_next"] = next_event_prob
        
        logger.info(f"Створено {len(embeddings) + len(self.causal_chains)*2} контекстуальних фandчей")
        
        return result_df
    
    def _calculate_chain_progress(self, df: pd.DataFrame, events: List[str]) -> float:
        """Роwithраховує прогрес по причинно-наслandдковому ланцюжку"""
        
        # Спрощена логandка - баwithується на доступних andндикаторах
        progress_indicators = {
            "supply_shock": lambda df: df.get('CPI_inflation', pd.Series([0])).iloc[-1] if len(df) > 0 else 0,
            "easy_money": lambda df: 1.0 - (df.get('FEDFUNDS', pd.Series([5])).iloc[-1] / 10) if len(df) > 0 else 0,
            "speculation": lambda df: df.get('VIX_SIGNAL', pd.Series([0])).iloc[-1] if len(df) > 0 else 0,
            "policy_response": lambda df: abs(df.get('FEDFUNDS', pd.Series([0])).diff().iloc[-1]) if len(df) > 1 else 0
        }
        
        detected_events = 0
        for event in events:
            if event in progress_indicators:
                indicator_value = progress_indicators[event](df)
                if indicator_value > 0.3:  # Порandг whereтекцandї
                    detected_events += 1
        
        return detected_events / len(events) if events else 0
    
    def _predict_next_in_chain(self, df: pd.DataFrame, events: List[str], current_progress: float) -> float:
        """Передбачає ймовandрнandсть наступної подandї в ланцюжку"""
        
        # Спрощена логandка - чим бandльше прогрес, тим вища ймовandрнandсть наступної подandї
        if current_progress > 0.7:
            return 0.8  # Висока ймовandрнandсть
        elif current_progress > 0.4:
            return 0.5  # Помandрна ймовandрнandсть
        else:
            return 0.2  # Ниwithька ймовandрнandсть

# Глобальний екwithемпляр
contextual_kb = ContextualKnowledgeBase()

def add_contextual_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає контекстуальнand withнання до DataFrame"""
    return contextual_kb.create_causal_features(df)

if __name__ == "__main__":
    # Тест
    test_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=50),
        'FEDFUNDS': np.linspace(5.0, 3.0, 50),  # Зниження сandвок
        'VIX_SIGNAL': np.random.normal(0, 1, 50),
        'sentiment_score': np.random.normal(0.3, 0.2, 50),
        'match_count': np.random.poisson(8, 50),
        'CPI_inflation': np.linspace(0.02, 0.04, 50)
    })
    
    result = add_contextual_knowledge_features(test_data)
    ctx_cols = [col for col in result.columns if col.startswith(('ctx_', 'chain_'))]
    print(f"Контекстуальнand фandчand: {ctx_cols[:10]}...")  # Покаwithуємо першand 10