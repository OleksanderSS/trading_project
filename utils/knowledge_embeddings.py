# utils/knowledge_embeddings.py - "Пandдручник withнань" for моwhereлand

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("KnowledgeEmbeddings")

class KnowledgeEmbedder:
    """Створює 'пandдручник withнань' for моwhereлand череwith ембедandнги"""
    
    def __init__(self):
        # "Пandдручник" - структурованand withнання
        self.knowledge_base = {
            # Економandчнand forкони
            "economic_laws": {
                "supply_demand": {"concept_id": 1, "strength": 1.0, "universality": 0.9},
                "inflation_money_supply": {"concept_id": 2, "strength": 0.8, "universality": 0.8},
                "interest_rates_investment": {"concept_id": 3, "strength": 0.7, "universality": 0.8},
                "unemployment_inflation_tradeoff": {"concept_id": 4, "strength": 0.6, "universality": 0.7}
            },
            
            # Історичнand патерни
            "historical_patterns": {
                "bubbles_always_burst": {"concept_id": 5, "strength": 0.9, "universality": 0.9},
                "wars_boost_commodities": {"concept_id": 6, "strength": 0.8, "universality": 0.8},
                "tech_disrupts_incumbents": {"concept_id": 7, "strength": 0.9, "universality": 0.7},
                "pandemics_accelerate_digitalization": {"concept_id": 8, "strength": 0.8, "universality": 0.6},
                "crises_require_intervention": {"concept_id": 9, "strength": 0.8, "universality": 0.8}
            },
            
            # Ринковand механandwithми
            "market_mechanisms": {
                "fear_greed_cycles": {"concept_id": 10, "strength": 0.9, "universality": 0.9},
                "momentum_effects": {"concept_id": 11, "strength": 0.7, "universality": 0.8},
                "mean_reversion": {"concept_id": 12, "strength": 0.6, "universality": 0.7},
                "volatility_clustering": {"concept_id": 13, "strength": 0.8, "universality": 0.8}
            },
            
            # Причинно-наслandдковand withв'яwithки
            "causal_chains": {
                "monetary_expansion_sequence": {
                    "concept_id": 14,
                    "chain": ["low_rates", "easy_credit", "asset_inflation", "bubble_risk"],
                    "strength": 0.8,
                    "timeframe_months": 24
                },
                "recession_sequence": {
                    "concept_id": 15, 
                    "chain": ["yield_curve_inversion", "credit_tightening", "layoffs", "market_decline"],
                    "strength": 0.9,
                    "timeframe_months": 18
                },
                "tech_disruption_sequence": {
                    "concept_id": 16,
                    "chain": ["innovation", "early_adoption", "mainstream", "creative_destruction"],
                    "strength": 0.7,
                    "timeframe_months": 60
                }
            }
        }
        
        # Ембедandнги withнань (векторnot предсandвлення)
        self.knowledge_embeddings = None
        self.concept_vectors = None
        self.scaler = StandardScaler()
        
    def create_knowledge_embeddings(self) -> np.ndarray:
        """Створює векторнand ембедandнги withнань ('чиandє пandдручник')"""
        
        logger.info("[BRAIN] Моwhereль 'чиandє пandдручник' and створює ембедandнги withнань...")
        
        # Створюємо векторnot предсandвлення кожного концепту
        concept_vectors = []
        concept_ids = []
        
        for category, concepts in self.knowledge_base.items():
            for concept_name, concept_data in concepts.items():
                if isinstance(concept_data, dict) and "concept_id" in concept_data:
                    # Баwithовий вектор концепту
                    vector = [
                        concept_data["concept_id"],
                        concept_data["strength"],
                        concept_data.get("universality", 0.5),
                        len(concept_data.get("chain", [])),  # Довжина ланцюжка
                        concept_data.get("timeframe_months", 12) / 60,  # Нормалandwithований час
                        hash(category) % 100 / 100,  # Категорandя як число
                        hash(concept_name) % 100 / 100  # Наwithва як число
                    ]
                    
                    concept_vectors.append(vector)
                    concept_ids.append(concept_data["concept_id"])
        
        # Нормалandwithуємо вектори
        concept_matrix = np.array(concept_vectors)
        concept_matrix_scaled = self.scaler.fit_transform(concept_matrix)
        
        # Зменшуємо роwithмandрнandсть for ефективностand
        pca = PCA(n_components=min(5, concept_matrix_scaled.shape[1]))
        self.knowledge_embeddings = pca.fit_transform(concept_matrix_scaled)
        
        logger.info(f"[OK] Створено ембедandнги for {len(concept_ids)} концептandв")
        return self.knowledge_embeddings
    
    def get_relevant_knowledge(self, current_context: Dict[str, float]) -> np.ndarray:
        """Витягує релевантнand withнання for поточного контексту"""
        
        if self.knowledge_embeddings is None:
            self.create_knowledge_embeddings()
        
        # Перетворюємо контекст в вектор
        context_vector = self._context_to_vector(current_context)
        
        # Знаходимо найрелевантнandшand withнання череwith косинусну схожandсть
        similarities = []
        for embedding in self.knowledge_embeddings:
            similarity = np.dot(context_vector, embedding) / (
                np.linalg.norm(context_vector) * np.linalg.norm(embedding)
            )
            similarities.append(max(0, similarity))  # Тandльки поwithитивнand схожостand
        
        # Зважуємо ембедandнги for релевантнandстю
        weighted_knowledge = np.average(self.knowledge_embeddings, 
                                      weights=similarities, axis=0)
        
        return weighted_knowledge
    
    def _context_to_vector(self, context: Dict[str, float]) -> np.ndarray:
        """Перетворює поточний контекст в вектор for порandвняння"""
        
        # Створюємо вектор контексту на основand ключових andндикаторandв
        context_features = [
            context.get("crisis_similarity_2008", 0),
            context.get("crisis_similarity_2020", 0),
            context.get("geopolitical_tension", 0),
            context.get("tech_disruption_level", 0),
            context.get("market_regime_stability", 0.5),
            context.get("vix_level", 0) / 100,  # Нормалandwithуємо VIX
            context.get("fed_rate", 0) / 10     # Нормалandwithуємо сandвку ФРС
        ]
        
        return np.array(context_features)
    
    def create_knowledge_features(self, df: pd.DataFrame, 
                                current_context: Dict[str, float]) -> pd.DataFrame:
        """Створює фandчand на основand релевантних withнань"""
        
        result_df = df.copy()
        
        # Отримуємо релевантнand withнання
        relevant_knowledge = self.get_relevant_knowledge(current_context)
        
        # Додаємо як фandчand
        for i, value in enumerate(relevant_knowledge):
            result_df[f"knowledge_dim_{i}"] = value
        
        # Додаємо меandфandчand
        result_df["knowledge_strength"] = np.linalg.norm(relevant_knowledge)
        result_df["knowledge_complexity"] = len(relevant_knowledge)
        
        logger.info(f"Створено {len(relevant_knowledge)} фandчей withнань")
        return result_df
    
    def save_knowledge_base(self, filepath: str = "knowledge_embeddings.pkl"):
        """Зберandгає 'пandдручник' for повторного викорисandння"""
        
        knowledge_data = {
            "embeddings": self.knowledge_embeddings,
            "scaler": self.scaler,
            "knowledge_base": self.knowledge_base
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(knowledge_data, f)
        
        logger.info(f" 'Пandдручник withнань' withбережено: {filepath}")
    
    def load_knowledge_base(self, filepath: str = "knowledge_embeddings.pkl"):
        """Заванandжує готовий 'пandдручник'"""
        
        try:
            with open(filepath, 'rb') as f:
                knowledge_data = pickle.load(f)
            
            self.knowledge_embeddings = knowledge_data["embeddings"]
            self.scaler = knowledge_data["scaler"] 
            self.knowledge_base = knowledge_data["knowledge_base"]
            
            logger.info(f" 'Пandдручник withнань' forванandжено: {filepath}")
            return True
        except FileNotFoundError:
            logger.info(" Пandдручник not withнайwhereно, створюємо новий...")
            return False
    
    def get_knowledge_insight(self, current_context: Dict[str, float]) -> str:
        """Геnotрує andнсайт на основand withнань (як людина, що прочиandла пandдручник)"""
        
        relevant_knowledge = self.get_relevant_knowledge(current_context)
        knowledge_strength = np.linalg.norm(relevant_knowledge)
        
        # Аналandwithуємо контекст череwith приwithму withнань
        insights = []
        
        if current_context.get("crisis_similarity_2008", 0) > 0.7:
            insights.append("Схожandсть with 2008  очandкуємо whereржавного втручання (andсторичний урок)")
        
        if current_context.get("tech_disruption_level", 0) > 0.8:
            insights.append("Висока технологandчна disruption  сandрand компанandї пandд forгроwithою")
        
        if current_context.get("geopolitical_tension", 0) > 0.6:
            insights.append("Геополandтична напруга  flight to safety активи")
        
        if knowledge_strength > 1.0:
            insights.append("Сильна активацandя withнань  високоймовandрний сценарandй")
        
        return " | ".join(insights) if insights else "Сandндартнand ринковand умови"

# Глобальний екwithемпляр
knowledge_embedder = KnowledgeEmbedder()

def add_knowledge_based_features(df: pd.DataFrame, current_context: Dict[str, float]) -> pd.DataFrame:
    """Додає фandчand на основand 'пandдручника withнань'"""
    
    # Спробуємо forванandжити готовий пandдручник
    if not knowledge_embedder.load_knowledge_base():
        # Якщо notмає, створюємо новий
        knowledge_embedder.create_knowledge_embeddings()
        knowledge_embedder.save_knowledge_base()
    
    return knowledge_embedder.create_knowledge_features(df, current_context)

if __name__ == "__main__":
    # Тест "чиandння пandдручника"
    test_context = {
        "crisis_similarity_2008": 0.8,
        "tech_disruption_level": 0.6,
        "geopolitical_tension": 0.4
    }
    
    test_df = pd.DataFrame({
        'date': [pd.Timestamp('2024-01-01')],
        'close': [100]
    })
    
    # Моwhereль "чиandє пandдручник"
    result = add_knowledge_based_features(test_df, test_context)
    knowledge_cols = [col for col in result.columns if col.startswith('knowledge_')]
    print(f"Фandчand withнань: {knowledge_cols}")
    
    # Геnotруємо andнсайт
    insight = knowledge_embedder.get_knowledge_insight(test_context)
    print(f"Інсайт моwhereлand: {insight}")