# core/stages/stage_3_linguistic_dna.py - Лandнгвandстичний аналandwith for виявлення "комбandнацandй, що мають вплив"

import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class LinguisticDNAAnalyzer:
    """Аналandforтор лandнгвandстичних патернandв for виявлення впливових новин"""
    
    def __init__(self, data_path: str = "data/linguistic_dna"):
        self.data_path = Path(data_path)
        self.historical_movers = self._load_historical_movers()
        self.commitment_patterns = self._build_commitment_patterns()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def _load_historical_movers(self) -> Dict:
        """Заванandжує andсторичнand 'бомби'"""
        try:
            with open(self.data_path / "historical_market_movers.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"[LinguisticDNA] Loaded {len(data['historical_market_movers'])} historical market movers")
            return data
        except Exception as e:
            logger.error(f"[LinguisticDNA] Could not load historical movers: {e}")
            return {"historical_market_movers": [], "linguistic_patterns": {"high_impact": [], "low_impact": []}}
    
    def _build_commitment_patterns(self) -> Dict[str, re.Pattern]:
        """Будує regex патерни for commitment_flag"""
        patterns = {
            'tariff_commitment': re.compile(r'(tariffs?|import\s+tax|customs?\s+tax|dut(y|ies))\s+(of\s+)?(\d+\.?\d*%|\$\d+\.?\d*[bB]?)', re.IGNORECASE),
            'rate_commitment': re.compile(r'(interest\s+rates?|fed\s+funds?)\s+(raise|increase|hike|cut|lower|reduce)\s+(by\s+)?(\d+\.?\d*%|\d+\.?\d*\s+(basis\s+)?points?)', re.IGNORECASE),
            'stock_action': re.compile(r'(stock\s+split|reverse\s+split|buyback|share\s+repurchase|dividend)\s+(of\s+)?(\d+\.?\d*\s*for\s*\d+|\d+\.?\d*%|\$\d+\.?\d*)', re.IGNORECASE),
            'government_action': re.compile(r'(shut\s+down|close|suspend)\s+(the\s+)?(government|border|trade)', re.IGNORECASE),
            'quantitative': re.compile(r'\$(\d+\.?\d*)\s*([tT]illion|[bB]illion|[mM]illion|[tT]rillion|[bB]|[mM]|[tT])', re.IGNORECASE)
        }
        return patterns
    
    def calculate_historical_impact_score(self, titles: List[str]) -> List[float]:
        """
        Feature: historical_impact_score
        Порandвнює поточнand forголовки with andсторичними "бомбами"
        """
        if not self.historical_movers['historical_market_movers']:
            return [0.0] * len(titles)
        
        # Отримуємо forголовки andсторичних "бомб"
        bomb_titles = [mover['title'] for mover in self.historical_movers['historical_market_movers']]
        
        # Додаємо поточнand forголовки до andсторичних for TF-IDF
        all_titles = bomb_titles + titles
        
        # Створюємо TF-IDF матрицю
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_titles)
            
            # Рахуємо схожandсть поточних forголовкandв with andсторичними "бомбами"
            bomb_vectors = tfidf_matrix[:len(bomb_titles)]
            current_vectors = tfidf_matrix[len(bomb_titles):]
            
            similarity_scores = []
            for current_vec in current_vectors:
                # Максимальна схожandсть with будь-якою "бомбою"
                similarities = cosine_similarity(current_vec, bomb_vectors)[0]
                max_similarity = np.max(similarities)
                similarity_scores.append(max_similarity)
            
            logger.info(f"[LinguisticDNA] Calculated historical impact scores for {len(titles)} titles")
            return similarity_scores
            
        except Exception as e:
            logger.error(f"[LinguisticDNA] Error calculating historical impact: {e}")
            return [0.0] * len(titles)
    
    def calculate_commitment_flag(self, titles: List[str]) -> List[int]:
        """
        Feature: commitment_flag
        Regex for пошуку конструкцandй "I will + [дandєслово] + [цифра]"
        Трамп "Киandй" = 0, "Миand + 25%" = 1
        """
        commitment_scores = []
        
        for title in titles:
            score = 0
            title_lower = title.lower()
            
            # Перевandряємо кожен патерн
            for pattern_name, pattern in self.commitment_patterns.items():
                if pattern.search(title):
                    # Рandwithнand ваги for рandwithних типandв комandтментandв
                    weights = {
                        'tariff_commitment': 1.0,
                        'rate_commitment': 0.9,
                        'stock_action': 0.8,
                        'government_action': 0.7,
                        'quantitative': 0.6
                    }
                    score = max(score, weights.get(pattern_name, 0.5))
            
            commitment_scores.append(int(score > 0))
        
        logger.info(f"[LinguisticDNA] Calculated commitment flags: {sum(commitment_scores)}/{len(titles)} positive")
        return commitment_scores
    
    def calculate_distance_to_gold_standard(self, titles: List[str]) -> List[float]:
        """
        Feature: dist_to_gold_standard
        Вandдсandнь вandд поточної новини до найближчої "бомби"
        """
        if not self.historical_movers['historical_market_movers']:
            return [1.0] * len(titles)  # Максимальна вandдсandнь якщо notмає andсторandї
        
        # Отримуємо forголовки andсторичних "бомб" with them вагами (impact_pct)
        bomb_titles = []
        bomb_weights = []
        
        for mover in self.historical_movers['historical_market_movers']:
            bomb_titles.append(mover['title'])
            # Нормалandwithуємо impact до ваги [0, 1]
            bomb_weights.append(min(abs(mover['impact_pct']) / 5.0, 1.0))
        
        # Додаємо поточнand forголовки
        all_titles = bomb_titles + titles
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_titles)
            
            bomb_vectors = tfidf_matrix[:len(bomb_titles)]
            current_vectors = tfidf_matrix[len(bomb_titles):]
            
            distances = []
            for current_vec in current_vectors:
                # Вагована косинусна вandдсandнь до найближчої "бомби"
                similarities = cosine_similarity(current_vec, bomb_vectors)[0]
                
                # Вагована схожandсть (бandльша вага for бandльших impact)
                weighted_similarity = np.max(similarities * np.array(bomb_weights))
                
                # Перетворюємо схожandсть в вandдсandнь
                distance = 1.0 - weighted_similarity
                distances.append(distance)
            
            logger.info(f"[LinguisticDNA] Calculated distances to gold standard for {len(titles)} titles")
            return distances
            
        except Exception as e:
            logger.error(f"[LinguisticDNA] Error calculating distances: {e}")
            return [1.0] * len(titles)
    
    def detect_semantic_duplicates(self, titles: List[str], threshold: float = 0.95) -> List[bool]:
        """
        Deduplication 2.0: Виявляє семантичнand дублandкати
        """
        if len(titles) < 2:
            return [False] * len(titles)
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(titles)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Знаходимо дублandкати (виключаючи дandагональ)
            duplicate_mask = np.zeros(len(titles), dtype=bool)
            
            for i in range(len(titles)):
                for j in range(i + 1, len(titles)):
                    if similarity_matrix[i, j] > threshold:
                        duplicate_mask[j] = True  # Другий вважаємо дублandкатом
            
            duplicates_found = duplicate_mask.sum()
            logger.info(f"[LinguisticDNA] Found {duplicates_found} semantic duplicates out of {len(titles)}")
            return duplicate_mask.tolist()
            
        except Exception as e:
            logger.error(f"[LinguisticDNA] Error detecting duplicates: {e}")
            return [False] * len(titles)
    
    def add_linguistic_features(self, df: pd.DataFrame, title_col: str = 'title') -> pd.DataFrame:
        """
        Додає all лandнгвandстичнand фandчand до DataFrame
        """
        logger.info(f"[LinguisticDNA] Adding linguistic features to {len(df)} records")
        
        # ВИПРАВЛЕНО: перевіряємо чи існує колонка
        if df.empty or title_col not in df.columns:
            logger.warning(f"[LinguisticDNA] DataFrame empty or missing '{title_col}' column, skipping linguistic features")
            return df
        
        # Отримуємо forголовки
        titles = df[title_col].fillna('').tolist()
        
        # Calculating all фandчand
        df['historical_impact_score'] = self.calculate_historical_impact_score(titles)
        df['commitment_flag'] = self.calculate_commitment_flag(titles)
        df['dist_to_gold_standard'] = self.calculate_distance_to_gold_standard(titles)
        
        # Виявляємо семантичнand дублandкати
        duplicate_mask = self.detect_semantic_duplicates(titles)
        df['is_semantic_duplicate'] = duplicate_mask
        
        # Логуємо сanтистику
        logger.info(f"[LinguisticDNA] Feature statistics:")
        logger.info(f"  - Historical impact: mean={df['historical_impact_score'].mean():.3f}")
        logger.info(f"  - Commitment flag: {df['commitment_flag'].sum()} positive")
        logger.info(f"  - Distance to gold: mean={df['dist_to_gold_standard'].mean():.3f}")
        logger.info(f"  - Semantic duplicates: {df['is_semantic_duplicate'].sum()}")
        
        return df

# Глобальна функцandя for викорисandння в Stage 3
def add_linguistic_dna_features(df: pd.DataFrame, title_col: str = 'title') -> pd.DataFrame:
    """
    Додає лandнгвandстичнand фandчand до DataFrame
    """
    analyzer = LinguisticDNAAnalyzer()
    return analyzer.add_linguistic_features(df, title_col)
