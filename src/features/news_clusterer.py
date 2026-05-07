"""
News Clusterer
Кластеризує схожі новини для пришвидшення тренування без втрати якості
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NewsClusterer")


class NewsClusterer:
    """
    Кластеризує схожі новини для зменшення обсягу даних без втрати якості
    
    Приклад:
    - "Fed raises rates by 0.25%" 
    - "Federal Reserve hikes interest rates"
    - "Fed increases rates"
    → Всі в один кластер, вибирається 1 представник
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2,
        use_embeddings: bool = True
    ):
        """
        Args:
            similarity_threshold: Поріг схожості для об'єднання в кластер (0-1)
            min_cluster_size: Мінімальний розмір кластера для об'єднання
            use_embeddings: Використовувати ембедінги (SBERT) чи TF-IDF
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.use_embeddings = use_embeddings
        
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Ініціалізувати модель для обчислення схожості"""
        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                # Використовуємо фінансову модель якщо доступна
                try:
                    self.model = SentenceTransformer('ProsusAI/finbert')
                    logger.info("✅ Loaded FinBERT for news clustering")
                except Exception:
                    # Fallback до загальної моделі
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✅ Loaded MiniLM for news clustering")
            except ImportError:
                logger.warning("⚠️ sentence-transformers not installed. Falling back to TF-IDF")
                self.use_embeddings = False
                self._initialize_tfidf()
        else:
            self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Ініціалізувати TF-IDF як fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.model = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        logger.info("✅ Initialized TF-IDF for news clustering")
    
    def cluster_news(
        self,
        news_df: pd.DataFrame,
        text_column: str = 'title',
        return_representatives: bool = True
    ) -> pd.DataFrame:
        """
        Кластеризувати новини та повернути датасет з cluster_id
        
        Args:
            news_df: DataFrame з новинами
            text_column: Назва колонки з текстом новини
            return_representatives: Повернути тільки представників кластерів
        
        Returns:
            DataFrame з додатковою колонкою cluster_id
        """
        if news_df.empty:
            logger.warning("Empty news dataframe provided")
            return news_df
        
        if text_column not in news_df.columns:
            logger.error(f"Column '{text_column}' not found in news_df")
            return news_df
        
        logger.info(f"Clustering {len(news_df)} news articles...")
        
        # Обчислити ембедінги/векторизацію
        embeddings = self._compute_embeddings(news_df[text_column].tolist())
        
        # Кластеризувати
        cluster_labels = self._cluster_embeddings(embeddings)
        
        # Додати cluster_id до датасету
        news_df = news_df.copy()
        news_df['cluster_id'] = cluster_labels
        news_df['is_cluster_representative'] = False
        
        # Вибрати представників кластерів
        representatives = self._select_representatives(news_df, embeddings)
        
        # Позначити представників
        news_df.loc[representatives, 'is_cluster_representative'] = True
        
        n_clusters = news_df['cluster_id'].nunique()
        n_representatives = len(representatives)
        reduction_pct = (1 - n_representatives / len(news_df)) * 100
        
        logger.info(f"✅ Clustered into {n_clusters} clusters")
        logger.info(f"✅ Selected {n_representatives} representatives ({reduction_pct:.1f}% reduction)")
        
        if return_representatives:
            return news_df[news_df['is_cluster_representative']].copy()
        else:
            return news_df
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Обчислити ембедінги для текстів"""
        # ✅ FIX: Check if model is initialized
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model() first.")
        
        if self.use_embeddings:
            # Sentence-BERT embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        else:
            # TF-IDF vectors
            embeddings = self.model.fit_transform(texts).toarray()
            return embeddings
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Кластеризувати ембедінги
        
        Використовуємо DBSCAN для автоматичного визначення кількості кластерів
        """
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Обчислити матрицю схожості
        similarity_matrix = cosine_similarity(embeddings)
        
        # Конвертувати в distance matrix (1 - similarity)
        # Clip to [0, 2] range to handle negative similarities
        distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
        
        # DBSCAN кластеризація
        clusterer = DBSCAN(
            eps=1 - self.similarity_threshold,  # Distance threshold
            min_samples=self.min_cluster_size,
            metric='precomputed'
        )
        
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # DBSCAN позначає outliers як -1, змінимо на унікальні ID
        max_cluster = cluster_labels.max()
        outlier_mask = cluster_labels == -1
        if outlier_mask.any():
            # Кожен outlier отримує свій унікальний кластер
            unique_ids = np.arange(max_cluster + 1, max_cluster + 1 + outlier_mask.sum())
            cluster_labels[outlier_mask] = unique_ids
        
        # ✅ FIX: Explicitly return as ndarray
        return np.asarray(cluster_labels)
    
    def _select_representatives(
        self,
        news_df: pd.DataFrame,
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Вибрати представників для кожного кластера
        
        Стратегія: вибрати новину найближчу до центроїда кластера
        """
        representatives = []
        
        for cluster_id in news_df['cluster_id'].unique():
            cluster_mask = news_df['cluster_id'] == cluster_id
            cluster_indices = news_df[cluster_mask].index.tolist()
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_indices) == 1:
                # Якщо кластер з 1 новини, вона і є представником
                representatives.append(cluster_indices[0])
            else:
                # Обчислити центроїд кластера
                centroid = cluster_embeddings.mean(axis=0)
                
                # Знайти найближчу новину до центроїда
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                closest_idx = distances.argmin()
                
                representatives.append(cluster_indices[closest_idx])
        
        return representatives
    
    def get_cluster_statistics(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Отримати статистику кластеризації"""
        if 'cluster_id' not in news_df.columns:
            return {}
        
        cluster_sizes = news_df['cluster_id'].value_counts()
        
        stats = {
            'total_news': len(news_df),
            'n_clusters': news_df['cluster_id'].nunique(),
            'avg_cluster_size': cluster_sizes.mean(),
            'max_cluster_size': cluster_sizes.max(),
            'min_cluster_size': cluster_sizes.min(),
            'singleton_clusters': (cluster_sizes == 1).sum(),
            'reduction_ratio': 1 - (news_df['cluster_id'].nunique() / len(news_df))
        }
        
        return stats
    
    def save_cluster_mapping(
        self,
        news_df: pd.DataFrame,
        output_path: Path
    ):
        """Зберегти мапінг кластерів для аналізу"""
        if 'cluster_id' not in news_df.columns:
            logger.warning("No cluster_id column found")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Створити мапінг: cluster_id → список новин
        cluster_mapping = {}
        
        for cluster_id in news_df['cluster_id'].unique():
            cluster_news = news_df[news_df['cluster_id'] == cluster_id]
            
            cluster_mapping[int(cluster_id)] = {
                'size': len(cluster_news),
                'representative': cluster_news[cluster_news['is_cluster_representative']].iloc[0]['title'] if 'is_cluster_representative' in cluster_news.columns else cluster_news.iloc[0]['title'],
                'news_titles': cluster_news['title'].tolist() if 'title' in cluster_news.columns else []
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Cluster mapping saved to {output_path}")


def cluster_news_simple(
    news_df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    text_column: str = 'title'
) -> pd.DataFrame:
    """
    Простий інтерфейс для кластеризації новин
    
    Args:
        news_df: DataFrame з новинами
        similarity_threshold: Поріг схожості (0-1)
        text_column: Назва колонки з текстом
    
    Returns:
        DataFrame тільки з представниками кластерів
    """
    clusterer = NewsClusterer(
        similarity_threshold=similarity_threshold,
        min_cluster_size=2,
        use_embeddings=True
    )
    
    return clusterer.cluster_news(
        news_df,
        text_column=text_column,
        return_representatives=True
    )
