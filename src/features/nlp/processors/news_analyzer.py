# src/feature_engineering/nlp/news_analyzer.py

import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from src.core.logging.logger import ProjectLogger


logger = ProjectLogger.get_logger("TradingProjectLogger")

class QuickNewsAnalyzer:
    """TF-IDF + KMeans + TextBlob sentiment."""

    def __init__(self,
                 n_clusters: int = 5,
                 max_features: int = 1000):
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        self.clustered_df: Optional[pd.DataFrame] = None
        logger.info(f"[QuickNewsAnalyzer] Initialized with n_clusters={n_clusters}, max_features={max_features}")

    def _add_empty_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['cluster'] = np.nan
        df['sentiment_score'] = 0.0
        df['subjectivity_score'] = 0.0
        return df

    def cluster_and_analyze(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        if df.empty or text_column not in df.columns:
            logger.warning(f"[QuickNewsAnalyzer] [WARN] Empty DataFrame or missing column '{text_column}'")
            return self._add_empty_cols(df)

        df_copy = df.copy()
        
        # ✅ ЗБЕРІГАЄМО ІНДЕКС (може бути DatetimeIndex)
        original_index = df_copy.index
        logger.info(f"[QuickNewsAnalyzer] Input index type: {type(original_index)}")
        
        texts = df_copy[text_column].fillna('').astype(str)
        nonempty_idx = texts.str.strip() != ""
        texts_nonempty = texts[nonempty_idx]

        if len(texts_nonempty) < 2:
            logger.warning("[QuickNewsAnalyzer] [WARN] Too few texts for clustering")
            return self._add_empty_cols(df_copy)

        try:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
            tfidf_matrix = self.vectorizer.fit_transform(texts_nonempty)

            n_clusters = min(self.n_clusters, tfidf_matrix.shape[0])
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_copy.loc[nonempty_idx, 'cluster'] = self.kmeans.fit_predict(tfidf_matrix)

            # Use existing sentiment from FinBERT if available, otherwise use TextBlob
            if 'sentiment' in df_copy.columns:
                logger.info("[QuickNewsAnalyzer] Using existing FinBERT sentiment scores")
                df_copy['sentiment_score'] = df_copy['sentiment']
            else:
                logger.info("[QuickNewsAnalyzer] Computing sentiment with TextBlob")
                sentiments = [TextBlob(t).sentiment for t in texts_nonempty]
                df_copy.loc[nonempty_idx, 'sentiment_score'] = [s.polarity for s in sentiments]
            
            # Always compute subjectivity with TextBlob
            sentiments = [TextBlob(t).sentiment for t in texts_nonempty]
            df_copy.loc[nonempty_idx, 'subjectivity_score'] = [s.subjectivity for s in sentiments]

            # ✅ ВІДНОВЛЮЄМО ОРИГІНАЛЬНИЙ ІНДЕКС
            df_copy.index = original_index
            
            self.clustered_df = df_copy
            logger.info(f"[QuickNewsAnalyzer] [OK] Clustering complete: {len(texts_nonempty)} news in {n_clusters} clusters")
            return df_copy

        except Exception as e:
            logger.error(f"[QuickNewsAnalyzer] [ERROR] Error in clustering: {e}", exc_info=True)
            return self._add_empty_cols(df_copy)

    def average_sentiment(self) -> Dict[str, float]:
        if self.clustered_df is None or self.clustered_df.empty:
            return {'sentiment_score': 0.0, 'subjectivity_score': 0.0}
        avg = {
            'sentiment_score': float(self.clustered_df['sentiment_score'].mean()),
            'subjectivity_score': float(self.clustered_df['subjectivity_score'].mean())
        }
        logger.info(f"[QuickNewsAnalyzer] [DATA] Average sentiment: {avg}")
        return avg

# --------------------------
# Compatibility for old NewsAnalyzer
# --------------------------
class NewsAnalyzer:
    """
    Compatibility class for old imports:
    uses QuickNewsAnalyzer under the hood.
    Methods: cluster_news, get_latest_news_sentiment
    """

    def __init__(self, n_clusters: int = 5, max_features: int = 1000):
        self._analyzer = QuickNewsAnalyzer(n_clusters=n_clusters, max_features=max_features)
        self.clustered_df: Optional[pd.DataFrame] = None

    def cluster_news(self,
        df: pd.DataFrame,
        text_column: str = 'title',
        date_column: str = 'published_at') -> pd.DataFrame:
        if df.empty or text_column not in df.columns:
            logger.warning("[WARN] Empty DataFrame or missing column '%s'", text_column)
            self.clustered_df = self._analyzer._add_empty_cols(df)
            return self.clustered_df

        df_copy = df.copy()

        # --- Date Handling ---
        if date_column in df_copy.columns:
            df_copy.index = pd.to_datetime(df_copy[date_column], errors='coerce')
            df_copy = df_copy[df_copy.index.notna()]
            if df_copy.empty:
                logger.warning("[WARN] DataFrame is empty after date check")
                self.clustered_df = self._analyzer._add_empty_cols(df_copy)
                return self.clustered_df
            logger.info(f"[NewsAnalyzer] Set DatetimeIndex from '{date_column}' column. Index type: {type(df_copy.index)}")

        # Combine title + summary into content for clustering
        if 'title' in df_copy.columns:
            title_series = df_copy['title'].fillna('')
        else:
            title_series = pd.Series('', index=df_copy.index)
        
        if 'summary' in df_copy.columns:
            summary_series = df_copy['summary'].fillna('')
        else:
            summary_series = pd.Series('', index=df_copy.index)
        
        df_copy['content'] = title_series.astype(str) + '. ' + summary_series.astype(str)

        self.clustered_df = self._analyzer.cluster_and_analyze(df_copy, text_column='content')
        
        # ✅ VERIFY: Check if index is preserved
        logger.info(f"[NewsAnalyzer] Returning DataFrame with index type: {type(self.clustered_df.index)}")
        
        return self.clustered_df

    def get_latest_news_sentiment(self, date: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        df = self.clustered_df
        if df is None or df.empty:
            return {'sentiment_score': 0.0, 'subjectivity_score': 0.0}

        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

        if date is not None:
            date = pd.to_datetime(date)
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= date]
        if df.empty:
            return {'sentiment_score': 0.0, 'subjectivity_score': 0.0}

        return {
            'sentiment_score': float(df['sentiment_score'].iloc[-1]),
            'subjectivity_score': float(df['subjectivity_score'].iloc[-1])
        }
