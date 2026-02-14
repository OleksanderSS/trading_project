# utils/news_processing.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils.logger import ProjectLogger
from config.config import DATA_PATH

logger = ProjectLogger.get_logger(__name__)

def process_news_data(news_df: pd.DataFrame, 
                     cluster_method: str = "kmeans",
                     n_clusters: int = 10,
                     save_csv: bool = True) -> pd.DataFrame:
    """Process and cluster news data"""
    
    if news_df.empty:
        logger.warning("Empty news dataframe provided")
        return pd.DataFrame()
    
    # Clean and preprocess
    processed_df = _preprocess_news(news_df)
    
    # Cluster news
    clustered_df = _cluster_news(processed_df, method=cluster_method, n_clusters=n_clusters)
    
    # Save if requested
    if save_csv:
        try:
            out_dir = os.path.join(DATA_PATH, 'news')
            os.makedirs(out_dir, exist_ok=True)
            fname = f"clustered_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = os.path.join(out_dir, fname)
            clustered_df.to_parquet(filepath, index=False)
            logger.info(f"Saved clustered news to {filepath}")
        except Exception as e:
            logger.error(f"Error saving clustered news: {e}")
    
    return clustered_df

def _preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess news data"""
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['title', 'text'])
    
    # Clean text
    df_clean['title'] = df_clean['title'].str.strip()
    df_clean['text'] = df_clean['text'].str.strip()
    
    # Filter empty texts
    df_clean = df_clean[(df_clean['title'].str.len() > 0) & 
                       (df_clean['text'].str.len() > 0)]
    
    return df_clean

def _cluster_news(df: pd.DataFrame, method: str = "kmeans", n_clusters: int = 10) -> pd.DataFrame:
    """Cluster news articles"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Add cluster labels
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        logger.info(f"Clustered {len(df)} news articles into {n_clusters} clusters")
        return df_clustered
        
    except ImportError:
        logger.warning("sklearn not available, returning original dataframe")
        return df
    except Exception as e:
        logger.error(f"Error clustering news: {e}")
        return df

def get_news_sentiment(news_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate sentiment scores for news data"""
    try:
        from textblob import TextBlob
        
        sentiments = []
        for text in news_df['text']:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        
        avg_sentiment = np.mean(sentiments)
        
        return {
            'average_sentiment': avg_sentiment,
            'positive_count': sum(1 for s in sentiments if s > 0.1),
            'negative_count': sum(1 for s in sentiments if s < -0.1),
            'neutral_count': sum(1 for s in sentiments if -0.1 <= s <= 0.1)
        }
        
    except ImportError:
        logger.warning("textblob not available, returning neutral sentiment")
        return {'average_sentiment': 0.0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': len(news_df)}
    except Exception as e:
        logger.error(f"Error calculating sentiment: {e}")
        return {'average_sentiment': 0.0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': 0}

def filter_news_by_date(news_df: pd.DataFrame, 
                       start_date: datetime = None,
                       end_date: datetime = None) -> pd.DataFrame:
    """Filter news by date range"""
    if 'published_at' not in news_df.columns:
        logger.warning("No 'published_at' column found")
        return news_df
    
    df_filtered = news_df.copy()
    df_filtered['published_at'] = pd.to_datetime(df_filtered['published_at'])
    
    if start_date:
        df_filtered = df_filtered[df_filtered['published_at'] >= start_date]
    
    if end_date:
        df_filtered = df_filtered[df_filtered['published_at'] <= end_date]
    
    return df_filtered
