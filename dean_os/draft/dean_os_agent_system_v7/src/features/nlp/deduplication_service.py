

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DeduplicationService:
    """
    A service to deduplicate news articles based on semantic similarity using TF-IDF and KMeans.
    """

    def __init__(self, n_clusters: int = 10, max_features: int = 1000):
        """
        Initializes the DeduplicationService.

        Args:
            n_clusters (int): The number of clusters to form.
            max_features (int): The maximum number of features to use for TF-IDF.
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

    def deduplicate(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        Deduplicates a DataFrame of news articles.

        Args:
            df (pd.DataFrame): The DataFrame to deduplicate.
            text_column (str): The name of the column containing the text to analyze.

        Returns:
            pd.DataFrame: A DataFrame with duplicate articles removed.
        """
        if df.empty or text_column not in df.columns:
            logger.warning("[DeduplicationService] Empty DataFrame or missing text column.")
            return df

        # 1. Drop exact duplicates
        df_deduped = df.drop_duplicates(subset=[text_column])

        # Reset index to avoid issues with indexing later
        df_deduped = df_deduped.reset_index(drop=True)

        texts = df_deduped[text_column].fillna('').astype(str)
        non_empty_mask = texts.str.strip() != ""

        if non_empty_mask.sum() < 2:
            logger.warning("[DeduplicationService] Not enough content to perform clustering.")
            return df_deduped

        # 2. Cluster remaining articles
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts[non_empty_mask])

            # Adjust n_clusters if there are fewer articles than clusters
            n_clusters = min(self.n_clusters, tfidf_matrix.shape[0])
            self.kmeans.n_clusters = n_clusters

            clusters = self.kmeans.fit_predict(tfidf_matrix)

            df_clustered = df_deduped[non_empty_mask].copy()
            df_clustered['cluster'] = clusters

            # 3. Select one representative article from each cluster (e.g., the longest one)
            unique_articles_idx = df_clustered.groupby('cluster')[text_column].apply(lambda x: x.str.len().idxmax())

            # Get the original indices from the non-empty dataframe
            final_df = df_deduped.loc[unique_articles_idx]

            logger.info(f"[DeduplicationService] Deduplicated {len(df)} articles down to {len(final_df)} unique articles.")
            return final_df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"[DeduplicationService] Error during clustering: {e}", exc_info=True)
            return df_deduped # Return the dataframe with only exact duplicates removed

