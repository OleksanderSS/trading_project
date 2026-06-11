

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DeduplicationService:
    """
    A service to deduplicate news articles based on semantic similarity using TF-IDF and Cosine Similarity.
    """

    def __init__(self, similarity_threshold: float = 0.85, max_features: int = 1000):
        """
        Initializes the DeduplicationService.

        Args:
            similarity_threshold (float): Cosine similarity threshold above which articles are considered duplicates.
            max_features (int): The maximum number of features to use for TF-IDF.
        """
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)

    def deduplicate(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        Deduplicates a DataFrame of news articles based on cosine similarity of TF-IDF vectors.

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
        df_deduped = df_deduped.reset_index(drop=True)

        texts = df_deduped[text_column].fillna('').astype(str)
        non_empty_mask = texts.str.strip() != ""

        if non_empty_mask.sum() < 2:
            logger.warning("[DeduplicationService] Not enough content to perform semantic deduplication.")
            return df_deduped

        # 2. Compute Cosine Similarity between articles
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts[non_empty_mask])
            
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            sim_matrix = cosine_similarity(tfidf_matrix)
            n_samples = sim_matrix.shape[0]
            to_remove = set()
            non_empty_indices = df_deduped[non_empty_mask].index.tolist()

            for i in range(n_samples):
                if i in to_remove:
                    continue
                # Find all articles similar to current article i
                similar_indices = np.where(sim_matrix[i] >= self.similarity_threshold)[0]
                for idx in similar_indices:
                    if idx != i:
                        # Keep the longer article to preserve context/details
                        len_i = len(texts.iloc[non_empty_indices[i]])
                        len_idx = len(texts.iloc[non_empty_indices[idx]])
                        if len_idx <= len_i:
                            to_remove.add(idx)
                        else:
                            to_remove.add(i)
                            break

            # Map the indices to remove back to original df_deduped indices
            removed_original_indices = [non_empty_indices[idx] for idx in to_remove]
            final_df = df_deduped.drop(index=removed_original_indices)

            logger.info(f"[DeduplicationService] Semantic deduplication removed {len(df) - len(final_df)} duplicate articles. Remaining: {len(final_df)}")
            return final_df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"[DeduplicationService] Error during semantic deduplication: {e}", exc_info=True)
            return df_deduped


