import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

class KnnSimilarityFinder:
    """
    Finds historical analogs using KNN/world-state cluster search over accumulated
    World State snapshots and pipeline contexts.
    
    This acts as the "calibrated base rates" layer replacing deterministic seed matching.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.knn_model = None
        self.feature_columns: list[str] = []
        self.historical_data: pd.DataFrame | None = None
        self._fitted = False

    def fit(self, historical_records: list[dict[str, Any]]) -> None:
        """
        Fit the KNN model with historical pipeline contexts.
        Each record should ideally contain numeric indicators.
        """
        if not historical_records:
            logger.info("No historical records provided to KnnSimilarityFinder.")
            return

        df = pd.DataFrame(historical_records)
        
        # Select only numeric columns for KNN
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warning("No numeric columns found in historical records for KNN.")
            return

        # Fill NaNs with median
        medians = numeric_df.median()
        numeric_df = numeric_df.fillna(medians)
        
        # Drop columns that are still completely NaN
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        if numeric_df.empty or len(numeric_df) == 0:
            logger.warning("Numeric dataframe is empty after preprocessing.")
            return

        self.feature_columns = numeric_df.columns.tolist()
        
        n_neighbors = min(len(numeric_df), self.n_neighbors)
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.knn_model.fit(numeric_df)
        
        self.historical_data = df
        self._fitted = True
        logger.info(f"KnnSimilarityFinder fitted on {len(numeric_df)} records with {len(self.feature_columns)} features.")

    def find_analogies(self, current_context: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Find the most similar historical records for the given context.
        """
        if not self._fitted or self.knn_model is None or self.historical_data is None:
            # Fallback when the system is not yet trained / lacks historical data
            return []

        target_df = pd.DataFrame([current_context])
        
        # Ensure target has the exact same feature columns as the fitted model
        missing_cols = set(self.feature_columns) - set(target_df.columns)
        for col in missing_cols:
            target_df[col] = np.nan
            
        X_target = target_df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        X_target = X_target.fillna(0) # Fill NaNs for inference if missing

        distances, indices = self.knn_model.kneighbors(X_target)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            record = self.historical_data.iloc[idx].to_dict()
            record["_knn_distance"] = float(dist)
            record["_knn_similarity"] = 1.0 / (1.0 + float(dist)) # Convert distance to a 0-1 similarity score
            results.append(record)
            
        return results
