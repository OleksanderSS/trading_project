# core/analysis/knn_similarity.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from utils.logger_fixed import ProjectLogger
from config.feature_config import CORE_FEATURES

logger = ProjectLogger.get_logger(__name__)


class KNNSimilarityAnalyzer:
    """KNN for пошуку аналогandчних ринкових ситуацandй"""

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Навчає на andсторичних data"""
        features = [f for f in CORE_FEATURES if f in df.columns]
        X = df[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled)
        self.is_fitted = True
        logger.info(f"KNN fitted on {len(df)} samples with {len(features)} features")

    def find_similar_situations(self, current_features: pd.Series) -> Dict:
        """Знаходить схожand ситуацandї в andсторandї"""
        if not self.is_fitted:
            return {"error": "KNN not fitted"}

        features = [f for f in CORE_FEATURES if f in current_features.index]
        X_current = current_features[features].fillna(0).values.reshape(1, -1)
        X_current_scaled = self.scaler.transform(X_current)

        distances, indices = self.knn.kneighbors(X_current_scaled)

        return {
            "similar_indices": indices[0],
            "distances": distances[0],
            "similarity_score": 1.0 / (1.0 + distances[0].mean())
        }

    def analyze_outcomes(self, df: pd.DataFrame, similar_indices: np.ndarray, 
                        time_horizons: List[str] = ["15m", "1h", "1d"]) -> Dict:
        """Аналandwithує реwithульandти схожих ситуацandй"""
        outcomes = {}
        
        for horizon in time_horizons:
            close_col = f"{horizon}_close_nvda"  # Приклад for NVDA
            if close_col not in df.columns:
                continue
                
            future_returns = []
            for idx in similar_indices:
                if idx + 1 < len(df):
                    current_price = df.iloc[idx][close_col]
                    next_price = df.iloc[idx + 1][close_col]
                    if pd.notna(current_price) and pd.notna(next_price):
                        ret = (next_price - current_price) / current_price
                        future_returns.append(ret)
            
            if future_returns:
                outcomes[horizon] = {
                    "mean_return": np.mean(future_returns),
                    "success_rate": sum(1 for r in future_returns if r > 0) / len(future_returns),
                    "volatility": np.std(future_returns),
                    "samples": len(future_returns)
                }
        
        return outcomes
    
    def get_recommendation(self, outcomes: Dict, threshold: float = 0.6) -> Dict:
        """Геnotрує рекомендацandю на основand аналогandй"""
        if not outcomes:
            return {"recommendation": "HOLD", "confidence": 0.0}
        
        success_rates = [v["success_rate"] for v in outcomes.values()]
        avg_success = np.mean(success_rates)
        
        if avg_success >= threshold:
            recommendation = "BUY"
        elif avg_success <= (1 - threshold):
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return {
            "recommendation": recommendation,
            "confidence": abs(avg_success - 0.5) * 2,
            "analysis": outcomes
        }