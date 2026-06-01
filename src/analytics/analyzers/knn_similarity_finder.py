import pandas as pd
from typing import Dict, Any, List, Optional

from ..interfaces import IAnalyzer
from ..utils.knn_model_wrapper import KnnModelWrapper
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger(__name__)


class KnnSimilarityFinder(IAnalyzer):
    """
    🎯 CONTEXTUAL KNN:
    Знаходить схожі історичні моменти, пріоритезуючи той самий ринковий режим.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_neighbors = self.config.get('n_neighbors', 5)
        self.min_regime_samples = self.config.get('min_regime_samples', 20)
        logger.info(f"KnnSimilarityFinder initialized with Contextual Filtering support.")

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Знаходить сусідів, враховуючи 'context_pattern_id'.
        """
        historical_df = data.get('historical_features')
        target_df = data.get('target_features')

        if historical_df is None or target_df is None:
            raise DataProcessingError("Missing input dataframes.")

        try:
            # ✅ ELITE: Контекстна фільтрація
            # Якщо у нас є pattern_id, спочатку шукаємо в ньому
            pattern_id = kwargs.get('context_pattern_id')
            
            working_historical = historical_df
            if pattern_id and 'context_pattern_id' in historical_df.columns:
                regime_df = historical_df[historical_df['context_pattern_id'] == pattern_id]
                if len(regime_df) >= self.min_regime_samples:
                    logger.info(f"🔍 KNN: Using regime-specific subset for pattern {pattern_id} ({len(regime_df)} samples)")
                    working_historical = regime_df
                else:
                    logger.info(f"ℹ️ KNN: Pattern {pattern_id} has too few samples ({len(regime_df)}). Falling back to global search.")

            # Очищуємо не-числові колонки для KNN
            X_hist = working_historical.select_dtypes(include=['number'])
            X_target = target_df.select_dtypes(include=['number'])
            
            # Вирівнюємо колонки
            common_cols = X_hist.columns.intersection(X_target.columns)
            if len(common_cols) == 0:
                raise DataProcessingError("No common numeric columns for KNN analysis.")
            
            X_hist = X_hist[common_cols].fillna(0)  # audit-ignore: FILLNA_ZERO_SUSPICIOUS
            X_target = X_target[common_cols].fillna(0)  # audit-ignore: FILLNA_ZERO_SUSPICIOUS

            n_neighbors = min(len(X_hist), kwargs.get('n_neighbors', self.n_neighbors))
            if n_neighbors <= 0: 
                return {"similarities": {}}
            
            knn_model = KnnModelWrapper(n_neighbors=n_neighbors)
            knn_model.fit(X_hist)
            distances, indices = knn_model.find_neighbors(X_target)

            results = self._format_results(distances, indices, X_target.index, X_hist.index)
            return {"similarities": results, "regime_used": pattern_id if working_historical is not historical_df else "global"}

        except Exception as e:
            logger.error(f"KNN analysis failed: {e}", exc_info=True)
            raise DataProcessingError(f"KNN analysis failed: {e}") from e

    def _format_results(self, distances: List, indices: List, target_index: pd.Index, historical_index: pd.Index) -> Dict:
        results = {}
        for i, target_id in enumerate(target_index):
            similar_items = []
            if i >= len(indices): continue
            for j, neighbor_idx in enumerate(indices[i]):
                neighbor_id = historical_index[neighbor_idx]
                similarity_score = 1 / (1 + distances[i][j])
                similar_items.append({
                    "id": neighbor_id,
                    "distance": round(float(distances[i][j]), 5),
                    "similarity_score": round(similarity_score, 4)
                })
            results[target_id] = sorted(similar_items, key=lambda x: x['similarity_score'], reverse=True)
        return results
