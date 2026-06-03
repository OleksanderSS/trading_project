import pandas as pd
import numpy as np
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
        self.knn_model: KnnModelWrapper | None = None
        self.feature_columns: list[str] = []
        self.feature_medians: pd.Series = pd.Series(dtype=float)
        self.historical_outcomes: pd.DataFrame = pd.DataFrame()
        self._fitted_positions: np.ndarray = np.array([], dtype=int)
        logger.info(f"KnnSimilarityFinder initialized with Contextual Filtering support.")

    def fit(self, historical_features: pd.DataFrame, historical_outcomes: Optional[pd.DataFrame] = None) -> None:
        """Fit the reusable KNN index for contextual model selection."""
        X_hist, _ = self._prepare_feature_matrices(historical_features, historical_features)
        n_neighbors = min(len(X_hist), self.n_neighbors)
        if n_neighbors <= 0:
            raise DataProcessingError("No historical rows available for KNN fitting.")

        self.knn_model = KnnModelWrapper(n_neighbors=n_neighbors)
        self.knn_model.fit(X_hist)
        self.feature_columns = X_hist.columns.tolist()
        self.feature_medians = X_hist.median()
        self.historical_outcomes = (
            historical_outcomes.reindex(historical_features.index)
            if historical_outcomes is not None
            else pd.DataFrame(index=historical_features.index)
        )
        self._fitted_positions = np.array([historical_features.index.get_loc(idx) for idx in X_hist.index], dtype=int)

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
            
            X_hist, X_target = self._prepare_feature_matrices(working_historical, target_df)

            n_neighbors = min(len(X_hist), kwargs.get('n_neighbors', self.n_neighbors))
            if n_neighbors <= 0: 
                return {"similarities": {}}
            
            knn_model = KnnModelWrapper(n_neighbors=n_neighbors)
            knn_model.fit(X_hist)
            distances, indices = knn_model.find_neighbors(X_target)

            results = self._format_results(distances, indices, X_target.index, X_hist.index)
            return {"similarities": results, "regime_used": pattern_id if working_historical is not historical_df else "global"}

        except Exception as e:
            raise DataProcessingError(f"KNN analysis failed: {e}") from e

    def _prepare_feature_matrices(
        self,
        historical_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align numeric KNN inputs and impute partial gaps from historical medians."""
        X_hist = historical_df.apply(pd.to_numeric, errors='coerce')
        X_target = target_df.apply(pd.to_numeric, errors='coerce')

        common_cols = X_hist.columns.intersection(X_target.columns)
        if len(common_cols) == 0:
            raise DataProcessingError("No common numeric columns for KNN analysis.")

        X_hist = X_hist[common_cols].replace([np.inf, -np.inf], np.nan)
        X_target = X_target[common_cols].replace([np.inf, -np.inf], np.nan)
        X_hist = X_hist.loc[X_hist.notna().any(axis=1)]
        X_target = X_target.loc[X_target.notna().any(axis=1)]
        if X_hist.empty or X_target.empty:
            raise DataProcessingError("No non-empty numeric rows for KNN analysis.")

        medians = X_hist.median()
        valid_cols = medians.dropna().index
        if len(valid_cols) == 0:
            raise DataProcessingError("No KNN columns with historical median values.")

        X_hist = X_hist[valid_cols].fillna(medians[valid_cols])
        X_target = X_target[valid_cols].fillna(medians[valid_cols])
        return X_hist, X_target

    def find_similar_situations(self, current_context: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility API used by ContextualModelSelector."""
        if self.knn_model is None or not self.feature_columns:
            raise DataProcessingError("KNN finder must be fitted before searching similar situations.")

        target_df = current_context.to_frame().T
        X_target = target_df.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors='coerce')
        X_target = X_target.replace([np.inf, -np.inf], np.nan)
        if X_target.notna().sum(axis=1).iloc[0] == 0:
            raise DataProcessingError("Current context has no usable numeric KNN features.")

        X_target = X_target.fillna(self.feature_medians)
        distances, indices = self.knn_model.find_neighbors(X_target)
        if len(indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        neighbor_positions = self._fitted_positions[np.asarray(indices[0], dtype=int)]
        return neighbor_positions, np.asarray(distances[0], dtype=float)

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
