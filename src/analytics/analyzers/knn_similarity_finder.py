from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from ..interfaces import IAnalyzer
from ..utils.knn_model_wrapper import KnnModelWrapper

logger = ProjectLogger.get_logger(__name__)


class KnnSimilarityFinder(IAnalyzer):
    """
    🎯 CONTEXTUAL KNN:
    Знаходить схожі історичні моменти, пріоритезуючи той самий ринковий режим.
    """

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.n_neighbors = self.config.get('n_neighbors', 5)
        self.min_regime_samples = self.config.get('min_regime_samples', 20)
        # Missing features are filled with historical medians, which makes a
        # mostly-unknown context look average -- and therefore a confident
        # match for every other average row. The only guard used to be
        # "at least one feature present", so 1 real value out of 20 produced
        # three neighbours with similarity scores and no warning. They were
        # the rows nearest the median, not the rows nearest this context.
        self.min_feature_coverage = float(
            self.config.get('min_feature_coverage', 0.5)
        )
        self.knn_model: KnnModelWrapper | None = None
        self.feature_columns: list[str] = []
        self.feature_medians: pd.Series = pd.Series(dtype=float)
        self.historical_outcomes: pd.DataFrame = pd.DataFrame()
        self._fitted_positions: np.ndarray = np.array([], dtype=int)
        logger.info("KnnSimilarityFinder initialized with Contextual Filtering support.")

    def fit(self, historical_features: pd.DataFrame, historical_outcomes: pd.DataFrame | None = None) -> None:
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

    def analyze(self, data: dict[str, pd.DataFrame], **kwargs) -> dict[str, Any]:
        """
        Знаходить сусідів, враховуючи 'context_pattern_id'.
        """
        historical_df = data.get('historical_features')
        target_df = data.get('target_features')

        if historical_df is None or target_df is None:
            raise DataProcessingError("Missing input dataframes.")

        try:
            # The fingerprint of one bar, not the sequence hash.
            #
            # This filtered on `context_pattern_id`, which is a SHA of the last
            # `pattern_length` fingerprints joined together. Measured on the
            # daily frame of 2026-08-28: 682,035 distinct values across 705,166
            # rows -- 1.03 rows per pattern, no pattern above 589 rows, none at
            # all above a thousand. Against `min_regime_samples = 20` that
            # condition can never hold, so every call fell through to the
            # global search while logging that it had considered the regime.
            #
            # `context_fingerprint` is the state of a single bar built from the
            # eight configured drivers, and it takes 2,027 values: 348 rows per
            # condition on average, 166 conditions above a thousand rows. That
            # is a regime one can actually match against.
            #
            # The sequence id is still accepted when a caller passes it, since
            # a caller that has one means it -- it simply is not the default any
            # more.
            working_historical = historical_df
            regime_key, regime_value = None, None
            for candidate in ('context_fingerprint', 'context_pattern_id'):
                supplied = kwargs.get(candidate)
                if supplied is not None and candidate in historical_df.columns:
                    regime_key, regime_value = candidate, supplied
                    break

            if regime_key is not None:
                regime_df = historical_df[historical_df[regime_key] == regime_value]
                if len(regime_df) >= self.min_regime_samples:
                    logger.info(
                        "🔍 KNN: %s=%s gives %d neighbours; searching inside it.",
                        regime_key, regime_value, len(regime_df),
                    )
                    working_historical = regime_df
                else:
                    logger.info(
                        "ℹ️ KNN: %s=%s has %d rows, under the %d needed; "
                        "searching globally instead.",
                        regime_key, regime_value, len(regime_df),
                        self.min_regime_samples,
                    )

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

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
        # Coverage, not mere presence: a row where one value in twenty is
        # real is imputed into an average-looking row and matches whatever
        # sits near the median.
        minimum = max(1, int(round(self.min_feature_coverage * len(common_cols))))
        hist_kept = X_hist.notna().sum(axis=1) >= minimum
        target_kept = X_target.notna().sum(axis=1) >= minimum
        dropped_targets = int((~target_kept).sum())
        if dropped_targets:
            logger.warning(
                "%d target row(s) carry fewer than %d of %d numeric features "
                "and were excluded from KNN matching; imputing that many "
                "medians would have matched them to the average row rather "
                "than to a similar context.",
                dropped_targets, minimum, len(common_cols),
            )
        X_hist = X_hist.loc[hist_kept]
        X_target = X_target.loc[target_kept]
        if X_hist.empty or X_target.empty:
            raise DataProcessingError(
                f"No rows with at least {minimum} of {len(common_cols)} "
                "numeric KNN features."
            )

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
        present = int(X_target.notna().sum(axis=1).iloc[0])
        minimum = max(1, int(round(self.min_feature_coverage * len(self.feature_columns))))
        if present < minimum:
            raise DataProcessingError(
                f"Current context has {present} of {len(self.feature_columns)} "
                f"numeric KNN features; at least {minimum} are required. "
                "Filling the rest with medians would match it to the average "
                "historical row rather than to a similar situation."
            )

        X_target = X_target.fillna(self.feature_medians)
        distances, indices = self.knn_model.find_neighbors(X_target)
        if len(indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        neighbor_positions = self._fitted_positions[np.asarray(indices[0], dtype=int)]
        return neighbor_positions, np.asarray(distances[0], dtype=float)

    def _format_results(self, distances: list, indices: list, target_index: pd.Index, historical_index: pd.Index) -> dict:
        results = {}
        for i, target_id in enumerate(target_index):
            similar_items = []
            if i >= len(indices):
                continue
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
