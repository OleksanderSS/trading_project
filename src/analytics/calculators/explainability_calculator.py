"""
Provides tools for eXplainable AI (XAI) on machine learning models.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

# Create numpy random generator
rng = np.random.default_rng(42)

logger = logging.getLogger(__name__)

class ExplainabilityCalculator:
    """A collection of static methods for model explainability."""

    @staticmethod
    def analyze_feature_importance(
        model: Any,
        X: pd.DataFrame,
        feature_names: list[str],
        y_true: pd.Series | np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Calculates feature importance using native methods or permutation if not available.

        Args:
            model (Any): The trained model object.
            X (pd.DataFrame): The input data (features) used for permutation.
            feature_names (List[str]): A list of feature names.

        Returns:
            Dict[str, float]: A dictionary mapping feature names to their importance scores, sorted descending.
        """
        try:
            importance = {}
            if hasattr(model, 'feature_importances_'):
                # Native support for Trees (LGBM, RF, XGB)
                importances = model.feature_importances_
                importance = dict(zip(feature_names, importances, strict=False))
            elif hasattr(model, 'coef_'):
                # Linear models - use absolute coefficient values
                importances = np.abs(model.coef_).flatten()
                importance = dict(zip(feature_names, importances, strict=False))
            else:
                # Fallback to Permutation Importance for black-box/heavy models
                logger.info(f"Using permutation importance for model {type(model).__name__}")

                # Permutation importance is the INCREASE IN ERROR when a
                # feature is shuffled, and error needs the true values. This
                # used to compute np.mean(np.abs(predictions)) and call it
                # "Mean Absolute Error" -- no y anywhere in it. That measures
                # how much the average prediction MAGNITUDE moves, which a
                # feature can leave untouched while being essential to
                # accuracy, and which a feature that merely shifts the level
                # can dominate without improving anything.
                if y_true is None:
                    logger.warning(
                        "Permutation importance requested without y_true; "
                        "falling back to the shift in mean absolute "
                        "prediction, which is NOT an error metric and ranks "
                        "features by how much they move the output level "
                        "rather than by how much accuracy they carry."
                    )

                def _score(predictions):
                    if y_true is None:
                        return float(np.mean(np.abs(predictions)))
                    return float(np.mean(np.abs(np.asarray(y_true) - predictions)))

                base_error = _score(model.predict(X))

                for i, col in enumerate(feature_names):
                    x_permuted = X.copy()
                    # Shuffle a single column
                    x_permuted.iloc[:, i] = rng.permutation(x_permuted.iloc[:, i])
                    perm_error = _score(model.predict(x_permuted))
                    # Only an INCREASE counts. A shuffle that improves the
                    # score says the feature was not carrying signal, which
                    # is importance zero, not importance |difference|.
                    importance[col] = (
                        max(0.0, perm_error - base_error)
                        if y_true is not None
                        else abs(perm_error - base_error)
                    )

            # Normalize importances to sum to 1.0 for comparability
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}

            # Sort by importance value in descending order
            return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception("Failed to analyze feature importance")
            raise RuntimeError(f"Failed to analyze feature importance: {e}") from e

    @staticmethod
    def explain_single_prediction(model: Any, data_row: pd.DataFrame) -> dict[str, float]:
        """
        Explains a single prediction by treating it as a batch of one.
        This is a convenience wrapper around analyze_feature_importance.

        Args:
            model (Any): The trained model object.
            data_row (pd.DataFrame): A single row of data to be explained.

        Returns:
            Dict[str, float]: A dictionary of feature importances for that specific prediction.
        """
        if not isinstance(data_row, pd.DataFrame) or data_row.shape[0] != 1:
            logger.warning("explain_single_prediction expects a single-row DataFrame.")
            return {}

        try:
            feature_names = data_row.columns.tolist()
            # We can reuse the main importance analyzer with a single row
            return ExplainabilityCalculator.analyze_feature_importance(model, data_row, feature_names)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception("Failed to explain prediction")
            raise RuntimeError(f"Failed to explain prediction: {e}") from e
