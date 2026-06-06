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
    def analyze_feature_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
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
                base_pred = model.predict(X)
                base_error = np.mean(np.abs(base_pred))  # Using Mean Absolute Error as the metric

                for i, col in enumerate(feature_names):
                    x_permuted = X.copy()
                    # Shuffle a single column
                    x_permuted.iloc[:, i] = rng.permutation(x_permuted.iloc[:, i])
                    perm_pred = model.predict(x_permuted)
                    perm_error = np.mean(np.abs(perm_pred))
                    # The importance is the increase in error
                    importance[col] = abs(perm_error - base_error)

            # Normalize importances to sum to 1.0 for comparability
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}

            # Sort by importance value in descending order
            return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Failed to analyze feature importance: {e}", exc_info=True)
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
            logger.error(f"Failed to explain prediction: {e}", exc_info=True)
            raise RuntimeError(f"Failed to explain prediction: {e}") from e
