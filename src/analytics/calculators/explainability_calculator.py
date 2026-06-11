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

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("⚠️ SHAP not installed. Install with: pip install shap")

class ExplainabilityCalculator:
    """A collection of methods for model explainability, now enhanced with SHAP."""

    @staticmethod
    def analyze_feature_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
        """
        Calculates feature importance using SHAP if available,
        falling back to native methods or permutation.
        """
        if SHAP_AVAILABLE:
            try:
                shap_values = ExplainabilityCalculator.calculate_shap_values(model, X)
                if shap_values is not None:
                    # Calculate mean absolute SHAP values for global importance
                    if isinstance(shap_values, list): # Multi-class
                        # Use average importance across classes
                        importances = np.mean([np.abs(v).mean(0) for v in shap_values], axis=0)
                    else:
                        importances = np.abs(shap_values).mean(0)

                    importance_dict = dict(zip(feature_names, importances, strict=False))
                    return ExplainabilityCalculator._normalize_and_sort(importance_dict)
            except Exception as e:
                logger.warning(f"SHAP feature importance failed: {e}. Falling back to native/permutation.")

        # Fallback to native or permutation
        return ExplainabilityCalculator._fallback_importance(model, X, feature_names)

    @staticmethod
    def calculate_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray | None:
        """
        Calculates SHAP values for the given model and data.
        Automatically selects the appropriate explainer.
        """
        if not SHAP_AVAILABLE:
            return None

        try:
            # ELITE FIX: Ensure X is numeric and only contains numeric columns
            if isinstance(X, pd.DataFrame):
                # Save original columns to identify problematic ones
                X.columns.tolist()

                # Coerce to numeric - strings will become NaN, then 0
                X = X.apply(pd.to_numeric, errors='coerce')

                # Identify columns that are still object or have all NaNs (meaning they were non-numeric)
                object_cols = X.select_dtypes(include=['object']).columns.tolist()
                if object_cols:
                    logger.warning(f"⚠️ SHAP: Removing non-numeric columns: {object_cols}")
                    X = X.drop(columns=object_cols)

                # Fill remaining NaNs with 0
                X = X.fillna(0)

            elif isinstance(X, np.ndarray) and X.dtype == object:
                # For numpy arrays, try converting each element safely
                try:
                    X = X.astype(float)
                except (ValueError, TypeError):
                    logger.warning("⚠️ SHAP: Numpy object array contains non-numeric data. Attempting element-wise coercion.")
                    X = np.array([pd.to_numeric(x, errors='coerce') for x in X.flatten()]).reshape(X.shape)
                    X = np.nan_to_num(X.astype(float), nan=0.0)

            # Select explainer based on model type
            # Check for native model attribute if wrapped in adapter
            native_model = getattr(model, 'model', model)
            model_name = type(native_model).__name__.lower()

            # 1. CatBoost specific (most reliable via native method)
            if 'catboost' in model_name:
                import catboost
                # CatBoost returns [samples, features + 1] where last col is expected value
                # We only want the features part
                shap_values = native_model.get_feature_importance(
                    catboost.Pool(X),
                    type='ShapValues'
                )
                if shap_values.ndim == 2:
                    return shap_values[:, :-1]
                elif shap_values.ndim == 3: # Multi-class
                    return [val[:, :-1] for val in shap_values]
                return shap_values

            # 2. Other Tree-based models (XGBoost, LightGBM, SKLearn)
            if any(t in model_name for t in ['gbm', 'boost', 'forest', 'tree']):
                explainer = shap.TreeExplainer(native_model)
                shap_values = explainer.shap_values(X)

            # 3. Linear models
            elif any(t in model_name for t in ['linear', 'logistic', 'ridge', 'lasso']):
                explainer = shap.LinearExplainer(native_model, X)
                shap_values = explainer.shap_values(X)

            # 4. Fallback to KernelExplainer for black-box/Neural Networks
            else:
                self_logger = logging.getLogger("ExplainabilityCalculator")
                self_logger.info(f"Using KernelExplainer for {model_name} (this may be slow)")

                # Sample background data for speed (very small for KernelExplainer)
                background = shap.sample(X, min(10, len(X)))  # Reduced for stability

                # Define prediction wrapper to handle potential multi-output
                def predict_wrapper(data):
                    try:
                        preds = native_model.predict(data)
                        return preds.flatten() if hasattr(preds, 'flatten') else preds
                    except Exception as e:
                        self_logger.error(f"Model prediction failed in SHAP wrapper: {e}")
                        raise

                try:
                    explainer = shap.KernelExplainer(predict_wrapper, background)
                    shap_values = explainer.shap_values(X, nsamples=50)  # Reduced samples for speed
                except Exception as e:
                    self_logger.error(f"KernelExplainer failed for {model_name}: {e}")
                    # Try with even simpler approach
                    try:
                        # Use mean values as background
                        background_mean = np.mean(X, axis=0).reshape(1, -1)
                        explainer = shap.KernelExplainer(predict_wrapper, background_mean)
                        shap_values = explainer.shap_values(X, nsamples=25)
                    except Exception as e2:
                        self_logger.error(f"Even simplified KernelExplainer failed: {e2}")
                        raise e

            # Standardize output format
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification often returns [neg_class_shap, pos_class_shap]
                return shap_values[1]

            return shap_values

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            return None

    @staticmethod
    def explain_single_prediction(model: Any, data_row: pd.DataFrame) -> dict[str, float]:
        """Explains a single prediction using SHAP or fallback."""
        if not isinstance(data_row, pd.DataFrame) or data_row.shape[0] != 1:
            logger.warning("explain_single_prediction expects a single-row DataFrame.")
            return {}

        feature_names = data_row.columns.tolist()

        if SHAP_AVAILABLE:
            try:
                # We still need background data or a tree model for SHAP
                # For single row, SHAP is actually more powerful than permutation
                shap_values = ExplainabilityCalculator.calculate_shap_values(model, data_row)
                if shap_values is not None:
                    # For single row, shap_values is already the importance for that row
                    val = shap_values[0] if shap_values.ndim > 1 else shap_values
                    importance_dict = dict(zip(feature_names, val, strict=False))
                    return ExplainabilityCalculator._normalize_and_sort(importance_dict)
            except Exception as e:
                logger.debug(f"SHAP single explanation failed: {e}")

        return ExplainabilityCalculator._fallback_importance(model, data_row, feature_names)

    @staticmethod
    def _fallback_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
        """Original importance logic as fallback."""
        importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_, strict=False))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
                importance = dict(zip(feature_names, importances, strict=False))
            else:
                # Permutation Importance
                base_pred = model.predict(X)
                base_error = np.mean(np.abs(base_pred))
                for i, col in enumerate(feature_names):
                    x_permuted = X.copy()
                    x_permuted.iloc[:, i] = rng.permutation(x_permuted.iloc[:, i])
                    perm_pred = model.predict(x_permuted)
                    perm_error = np.mean(np.abs(perm_pred))
                    importance[col] = abs(perm_error - base_error)

            return ExplainabilityCalculator._normalize_and_sort(importance)
        except Exception as e:
            logger.error(f"Fallback importance failed: {e}")
            return {}

    @staticmethod
    def _normalize_and_sort(importance: dict[str, float]) -> dict[str, float]:
        """Normalizes scores to sum to 1.0 and sorts descending."""
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        return dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

