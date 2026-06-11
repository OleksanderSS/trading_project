"""
SHAP (SHapley Additive exPlanations) Analyzer
Calculates feature importance and prediction explanations for champion models.
"""
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.models.loader import ModelLoaderStrategy

from ..calculators.explainability_calculator import ExplainabilityCalculator
from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger(__name__)

class ShapAnalyzer(IAnalyzer):
    """
    Analyzes model behavior using SHAP values.
    Provides global feature importance and local prediction explanations.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.loader = ModelLoaderStrategy(logger)
        logger.info("ShapAnalyzer initialized.")

    def analyze(self, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Executes SHAP analysis on provided models and data.

        Args:
            data: Dictionary containing:
                - 'models_metadata': Metadata for models (paths, types)
                - 'features_data': Feature matrix for explanation
            **kwargs:
                - 'tickers': List of tickers to analyze (optional)
        """
        models_meta = data.get('models_metadata', {})
        features_df = data.get('features_data')

        if not models_meta:
            logger.warning("No models_metadata found for SHAP analysis.")
            return {"status": "SKIPPED", "reason": "No models_metadata"}

        if features_df is None or features_df.empty:
            logger.warning("No features_data found for SHAP analysis.")
            return {"status": "SKIPPED", "reason": "No features_data"}

        tickers = kwargs.get('tickers', list({m.get('ticker') for m in models_meta.values() if m.get('ticker')}))

        results = {}
        for ticker in tickers:
            ticker_results = self._analyze_ticker_models(ticker, models_meta, features_df)
            if ticker_results:
                results[ticker] = ticker_results

        return {
            "status": "OK" if results else "EMPTY",
            "ticker_analysis": results,
            "summary": self._create_summary(results)
        }

    def _analyze_ticker_models(self, ticker: str, models_meta: dict[str, Any], features_df: pd.DataFrame) -> dict[str, Any]:
        """Analyzes all models for a specific ticker."""
        # Find champion model for ticker
        ticker_models = {cid: meta for cid, meta in models_meta.items() if meta.get('ticker') == ticker}
        if not ticker_models:
            return {}

        # For now, we'll explain the first one or a "winner" if marked
        winner_id = next((cid for cid, m in ticker_models.items() if m.get('is_winner')), list(ticker_models.keys())[0])
        meta = ticker_models[winner_id]

        model_path = meta.get('model_path')
        if not model_path:
            return {}

        try:
            # Load model
            model_metadata = {
                'model_path': model_path,
                'model_type': meta.get('model_type', 'lightgbm')
            }
            model = self.loader.load_model(model_metadata)
            if not model:
                return {}

            # Check for incompatible model types
            model_type = meta.get('model_type', '').lower()
            if 'autoencoder' in model_type:
                logger.warning(f"Skipping SHAP analysis for autoencoder model {winner_id} - not suitable for regression explanation")
                return {}

            # Check if model is TensorFlow Sequential without defined input shape
            native_model = getattr(model, 'model', model)
            if hasattr(native_model, '__class__'):
                model_class_name = native_model.__class__.__name__.lower()
                if 'sequential' in model_class_name:
                    logger.warning("TensorFlow Sequential model detected - SHAP may fail due to undefined input shape")
                    # Try to proceed but with warning

            # Prepare data
            feature_names = meta.get('selected_features', [])
            if not feature_names:
                # Fallback to features_df columns
                feature_names = [c for c in features_df.columns if not any(x in c.lower() for x in ['ticker', 'date', 'target', 'hash'])]

            # Validate and clean data
            X = features_df[feature_names].tail(100) # Use recent data for global importance

            # Check for non-numeric data
            non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"Non-numeric columns found: {non_numeric_cols}. Converting to numeric where possible.")
                for col in non_numeric_cols:
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numeric. Dropping it. Error: {e}")
                        X = X.drop(columns=[col])
                        feature_names = [f for f in feature_names if f != col]

            # Drop any remaining NaN values
            X = X.dropna()
            if X.empty:
                logger.warning("No valid numeric data available for SHAP analysis after cleaning.")
                return {}

            # Calculate Global Importance
            global_importance = ExplainabilityCalculator.analyze_feature_importance(model, X, feature_names)

            # Calculate Local Explanation for the last point
            last_row = X.tail(1)
            local_explanation = ExplainabilityCalculator.explain_single_prediction(model, last_row)

            return {
                "model_id": winner_id,
                "model_type": meta.get('model_type'),
                "global_importance": global_importance,
                "local_explanation": local_explanation,
                "top_features": list(global_importance.keys())[:5]
            }

        except Exception as e:
            logger.error(f"SHAP analysis failed for {ticker}: {e}")
            return {}

    def _create_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Aggregates findings across tickers."""
        if not results:
            return {}

        all_features: dict[str, float] = {}
        for _ticker, res in results.items():
            for feat, imp in res.get('global_importance', {}).items():
                all_features[feat] = all_features.get(feat, 0) + imp

        # Normalize aggregate importance
        total = sum(all_features.values())
        if total > 0:
            all_features = {k: v / total for k, v in all_features.items()}

        return {
            "aggregate_importance": dict(sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:10]),
            "analyzed_tickers": list(results.keys()),
            "model_types_explained": list({r.get('model_type') for r in results.values()})
        }
