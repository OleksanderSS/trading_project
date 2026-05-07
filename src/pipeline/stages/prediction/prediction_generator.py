"""
PredictionGenerator: ensemble/single-model prediction and denormalization
extracted from PredictionStage to reduce file size.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class PredictionGenerator:
    """Generates ensemble or single-model predictions and denormalizes them."""

    def __init__(self, ensemble_factory: Any, ensemble_cache: Any, adjuster: Any):
        self.logger = ProjectLogger.get_logger("PredictionGenerator")
        self.ensemble_factory = ensemble_factory
        self.ensemble_cache = ensemble_cache
        self.adjuster = adjuster

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_prediction(
        self,
        models: Dict[str, Any],
        best_model_name: str,
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: List[str],
        market_regime: str,
        context_id: str,
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Route to ensemble or single-model prediction."""
        if len(models) > 1:
            return self.generate_ensemble_prediction(
                models, ticker_df_clean, filtered_features_list, market_regime, context_id
            )
        return self.generate_single_model_prediction(
            models, best_model_name, ticker_df_clean, filtered_features_list
        )

    def generate_ensemble_prediction(
        self,
        models: Dict[str, Any],
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: List[str],
        market_regime: str,
        context_id: str,
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Generate ensemble prediction from multiple models."""
        model_preds: Dict[str, Any] = {}

        for m_name, m_inst in models.items():
            feature_cols = filtered_features_list or ticker_df_clean.columns.tolist()
            model_features = (
                ticker_df_clean[feature_cols]
                if all(c in ticker_df_clean.columns for c in feature_cols)
                else ticker_df_clean
            )
            self.logger.debug(f"   {m_name}: X shape={model_features.shape}, features={len(feature_cols)}")

            if 'autoencoder' in m_name.lower():
                self.logger.debug("   ⏭️ Skipping autoencoder (used only for anomaly detection)")
                continue

            model_preds[m_name] = self.ensemble_cache.get_or_compute_model_prediction(
                features=model_features,
                model_id=m_name,
                model_fn=lambda features=model_features, model=m_inst: model.predict(features),
            )

        if not model_preds:
            self.logger.warning(f"⚠️ No models for prediction (only autoencoder), skipping {context_id}")
            return None, {}

        preds_df = pd.DataFrame(model_preds)
        ensemble_result = self.ensemble_factory.predict(
            X=preds_df,
            context_params={
                "ticker": ticker_df_clean.get('ticker', 'unknown'),
                "regime": market_regime,
            },
        )
        return ensemble_result.final_signal, ensemble_result.active_weights

    def generate_single_model_prediction(
        self,
        models: Dict[str, Any],
        best_model_name: str,
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: List[str],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Generate prediction from a single selected model."""
        selected_model = models.get(best_model_name, list(models.values())[0])
        if 'autoencoder' in best_model_name.lower():
            self.logger.warning("⚠️ Autoencoder not suitable for regression prediction")
            return None, {}

        feature_cols = filtered_features_list or ticker_df_clean.columns.tolist()
        X = (
            ticker_df_clean[feature_cols]
            if all(c in ticker_df_clean.columns for c in feature_cols)
            else ticker_df_clean
        )
        self.logger.debug(f"   {best_model_name}: X shape={X.shape}, features={len(feature_cols)}")

        raw_prediction = selected_model.predict(X)
        pred_value = raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction
        return raw_prediction, {best_model_name: pred_value}

    def adjust_prediction_contextually(
        self, raw_prediction: Any, best_model_name: str, market_regime: str, ticker: str
    ) -> float:
        """Adjust prediction based on market context."""
        pred_val = raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction
        adjustment_result = self.adjuster.analyze(
            data={
                'predictions': {best_model_name: pred_val},
                'market_regime': market_regime,
                'ticker': ticker,
            }
        )
        result = adjustment_result.get('enhanced_predictions', {}).get(best_model_name, raw_prediction)
        return float(result) if result is not None else float(raw_prediction)

    def denormalize_prediction(self, adjusted_prediction: Any, target_scaler: Any) -> float:
        """Denormalize prediction using target scaler."""
        if target_scaler is None:
            return float(adjusted_prediction)
        try:
            if isinstance(adjusted_prediction, np.ndarray):
                pred_to_denorm = (
                    adjusted_prediction[-1:].reshape(-1, 1)
                    if adjusted_prediction.ndim == 1
                    else adjusted_prediction.reshape(-1, 1)
                )
            else:
                pred_to_denorm = np.array([[adjusted_prediction]])

            if hasattr(target_scaler, 'scale_') and target_scaler.scale_.shape[0] != 1:
                raise ValueError(
                    f"Scaler has wrong number of features: {target_scaler.scale_.shape[0]} instead of 1"
                )

            denormalized = target_scaler.inverse_transform(pred_to_denorm)
            result = float(denormalized.flatten()[-1])
            self.logger.info(f"✅ Denormalized prediction: {result:.6f}")
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to denormalize prediction: {e}")
            return float(adjusted_prediction)

    def extract_prediction_value(self, adjusted_prediction: Any) -> float:
        """Extract scalar prediction value from various prediction formats."""
        if hasattr(adjusted_prediction, '__len__') and len(adjusted_prediction) > 0:
            return (
                adjusted_prediction[-1]
                if hasattr(adjusted_prediction, '__getitem__')
                else float(adjusted_prediction)
            )
        return float(adjusted_prediction)
