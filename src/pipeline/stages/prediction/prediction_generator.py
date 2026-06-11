"""
PredictionGenerator: ensemble/single-model prediction and denormalization
extracted from PredictionStage to reduce file size.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger


class SanityFilter:
    """Clamps extreme predictions to realistic bounds for all tickers."""
    @staticmethod
    def clamp(value: float, ticker: str) -> float:
        """
        Clamps value to realistic bounds based on ticker type.
        Detects common scaling errors where prices are returned instead of percentage returns.
        """
        ticker_up = ticker.upper()
        # Generic crypto detection (most crypto pairs end in USD or USDT)
        is_crypto = any(suffix in ticker_up for suffix in ['USD', 'ETH', 'BTC', 'SOL', 'DOGE', 'BNB', 'ADA'])

        # Define realistic bounds for percentage returns (e.g. 15% for stocks, 60% for crypto)
        limit = 0.6 if is_crypto else 0.15

        # 1. Handle price-vs-return confusion
        # If the value is > 1000, it's almost certainly a raw price (e.g., AAPL at 170.0 or BTC at 60000.0)
        # If the value is between 2.0 and 1000, it's an unrealistic return (>200%)
        if abs(value) > limit:
            if abs(value) > 2.0:
                 # If it looks like a price or a completely broken scaler, mute it
                 # In a real system, we'd log this for a dev to fix the scaler mapping
                 return 0.0
            else:
                 # If it's just a very strong signal (e.g. 18%), clamp it to our safety limit
                 return float(np.clip(value, -limit, limit))

        return value

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
        models: dict[str, Any],
        best_model_name: str,
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: list[str],
        market_regime: str,
        context_id: str,
    ) -> tuple[float | None, dict[str, float], dict[str, Any]]:
        """
        Route to ensemble or single-model prediction.
        Returns: (prediction, contributions, shap_explanations)
        """
        if len(models) > 1:
            prediction, contributions = self.generate_ensemble_prediction(
                models, ticker_df_clean, filtered_features_list, market_regime, context_id
            )
        else:
            prediction, contributions = self.generate_single_model_prediction(
                models, best_model_name, ticker_df_clean, filtered_features_list
            )

        # Calculate SHAP explanation for the primary model (now enabled)
        shap_explanations = self._calculate_shap_explanations(
            models.get(best_model_name, list(models.values())[0]),
            ticker_df_clean,
            filtered_features_list
        )

        return prediction, contributions, shap_explanations

    def _calculate_shap_explanations(self, model: Any, X: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
        """Calculates SHAP values for the prediction (re-enabled with safety)."""
        try:
            # Only import when needed to avoid dependency issues if shap is not installed
            import shap
            
            # Use last row for explanation
            last_row = X.tail(1)
            
            # Simple model check for TreeExplainer vs KernelExplainer
            if hasattr(model, 'feature_importances_') or 'Tree' in str(type(model)):
                explainer = shap.TreeExplainer(model)
            else:
                # Use a small background sample for non-tree models
                explainer = shap.Explainer(model, X.head(10))
                
            shap_values = explainer(last_row)
            
            # Format results
            return {
                'values': shap_values.values.tolist()[0],
                'base_value': float(shap_values.base_values[0]),
                'feature_names': feature_names
            }
        except Exception as e:
            self.logger.debug(f"SHAP explanation failed: {e}")
            return {}

    def _align_features(self, model: Any, X: pd.DataFrame, filtered_features_list: list[str]) -> pd.DataFrame:
        """
        Robustly align input DataFrame X to the features expected by the model.
        Supports both scikit-learn (via feature_names_in_) and Keras (via metadata list).
        """
        expected_features = None

        # Unpack the Keras wrapper to inspect the underlying model if needed
        unwrapped_model = model.model if hasattr(model, 'model') else model

        # 1. Check if it's a scikit-learn/other model with feature_names_in_
        if hasattr(unwrapped_model, 'feature_names_in_'):
            expected_features = list(unwrapped_model.feature_names_in_)
            self.logger.debug(f"📊 Aligned via feature_names_in_: {len(expected_features)} features")

        # 2. Fall back to filtered_features_list if available
        if not expected_features and filtered_features_list:
            expected_features = filtered_features_list
            self.logger.debug(f"📊 Aligned via filtered_features_list: {len(expected_features)} features")

        if not expected_features:
            # Fall back to passing X as is if we have no expected features
            return X

        # Reconstruct DataFrame with expected features in correct order
        X_aligned = pd.DataFrame(index=X.index)
        missing_count = 0
        for col in expected_features:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = 0.0
                missing_count += 1

        if missing_count > 0:
            self.logger.warning(f"⚠️ Filled {missing_count} missing features with 0.0 for model prediction: {expected_features if len(expected_features) <= 5 else expected_features[:5]}")

        return X_aligned

    def generate_ensemble_prediction(
        self,
        models: dict[str, Any],
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: list[str],
        market_regime: str,
        context_id: str,
    ) -> tuple[float | None, dict[str, float]]:
        """Generate ensemble prediction from multiple models."""
        model_preds: dict[str, Any] = {}

        for m_name, m_inst in models.items():
            if 'autoencoder' in m_name.lower():
                self.logger.debug("   ⏭️ Skipping autoencoder (used only for anomaly detection)")
                continue

            # Align features for each model in the ensemble dynamically
            model_features = self._align_features(m_inst, ticker_df_clean, filtered_features_list)
            self.logger.debug(f"   {m_name}: X shape={model_features.shape}, features={model_features.shape[1]}")

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
        models: dict[str, Any],
        best_model_name: str,
        ticker_df_clean: pd.DataFrame,
        filtered_features_list: list[str],
    ) -> tuple[float | None, dict[str, float]]:
        """Generate prediction from a single selected model."""
        selected_model = models.get(best_model_name, list(models.values())[0])
        if 'autoencoder' in best_model_name.lower():
            self.logger.info(f"🔄 Calculating reconstruction MSE for Autoencoder context: {best_model_name}")
            try:
                X = self._align_features(selected_model, ticker_df_clean, filtered_features_list)
                raw_reconstruction = selected_model.predict(X)
                x_input_flat = X.iloc[-1:].values.flatten()
                reconstruction_flat = raw_reconstruction.flatten()
                min_len = min(len(x_input_flat), len(reconstruction_flat))
                mse = float(np.mean((x_input_flat[:min_len] - reconstruction_flat[:min_len]) ** 2))
                self.logger.info(f"✅ Autoencoder reconstruction MSE: {mse:.6f}")
                return mse, {best_model_name: mse}
            except Exception as e:
                self.logger.error(f"❌ Autoencoder calculation failed: {e}")
                return None, {}

        # Align X with model expectations (handles feature name/count mismatches)
        X = self._align_features(selected_model, ticker_df_clean, filtered_features_list)
        self.logger.debug(f"   {best_model_name}: X shape={X.shape}, features={X.shape[1]}")

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

    def denormalize_prediction(self, adjusted_prediction: Any, target_scaler: Any, ticker: str = "unknown") -> float:
        """Denormalize prediction using target scaler."""
        if target_scaler is None:
            result = float(self.extract_prediction_value(adjusted_prediction))
            if np.isnan(result):
                self.logger.warning("⚠️ Adjusted prediction resulted in NaN, defaulting to 0.0")
                return 0.0
            
            # ✅ ENHANCED: Apply sanity filter even if no scaler is present
            result = SanityFilter.clamp(result, ticker)
            return result
        try:
            pred_val = self.extract_prediction_value(adjusted_prediction)
            pred_to_denorm = np.array([[pred_val]])

            if hasattr(target_scaler, 'scale_') and target_scaler.scale_.shape[0] != 1:
                raise ValueError(
                    f"Scaler has wrong number of features: {target_scaler.scale_.shape[0]} instead of 1"
                )

            denormalized = target_scaler.inverse_transform(pred_to_denorm)
            result = float(denormalized.flatten()[-1])

            if np.isnan(result):
                self.logger.warning("⚠️ Denormalized prediction resulted in NaN, defaulting to 0.0")
                return 0.0

            # Apply sanity filter
            result = SanityFilter.clamp(result, ticker)

            self.logger.info(f"✅ Denormalized prediction: {result:.6f}")
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to denormalize prediction: {e}")
            result = float(self.extract_prediction_value(adjusted_prediction))
            return SanityFilter.clamp(result, ticker)

    def extract_prediction_value(self, adjusted_prediction: Any) -> float:
        """Extract scalar prediction value from various prediction formats."""
        if hasattr(adjusted_prediction, '__len__') and len(adjusted_prediction) > 0:
            return (
                adjusted_prediction[-1]
                if hasattr(adjusted_prediction, '__getitem__')
                else float(adjusted_prediction)
            )
        return float(adjusted_prediction)
