# src/predictions/models_predict.py

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, cast
from src.core.logging.logger import ProjectLogger
from src.ensembling.stacked_ensemble import ensemble_forecast
from .deep_predict import predict_lstm, predict_cnn, predict_transformer, predict_autoencoder
from .sentiment_integration import get_sentiment_integrator

logger = ProjectLogger.get_logger(__name__)

# --------------------
# Safe inverse transform
# --------------------
def safe_inverse_transform(scaler, y_pred: np.ndarray) -> np.ndarray:
    """Inverse transform with NaN-safe fallback."""
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    except Exception:
        logger.warning("Failed to inverse_transform, returning original values.")
        return y_pred

# --------------------
# Classic ML Models
# --------------------
def predict_ml(model: Any, X: np.ndarray) -> np.ndarray:
    """Predictions for classic ML models (with predict_proba support)."""
    x_safe = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(x_safe)[:, 1]
    else:
        y_pred = model.predict(x_safe)
    return np.asarray(y_pred).reshape(-1)

# --------------------
# Universal Router
# --------------------
def predict_any(model: Any, X: np.ndarray, model_type: str) -> np.ndarray:
    """Selects the correct prediction function based on model type."""
    try:
        if "lstm" in model_type:
            return predict_lstm(model, X)
        elif "cnn" in model_type:
            return predict_cnn(model, X)
        elif "transformer" in model_type:
            return predict_transformer(model, X)
        elif "autoencoder" in model_type:
            return predict_autoencoder(model, X)
        else:
            return predict_ml(model, X)
    except Exception as e:
        logger.exception(f"[ERROR] Error predicting for {model_type}: {e}")
        return np.array([])

# --------------------
# Get Predictions from All Models + Ensemble
# --------------------
def get_predictions(
    models_dict: Dict[str, Any],
    df_features: pd.DataFrame,
    target_scaler=None,
    ensemble_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Get predictions from all models with safe inverse transform and ensembling."""
    X = df_features.values
    preds = {}

    for name, model in models_dict.items():
        y_pred = predict_any(model, X, model_type=name.lower())
        if y_pred.size == 0:
            logger.warning(f"[WARN] Prediction for {name} is empty. Skipped.")
            continue

        if target_scaler is not None:
            y_pred = safe_inverse_transform(target_scaler, y_pred)

        preds[name] = y_pred
        logger.info(f"[OK] Prediction for {name} ready ({y_pred.shape[0]} points).")

    # --- Ensemble ---
    if preds:
        ensemble_result = ensemble_forecast(
            model_predictions=preds,
            weights=ensemble_weights,
            rolling_window=3
        )
        preds["ensemble"] = ensemble_result.final_signal
        preds["ensemble_stats"] = ensemble_result.stats
        logger.info(f"[DATA] Ensemble forecast ready ({len(ensemble_result.final_signal)} points).")
    else:
        logger.warning("[WARN] No predictions available for ensemble.")

    return preds

# --------------------
# Load Models and Predict with Parquet
# --------------------
def predict_from_parquet(parquet_path: str, models_path: str = "data/trained_models") -> Dict[str, Any]:
    """Full prediction based on final_features.parquet."""
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_file)
    df = df.drop(columns=["date", "ticker", "scope", "target"], errors="ignore")

    models_dir = Path(models_path)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    models_dict = {}
    target_scaler = None

    for f in model_files:
        model = joblib.load(os.path.join(models_path, f))
        if "scaler" in f.lower():
            target_scaler = model
            continue
        models_dict[f.replace(".pkl", "")] = model

    logger.info(f"Loaded {len(models_dict)} models.")

    ensemble_result = get_predictions(models_dict, df, target_scaler=target_scaler)
    return ensemble_result

# --------------------
# Sentiment Models Prediction
# --------------------
def predict_sentiment_models(news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
    """Prediction using sentiment models."""
    try:
        sentiment_integrator = get_sentiment_integrator()
        sentiment_signal = sentiment_integrator.get_sentiment_signal(news_data, price_data)
        
        signal_result = {
            'signal_type': sentiment_signal.get('signal_type', 'hold'),
            'signal_strength': float(sentiment_signal.get('signal_strength', 0.0)),
            'confidence': float(sentiment_signal.get('confidence', 0.0)),
            'reasoning': sentiment_signal.get('reasoning', 'No reasoning provided'),
            'model_type': 'sentiment'
        }
        
        logger.info(f"[SENTIMENT] Signal: {signal_result['signal_type']} (confidence: {signal_result['confidence']:.2f})")
        
        return signal_result
        
    except Exception as e:
        logger.error(f"[SENTIMENT] Prediction error: {e}")
        return {
            'signal_type': 'hold',
            'signal_strength': 0.0,
            'confidence': 0.0,
            'reasoning': f'Sentiment analysis error: {str(e)}',
            'model_type': 'sentiment_error'
        }

def convert_sentiment_signal_to_numeric(signal: str) -> float:
    """Converts sentiment signal string to numeric prediction value."""
    signal_map = {
        'buy': 0.015,      # +1.5% expected change
        'sell': -0.015,    # -1.5% expected change
        'hold': 0.0,       # 0% expected change
        'strong_buy': 0.025,    # +2.5% expected change
        'strong_sell': -0.025   # -2.5% expected change
    }
    
    return signal_map.get(signal.lower(), 0.0)
