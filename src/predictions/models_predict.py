# src/predictions/models_predict.py

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any
from src.core.logging.logger import ProjectLogger
from src.ensembling.ensemble import ensemble_forecast
from .deep_predict import predict_lstm, predict_cnn, predict_transformer, predict_autoencoder
from .dean_integration import get_dean_integrator
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
    ensemble_weights: Dict[str, float] = None
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
            normalize_weights=True,
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
# Dean Models Prediction
# --------------------
def predict_dean_models(data: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
    """Prediction using Dean RL models."""
    try:
        dean_integrator = get_dean_integrator()
        dean_decision = dean_integrator.get_trading_decision(data, ticker, timeframe)
        
        logger.info(f"[DEAN] Prediction for {ticker}: {dean_decision['type']} (confidence: {dean_decision['confidence']:.2f})")
        
        return {
            'dean_prediction': dean_decision,
            'model_type': 'dean_ensemble',
            'confidence': dean_decision.get('confidence', 0.0),
            'recommendation': dean_decision.get('type', 'hold'),
            'reasoning': dean_decision.get('reasoning', ''),
            'risk_level': dean_decision.get('risk_level', 'medium')
        }
        
    except Exception as e:
        logger.error(f"[DEAN] Prediction error: {e}")
        return {
            'dean_prediction': None,
            'model_type': 'dean_error',
            'confidence': 0.0,
            'recommendation': 'hold',
            'reasoning': f'Dean models error: {str(e)}',
            'risk_level': 'low'
        }

# --------------------
# Enhanced Prediction with All Models
# --------------------
def predict_all_models_enhanced(
    models_dict: Dict[str, Any],
    df_features: pd.DataFrame,
    target_scaler=None,
    ensemble_weights: Dict[str, float] = None,
    use_dean=True,
    ticker: str = "UNKNOWN",
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """Enhanced prediction with all models including Dean."""
    
    # Traditional predictions
    predictions = get_predictions(models_dict, df_features, target_scaler, ensemble_weights)
    
    # Dean prediction
    if use_dean:
        try:
            dean_prediction = predict_dean_models(df_features, ticker, timeframe)
            predictions['dean'] = dean_prediction
            
            # Add Dean to ensemble with weight 0.2 if confidence is high
            if 'ensemble' in predictions and dean_prediction['confidence'] > 0.3:
                ensemble_preds = predictions['ensemble']
                dean_weight = dean_prediction['confidence'] * 0.2
                
                # Convert Dean decision to numeric prediction
                dean_numeric = convert_dean_decision_to_numeric(dean_prediction['recommendation'])
                
                if len(ensemble_preds) > 0:
                    enhanced_ensemble = ensemble_preds * (1 - dean_weight) + dean_numeric * dean_weight
                    predictions['ensemble_enhanced'] = enhanced_ensemble
                    predictions['ensemble_weights'] = {
                        'traditional': 1 - dean_weight,
                        'dean': dean_weight
                    }
                    
        except Exception as e:
            logger.warning(f"[WARN] Dean prediction failed: {e}")
    
    return predictions

def convert_dean_decision_to_numeric(decision: str) -> float:
    """Converts Dean decision string to numeric prediction value."""
    decision_map = {
        'buy': 0.02,      # +2% expected change
        'sell': -0.02,    # -2% expected change
        'hold': 0.0,      # 0% expected change
        'wait': 0.0,      # 0% expected change
        'reduce_position': -0.01,  # -1% expected change
        'increase_position': 0.01   # +1% expected change
    }
    
    return decision_map.get(decision.lower(), 0.0)

# --------------------
# Sentiment Models Prediction
# --------------------
def predict_sentiment_models(news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
    """Prediction using sentiment models."""
    try:
        sentiment_integrator = get_sentiment_integrator()
        sentiment_signal = sentiment_integrator.get_sentiment_signal(news_data, price_data)
        
        logger.info(f"[SENTIMENT] Signal: {sentiment_signal['signal_type']} (confidence: {sentiment_signal['confidence']:.2f})")
        
        return sentiment_signal
        
    except Exception as e:
        logger.error(f"[SENTIMENT] Prediction error: {e}")
        return {
            'signal_type': 'hold',
            'signal_strength': 0.0,
            'confidence': 0.0,
            'reasoning': f'Sentiment analysis error: {str(e)}',
            'model_type': 'sentiment_error'
        }

# --------------------
# Final Enhanced Prediction with All Models
# --------------------
def predict_all_models_final(
    models_dict: Dict[str, Any],
    df_features: pd.DataFrame,
    target_scaler=None,
    ensemble_weights: Dict[str, float] = None,
    use_dean=True,
    use_sentiment=True,
    ticker: str = "UNKNOWN",
    timeframe: str = "1h",
    news_data: pd.DataFrame = None,
    price_data: pd.DataFrame = None
) -> Dict[str, Any]:
    """Final prediction with all models integrated."""
    
    # Traditional + Dean predictions
    predictions = predict_all_models_enhanced(
        models_dict, df_features, target_scaler, ensemble_weights,
        use_dean, ticker, timeframe
    )
    
    # Sentiment prediction
    if use_sentiment and news_data is not None and price_data is not None:
        try:
            sentiment_prediction = predict_sentiment_models(news_data, price_data)
            predictions['sentiment'] = sentiment_prediction
            
            # Integrate sentiment into final ensemble
            if 'ensemble_enhanced' in predictions and sentiment_prediction['confidence'] > 0.3:
                ensemble_preds = predictions['ensemble_enhanced']
                sentiment_weight = sentiment_prediction['confidence'] * 0.15  # 15% weight for sentiment
                
                # Convert sentiment signal to numeric prediction
                sentiment_numeric = convert_sentiment_signal_to_numeric(sentiment_prediction['signal_type'])
                
                if len(ensemble_preds) > 0:
                    final_ensemble = ensemble_preds * (1 - sentiment_weight) + sentiment_numeric * sentiment_weight
                    predictions['final_ensemble'] = final_ensemble
                    predictions['final_weights'] = {
                        'traditional': predictions.get('ensemble_weights', {}).get('traditional', 0.8),
                        'dean': predictions.get('ensemble_weights', {}).get('dean', 0.0),
                        'sentiment': sentiment_weight
                    }
                    
        except Exception as e:
            logger.warning(f"[WARN] Sentiment prediction failed: {e}")
    
    return predictions

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
