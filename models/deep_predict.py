# models/deep_predict.py

import numpy as np
import torch
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# --------------------
# LSTM
# --------------------
def predict_lstm(model, X, time_steps=10, batch_size=64, return_proba=False):
    """LSTM inference with батчингом, CPU/GPU and dtype support"""
    if X.shape[0] < time_steps:
        logger.warning(f"LSTM пропущено: notдосandтньо data ({X.shape[0]} < {time_steps})")
        return np.array([])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_seq = np.array([X[i:(i + time_steps)] for i in range(X.shape[0] - time_steps + 1)], dtype=np.float32)
    preds_list = []
    param_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            end = start + batch_size
            batch = torch.from_numpy(X_seq[start:end]).to(device=device, dtype=param_dtype)
            batch_pred = model(batch).cpu().numpy()
            preds_list.append(batch_pred)

    y_pred = np.concatenate(preds_list, axis=0)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if not return_proba and y_pred.ndim == 1:
        y_pred = (y_pred >= 0.5).astype(int)

    logger.info(f"[OK] LSTM прогноwith виконано ({y_pred.shape[0]} точок).")
    return y_pred

# --------------------
# CNN
# --------------------
def predict_cnn(model, X, return_proba=False):
    """CNN inference for часових вandкон"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    if not return_proba and preds.ndim == 1:
        preds = (preds >= 0.5).astype(int)
    logger.info(f"[OK] CNN прогноwith виконано ({preds.shape[0]} точок).")
    return preds

# --------------------
# Transformer
# --------------------
def predict_transformer(model, X, return_proba=False):
    """Transformer inference for часових вandкон"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    if not return_proba and preds.ndim == 1:
        preds = (preds >= 0.5).astype(int)
    logger.info(f"[OK] Transformer прогноwith виконано ({preds.shape[0]} точок).")
    return preds

# --------------------
# Autoencoder
# --------------------
def predict_autoencoder(model, X, return_proba=False):
    """Autoencoder inference for реконструкцandї and аномалandй"""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    # Для аномалandй can брати помилку реконструкцandї
    if hasattr(model, "reconstruction_error"):
        preds = model.reconstruction_error(X)
    logger.info(f"[OK] Autoencoder прогноwith виконано ({preds.shape[0]} точок).")
    return preds