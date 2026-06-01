# src/predictions/deep_predict.py

import numpy as np
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# --------------------
# LSTM
# --------------------
def predict_lstm(model, X, time_steps=10, batch_size=64):
    """LSTM inference with batching, CPU/GPU, and dtype support."""
    # Lazy import: avoids importing torch for pipelines that never use deep models.
    import torch
    if X.shape[0] < time_steps:
        logger.warning(f"LSTM skipped: insufficient data ({X.shape[0]} < {time_steps})")
        return np.array([])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    x_seq = np.array([X[i:(i + time_steps)] for i in range(X.shape[0] - time_steps + 1)], dtype=np.float32)
    preds_list = []
    param_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for start in range(0, len(x_seq), batch_size):
            end = start + batch_size
            batch = torch.from_numpy(x_seq[start:end]).to(device=device, dtype=param_dtype)
            batch_pred = model(batch).cpu().numpy()
            preds_list.append(batch_pred)

    y_pred = np.concatenate(preds_list, axis=0)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if y_pred.ndim == 1:
        y_pred = (y_pred >= 0.5).astype(int)

    logger.info(f"[OK] LSTM prediction complete ({y_pred.shape[0]} points).")
    return y_pred

# --------------------
# CNN
# --------------------
def predict_cnn(model, X):
    """CNN inference for time windows."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    if preds.ndim == 1:
        preds = (preds >= 0.5).astype(int)
    logger.info(f"[OK] CNN prediction complete ({preds.shape[0]} points).")
    return preds

# --------------------
# Transformer
# --------------------
def predict_transformer(model, X):
    """Transformer inference for time windows."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    if preds.ndim == 1:
        preds = (preds >= 0.5).astype(int)
    logger.info(f"[OK] Transformer prediction complete ({preds.shape[0]} points).")
    return preds

# --------------------
# Autoencoder
# --------------------
def predict_autoencoder(model, X):
    """Autoencoder inference for reconstruction and anomalies."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    preds = model.predict(X)
    # For anomalies, reconstruction error can be used
    if hasattr(model, "reconstruction_error"):
        preds = model.reconstruction_error(X)
    logger.info(f"[OK] Autoencoder prediction complete ({preds.shape[0]} points).")
    return preds
