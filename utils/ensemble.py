# utils/ensemble.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def ensemble_forecast(
    model_predictions: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None,
    normalize_weights: bool = True,
    max_weight: float = 0.7,
    min_weight: float = 0.0,
    return_stats: bool = False,
    fill_na: Union[float, None] = None,
    rolling_window: Optional[int] = None,
    pad_value: Union[float, None] = np.nan,
    method: str = "weighted"
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Ансамблевий прогноwith:
    - вирandвнювання серandй
    - контроль NaN
    - обмеження ваг
    - rolling_window for withгладження
    - методи: weighted (for forмовчуванням), mean, median
    """

    if not model_predictions:
        logger.warning("[Ensemble] Не передано жодного прогноwithу")
        return np.array([]), None

    # --- Всandновлюємо ваги ---
    full_weights = {}
    for m in model_predictions.keys():
        w = weights[m] if weights and m in weights else 1.0
        w = max(min_weight, min(w, max_weight))
        full_weights[m] = w

    if normalize_weights:
        total_weight = sum(full_weights.values())
        if total_weight > 0:
            full_weights = {k: w / total_weight for k, w in full_weights.items()}

    logger.info(f"[Ensemble] Викорисandнand ваги: {full_weights}")

    max_len = max((len(v) for v in model_predictions.values()), default=0)
    if max_len == 0:
        logger.warning("[Ensemble] Усand переданand прогноwithи порожнand")
        return np.array([]), None

    aligned_preds = []
    for model_name, preds in model_predictions.items():
        preds_arr = np.array(preds, dtype=float)
        if len(preds_arr) < max_len:
            preds_arr = np.pad(preds_arr, (max_len - len(preds_arr), 0), constant_values=pad_value)
        aligned_preds.append((model_name, preds_arr))

    # --- Формуємо фandнальний прогноwith ---
    if method == "mean":
        result = np.nanmean([arr for _, arr in aligned_preds], axis=0)
    elif method == "median":
        result = np.nanmedian([arr for _, arr in aligned_preds], axis=0)
    else:  # weighted
        weighted_sum = np.zeros(max_len, dtype=float)
        total_weights_arr = np.zeros(max_len, dtype=float)
        used_models = 0
        for model_name, preds_arr in aligned_preds:
            mask = ~np.isnan(preds_arr)
            if not mask.any():
                logger.warning(f"[Ensemble] Моwhereль {model_name} дала лише NaN  пропускаємо")
                continue
            w = full_weights.get(model_name, 1.0)
            weighted_sum[mask] += preds_arr[mask] * w
            total_weights_arr[mask] += w
            used_models += 1
        out = np.full_like(weighted_sum, fill_na if fill_na is not None else np.nan)
        result = np.divide(weighted_sum, total_weights_arr, out=out, where=total_weights_arr != 0)

    # --- Rolling усередnotння ---
    if rolling_window and rolling_window > 1:
        result = pd.Series(result).ffill().rolling(rolling_window, min_periods=1).mean().to_numpy()

    # --- Сandтистика ---
    stats = None
    if return_stats:
        stacked = np.vstack([arr for _, arr in aligned_preds])
        stds = np.nanstd(stacked, axis=0)
        stats = {
            "mean_std": float(np.nanmean(stds)),
            "min_std": float(np.nanmin(stds)),
            "max_std": float(np.nanmax(stds)),
            "n_models_used": len(aligned_preds)
        }
        logger.info(f"[Ensemble] Сandтистика: {stats}")

    return result, stats