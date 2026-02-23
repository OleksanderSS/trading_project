# utils/metrics.py

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from utils.system_monitor import SystemMonitor
from utils.ensemble import ensemble_forecast
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("TradingProjectLogger")

def infer_task_type(y_true, y_pred) -> str:
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1 and set(np.unique(y_pred)).issubset({0, 1}):
        return "classification"
    elif y_pred.ndim == 1 and np.issubdtype(y_pred.dtype, np.floating):
        return "regression"
    elif y_pred.ndim == 1 and np.all((0 <= y_pred) & (y_pred <= 1)):
        return "probabilistic"
    logger.warning("[metrics] Невandдомий тип forдачand")
    return "unknown"

def calculate_all_metrics(y_true,
    y_pred,
    task_type: Optional[str] = None,
    returns: Optional[np.ndarray] = None) -> Dict[str,
    float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if task_type is None:
        task_type = infer_task_type(y_true, y_pred)
        logger.info(f"[metrics] [DATA] Тип forдачand: {task_type}, валandдних спостережень: {len(y_true)}")

    metrics = {}
    if task_type == "regression":
        metrics["MAE"] = mean_absolute_error(y_true, y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["R2"] = r2_score(y_true, y_pred)
        if returns is not None and len(returns) == len(y_pred):
            returns = np.asarray(returns)[mask]
            metrics["Sharpe"] = returns.mean() / returns.std() if returns.std() > 0 else np.nan
    elif task_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Precision"] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["Recall"] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics["F1"] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    elif task_type == "probabilistic":
        metrics["ROC AUC"] = roc_auc_score(y_true, y_pred)
        metrics["Log Loss"] = log_loss(y_true, y_pred)
    else:
        logger.warning("[metrics] Метрики not роwithрахованand  notвandдомий тип forдачand")
    return metrics

# --------------------------
# Спрощенand регресandйнand метрики
# --------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return calculate_all_metrics(y_true, y_pred, task_type="regression")

# --------------------------
# Витяг основних метрик
# --------------------------
def extract_core_metrics(metrics: Dict[str, Any], model_name: str = "ensemble") -> Dict[str, float]:
    m = metrics.get(model_name, {})
    return {
        "mae": m.get("MAE", m.get("mae", np.nan)),  # тепер дивиться and на "mae"
        "rmse": m.get("RMSE", np.nan),
        "r2": m.get("R2", np.nan),
        "sharpe": m.get("Sharpe", np.nan),
        "f1": m.get("F1", m.get("f1", np.nan))  # пandдтримка класифandкацandї
    }

# --------------------------
# Evaluate models + ensemble
# --------------------------
def evaluate_models_ensemble(models: Optional[Dict[str, Any]],
                             X_data_scaled: np.ndarray,
                             y_true_scaled: np.ndarray,
                             y_true_original: np.ndarray,
                             target_scaler: Optional[Any] = None,
                             ensemble_weights: Optional[Dict[str, float]] = None,
                             interval: str = "",
                             output_path: str = "") -> Dict[str, Any]:
    logger.info(f"[metrics]  Початок evaluate_models_ensemble for andнтервалу: {interval}")
    results = {"predictions": {}, "metrics": {}}

    if not models:
        logger.warning(f"[metrics] models=None or пустий словник for andнтервалу {interval}. Поверandємо NaN")
        results["predictions"]["ensemble"] = np.zeros(len(y_true_original))
        results["metrics"]["ensemble"] = {
            "MAE": np.nan, "mae": np.nan,
            "RMSE": np.nan, "R2": np.nan, "Sharpe": np.nan,
            "F1": np.nan, "f1": np.nan,
            "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan
        }
        return results

    X_safe = np.nan_to_num(X_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_true_safe = y_true_original.flatten()

    with SystemMonitor(f"Evaluate models + ensemble ({interval})", auto_gc=True):
        for name, model in models.items():
            if model is None:
                logger.warning(f"Моwhereль {name} None, пропускаємо")
                results["predictions"][name] = np.zeros(len(y_true_original))
                results["metrics"][name] = {
                    "MAE": np.nan, "mae": np.nan,
                    "RMSE": np.nan, "R2": np.nan,
                    "F1": np.nan, "f1": np.nan,
                    "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan
                }
                continue
            try:
                y_pred_scaled = model.predict(X_safe)
                y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1)

                if target_scaler:
                    try:
                        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    except Exception:
                        y_pred = y_pred_scaled
                else:
                    y_pred = y_pred_scaled

                if len(y_pred) != len(y_true_original):
                    if len(y_pred) > len(y_true_original):
                        y_pred = y_pred[:len(y_true_original)]
                    else:
                        y_pred = np.pad(y_pred, (0, len(y_true_original) - len(y_pred)), constant_values=np.nan)

                results["predictions"][name] = y_pred
                results["metrics"][name] = calculate_all_metrics(
                    y_true_safe, y_pred,
                    task_type=infer_task_type(y_true_safe, y_pred)
                )
                logger.info(f"[metrics] [OK] Моwhereль {name}  прогноwith forвершено")
            except Exception:
                logger.exception(f"[metrics] Error моwhereлand {name}")
                results["predictions"][name] = np.zeros(len(y_true_original))
                results["metrics"][name] = {
                    "MAE": np.nan, "mae": np.nan,
                    "RMSE": np.nan, "R2": np.nan,
                    "F1": np.nan, "f1": np.nan,
                    "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan
                }

        try:
            ensemble_pred, stats = ensemble_forecast(
                model_predictions=results["predictions"],
                weights=ensemble_weights,
                normalize_weights=True,
                return_stats=True
            )
            ensemble_pred = np.asarray(ensemble_pred)

            if len(ensemble_pred) != len(y_true_original):
                if len(ensemble_pred) > len(y_true_original):
                    ensemble_pred = ensemble_pred[:len(y_true_original)]
                else:
                    ensemble_pred = np.pad(ensemble_pred, (0, len(y_true_original) - len(ensemble_pred)),
                                           constant_values=np.nan)

            results["predictions"]["ensemble"] = ensemble_pred
            results["metrics"]["ensemble"] = calculate_all_metrics(
                y_true_safe, ensemble_pred,
                task_type=infer_task_type(y_true_safe, ensemble_pred)
            )

            if stats:
                logger.info(f"[DATA] Ensemble stats: mean_std={stats.get('mean_std', 0):.4f}, "
                            f"min_std={stats.get('min_std', 0):.4f}, max_std={stats.get('max_std', 0):.4f}, "
                            f"n_models_used={stats.get('n_models_used', 0)}")
        except Exception:
            logger.exception("[metrics] Error формування ансамблю")
            results["predictions"]["ensemble"] = np.zeros(len(y_true_original))
            results["metrics"]["ensemble"] = {
                "MAE": np.nan, "mae": np.nan,
                "RMSE": np.nan, "R2": np.nan, "Sharpe": np.nan,
                "F1": np.nan, "f1": np.nan,
                "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan
            }

    return results