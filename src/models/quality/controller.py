"""
ModelQualityController: Quality control for model predictions and drift detection

✅ PATTERN-AWARE QUALITY:
- Tracks baseline statistics separately per 'context_pattern_id'.
- Detects regime-specific drift (e.g. model works in Bull but drifts in Chaos).
"""
import logging

from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ModelQualityController:
    """
    Quality control for model predictions and drift detection.
    
    🎯 PATTERN-EXPERT TRACKING:
    - baseline_stats: Dict[f"{model_id}_{pattern_id}", statistics]
    """

    def __init__(self, drift_threshold: float = 0.3):
        self.drift_threshold = drift_threshold
        self.baseline_stats: dict[str, dict[str, Any]] = {}
        self.logger = ProjectLogger.get_logger(__name__)

    def validate_predictions(self, predictions: np.ndarray) -> bool:
        if np.any(np.isnan(predictions)):
            self.logger.warning("Predictions contain NaN values")
            return False
        if np.any(np.isinf(predictions)):
            self.logger.warning("Predictions contain Inf values")
            return False
        if np.any(np.abs(predictions) > 10):
            self.logger.warning(f"Predictions contain unrealistic values: max={np.max(np.abs(predictions)):.2f}")
            return False
        return True

    def check_drift(self, current: np.ndarray, baseline: np.ndarray) -> float:
        """
        Check drift between current and baseline distributions using standard standardization.
        """
        current_mean = np.mean(current)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
        if drift > self.drift_threshold:
            self.logger.warning(f"Drift detected: {drift:.3f} > {self.drift_threshold}")
        return float(drift)

    def check_drift_adaptive(self, current: np.ndarray, model_id: str, pattern_id: str = "normal") -> float:
        """
        🎯 PATTERN-AWARE DRIFT:
        Перевіряє дріфт саме для конкретного режиму.
        """
        key = f"{model_id}_{pattern_id}"
        baseline = self.baseline_stats.get(key)
        
        if not baseline:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"No baseline for {key}. Initializing now.")
            self.update_baseline(model_id, current, pattern_id)
            return 0.0

        current_mean = np.mean(current)
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']

        drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)

        if drift > self.drift_threshold:
            self.logger.warning(
                f"🚨 Regime Drift in {pattern_id} for {model_id}: {drift:.3f} > {self.drift_threshold}"
            )
        return float(drift)

    def get_quality_score(
        self,
        ensemble_pred: float,
        predictions: dict[str, float],
        weights: dict[str, float]
    ) -> float:
        pred_values = list(predictions.values())
        variance = np.var(pred_values)
        agreement = 1.0 / (1.0 + variance)

        weight_values = list(weights.values())
        weight_entropy = -sum(w * np.log(w + 1e-6) for w in weight_values)
        max_entropy = np.log(len(weights)) if len(weights) > 1 else 1.0
        balance = weight_entropy / max_entropy if max_entropy > 0 else 1.0

        quality = 0.6 * agreement + 0.4 * balance
        return float(quality)

    def update_baseline(self, model_id: str, predictions: np.ndarray, pattern_id: str = "normal") -> None:
        """Зберігає базові стат-показники для пари (Модель, Патерн)."""
        stats_dict = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'count': len(predictions),
            'pattern': pattern_id,
            'updated_at': datetime.now().isoformat()
        }
        key = f"{model_id}_{pattern_id}"
        self.baseline_stats[key] = stats_dict
        
        # Backward compatibility for tests expecting model_id key
        if pattern_id == "normal" or pattern_id == "":
            self.baseline_stats[model_id] = stats_dict
            
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"✅ Baseline updated for Expert: {key}")

    def get_baseline(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get baseline statistics for a model."""
        for key in [f"{model_id}_normal", model_id]:
            if key in self.baseline_stats:
                return self.baseline_stats[key]
        for key, val in self.baseline_stats.items():
            if key.startswith(f"{model_id}_") or key == model_id:
                return val
        return None

    def flag_anomalies(self, predictions: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Flag predictions that are anomalies (exceed standard deviation threshold)."""
        if len(predictions) == 0:
            return np.array([], dtype=bool)
        mean = np.mean(predictions)
        std = np.std(predictions)
        if std == 0:
            return np.zeros_like(predictions, dtype=bool)
        z_scores = np.abs(predictions - mean) / std
        return z_scores > threshold

    def compare_models(self, model_a_preds: np.ndarray, model_b_preds: np.ndarray, actuals: Optional[np.ndarray] = None) -> dict[str, Any]:
        """Compare predictions of two models."""
        correlation = float(np.corrcoef(model_a_preds, model_b_preds)[0, 1]) if len(model_a_preds) > 1 else 1.0
        mean_abs_diff = float(np.mean(np.abs(model_a_preds - model_b_preds)))
        
        report = {
            'correlation': correlation,
            'mean_absolute_difference': mean_abs_diff,
            'model_a_mean': float(np.mean(model_a_preds)),
            'model_b_mean': float(np.mean(model_b_preds)),
            'model_a_std': float(np.std(model_a_preds)),
            'model_b_std': float(np.std(model_b_preds))
        }
        
        if actuals is not None:
            mae_a = float(np.mean(np.abs(model_a_preds - actuals)))
            mae_b = float(np.mean(np.abs(model_b_preds - actuals)))
            report['model_a_mae'] = mae_a
            report['model_b_mae'] = mae_b
            report['better_model'] = "A" if mae_a < mae_b else "B"
            report['improvement'] = float(abs(mae_a - mae_b))
            
        return report

    def generate_report(self) -> dict[str, Any]:
        unique_models = set()
        for key in self.baseline_stats.keys():
            if "_" in key:
                base_id = key.split("_")[0]
            else:
                base_id = key
            unique_models.add(base_id)
            
        baseline_models = list(unique_models)
        return {
            'drift_threshold': self.drift_threshold,
            'total_baselines': len(baseline_models),
            'baseline_models': baseline_models,
            'regimes_tracked': list(set(s['pattern'] for s in self.baseline_stats.values())),
            'timestamp': datetime.now().isoformat()
        }

# Global singleton
_quality_controller: Optional[ModelQualityController] = None

def get_quality_controller(drift_threshold: float = 0.3) -> ModelQualityController:
    global _quality_controller
    if _quality_controller is None:
        _quality_controller = ModelQualityController(drift_threshold=drift_threshold)
    return _quality_controller

