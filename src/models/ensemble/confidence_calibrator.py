#!/usr/bin/env python3
"""
Confidence Calibrator - Calibrates prediction confidence for ensemble models
Implements various calibration methods to ensure reliable probability estimates.
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

import joblib

from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError
from src.models.ensemble.calibration.strategies import PlattScalingStrategy, IsotonicRegressionStrategy

logger = ProjectLogger.get_logger("ConfidenceCalibrator")

class ConfidenceCalibrator:
    """
    Confidence calibrator for ensemble models.
    """
    
    def __init__(self, method: str = "isotonic", task_type: str = "classification"):
        self.logger = logger
        self.method = method
        self.task_type = task_type
        self.is_fitted = False
        
        self.strategy = self._get_strategy(method, task_type)
        
        self.logger.info(f"✅ ConfidenceCalibrator initialized with method: {method}")

    def _get_strategy(self, method: str, task_type: str):
        if method == 'platt': return PlattScalingStrategy()
        if method == 'isotonic': return IsotonicRegressionStrategy(task_type)
        raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, 
             predictions: np.ndarray,
             targets: np.ndarray,
             model_type: Optional[str] = None) -> Dict[str, Any]:
        """Fit calibrator on validation data."""
        self.logger.info(f"🎯 Fitting confidence calibrator with method: {self.method}")
        
        try:
            if len(predictions) != len(targets):
                raise ValueError("Predictions and targets must have same length")
            
            results = self.strategy.fit(predictions, targets)
            self.is_fitted = True
            results['metrics'] = self._calculate_calibration_metrics(
                self.transform(predictions),
                targets,
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error fitting calibrator: {e}", exc_info=True)
            raise DataProcessingError(f"Fitting calibrator failed: {e}") from e
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions."""
        if not self.is_fitted:
            self.logger.warning("Calibrator not fitted, returning raw predictions")
            return predictions
        
        try:
            return self.strategy.transform(predictions)
        except Exception as e:
            self.logger.error(f"Error applying calibration: {e}")
            raise DataProcessingError(f"Applying calibration failed: {e}") from e
    
    # Keeping original helper methods for backward compatibility, 
    # but delegating their logic or deprecating.
    def _calculate_calibration_metrics(self, 
                                    predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, float]:
        """Calculate expected calibration error and Brier score."""
        preds = np.asarray(predictions, dtype=float).reshape(-1)
        actual = np.asarray(targets, dtype=float).reshape(-1)
        if len(preds) != len(actual):
            raise ValueError("Predictions and targets must have same length")
        clipped = np.clip(preds, 0.0, 1.0)
        brier_score = float(np.mean((clipped - actual) ** 2))
        bins = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for lower, upper in zip(bins[:-1], bins[1:]):
            in_bin = (clipped >= lower) & (clipped <= upper if upper == 1.0 else clipped < upper)
            if not np.any(in_bin):
                continue
            bin_confidence = float(np.mean(clipped[in_bin]))
            bin_accuracy = float(np.mean(actual[in_bin]))
            ece += float(np.mean(in_bin)) * abs(bin_confidence - bin_accuracy)
        return {'ece': float(ece), 'brier_score': brier_score}
    
    def save_calibrator(self, filepath: str) -> bool:
        """Save calibrator to file."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'method': self.method,
                'task_type': self.task_type,
                'is_fitted': self.is_fitted,
                'strategy': self.strategy,
            }, path)
            return True
        except Exception as e:
            self.logger.error(f"Error saving calibrator: {e}", exc_info=True)
            return False
    
    def load_calibrator(self, filepath: str) -> bool:
        """Load calibrator from file with security validation."""
        try:
            # Security validation: Ensure path is within expected data or models directories
            abs_p = Path(filepath).resolve()
            allowed_bases = [
                Path('data').resolve(),
                Path('models').resolve()
            ]
            if not any(abs_p.is_relative_to(base) for base in allowed_bases):
                self.logger.warning(f"🚫 Blocking unsafe calibrator load attempt from: {filepath}")
                return False

            # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            payload = joblib.load(filepath)
            self.method = payload['method']
            self.task_type = payload['task_type']
            self.is_fitted = payload['is_fitted']
            self.strategy = payload['strategy']
            return True
        except Exception as e:
            self.logger.error(f"Error loading calibrator: {e}", exc_info=True)
            return False

# Factory function for easy instantiation
def get_confidence_calibrator(method: str = "isotonic",
                            task_type: str = "classification") -> ConfidenceCalibrator:
    """Factory function to get ConfidenceCalibrator instance."""
    return ConfidenceCalibrator(method, task_type)

# Convenience function for quick calibration
def calibrate_confidence_quick(predictions: np.ndarray,
                           targets: np.ndarray,
                           method: str = "isotonic",
                           task_type: str = "classification") -> Dict[str, Any]:
    """Quick confidence calibration."""
    calibrator = get_confidence_calibrator(method, task_type)
    return calibrator.fit(predictions, targets)
