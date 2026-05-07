"""
ModelQualityController: Quality control for model predictions and drift detection

Features:
- Prediction validation (NaN/Inf checks)
- Drift detection (distribution shifts)
- Quality scoring (agreement + balance)
- Baseline tracking
- Report generation

Usage:
    controller = ModelQualityController(drift_threshold=0.3)
    
    # Validate predictions
    if controller.validate_predictions(predictions):
        # Use predictions
        pass
    
    # Check drift
    drift = controller.check_drift(current_preds, baseline_preds)
    if drift > threshold:
        # Retrain model
        pass
    
    # Calculate quality score
    score = controller.get_quality_score(ensemble_pred, predictions, weights)
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ModelQualityController:
    """
    Quality control for model predictions and drift detection.
    
    Features:
    - Validates predictions for NaN/Inf and reasonable values
    - Detects distribution drift using KL divergence approximation
    - Calculates quality scores based on agreement and weight distribution
    - Tracks baseline statistics for drift detection
    - Generates quality reports
    
    Attributes:
        drift_threshold: Threshold for drift detection
        baseline_stats: Dict[model_id, baseline_statistics]
    """
    
    def __init__(self, drift_threshold: float = 0.3):
        """
        Initialize quality controller.
        
        Args:
            drift_threshold: Threshold for drift detection (default: 0.3)
                           Higher = more tolerant to drift
        """
        self.drift_threshold = drift_threshold
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
        self.logger = ProjectLogger.get_logger(__name__)
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """
        Validate predictions for NaN/Inf and reasonable values.
        
        Args:
            predictions: Array of predictions to validate
        
        Returns:
            True if valid, False otherwise
        
        Example:
            predictions = model.predict(X)
            if controller.validate_predictions(predictions):
                # Use predictions
                pass
            else:
                # Handle invalid predictions
                logger.error("Invalid predictions detected")
        """
        # Check for NaN/Inf
        if np.any(np.isnan(predictions)):
            self.logger.warning("Predictions contain NaN values")
            return False
        
        if np.any(np.isinf(predictions)):
            self.logger.warning("Predictions contain Inf values")
            return False
        
        # Check for unrealistic values (>1000% return)
        if np.any(np.abs(predictions) > 10):
            self.logger.warning(f"Predictions contain unrealistic values: max={np.max(np.abs(predictions)):.2f}")
            return False
        
        return True
    
    def check_drift(self, current: np.ndarray, baseline: np.ndarray) -> float:
        """
        Check for distribution drift using KL divergence approximation.
        
        Args:
            current: Current predictions
            baseline: Baseline predictions
        
        Returns:
            Drift score (higher = more drift)
        
        Example:
            drift = controller.check_drift(current_preds, baseline_preds)
            if drift > controller.drift_threshold:
                logger.warning(f"Drift detected: {drift:.3f}")
                # Retrain model
        """
        current_mean = np.mean(current)
        current_std = np.std(current)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        # KL divergence approximation
        # drift = |μ_current - μ_baseline| / σ_baseline
        drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
        
        if drift > self.drift_threshold:
            self.logger.warning(
                f"Drift detected: {drift:.3f} > {self.drift_threshold} "
                f"(Δμ={abs(current_mean - baseline_mean):.4f}, σ_baseline={baseline_std:.4f})"
            )
        
        return float(drift)
    
    def get_quality_score(
        self,
        ensemble_pred: float,
        predictions: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate quality score based on agreement and weight distribution.
        
        Quality = 0.6 * agreement + 0.4 * balance
        
        Where:
        - agreement: 1 / (1 + variance) - lower variance = higher agreement
        - balance: entropy / max_entropy - more balanced weights = better
        
        Args:
            ensemble_pred: Final ensemble prediction
            predictions: Dict[model_id, prediction]
            weights: Dict[model_id, weight]
        
        Returns:
            Quality score (0.0-1.0)
        
        Example:
            predictions = {"model1": 0.05, "model2": 0.06, "model3": 0.05}
            weights = {"model1": 0.33, "model2": 0.33, "model3": 0.34}
            score = controller.get_quality_score(0.053, predictions, weights)
            # score ≈ 0.85 (high agreement, balanced weights)
        """
        # Variance of predictions (lower = better agreement)
        pred_values = list(predictions.values())
        variance = np.var(pred_values)
        
        # Agreement score (lower variance = higher agreement)
        agreement = 1.0 / (1.0 + variance)
        
        # Weight distribution (more balanced = better)
        weight_values = list(weights.values())
        weight_entropy = -sum(w * np.log(w + 1e-6) for w in weight_values)
        max_entropy = np.log(len(weights))
        balance = weight_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combined score
        quality = 0.6 * agreement + 0.4 * balance
        
        return float(quality)
    
    def update_baseline(self, model_id: str, predictions: np.ndarray) -> None:
        """
        Update baseline statistics for drift detection.
        
        Args:
            model_id: Model identifier
            predictions: Predictions to use as baseline
        
        Example:
            # After training
            train_preds = model.predict(X_train)
            controller.update_baseline("BTC_LSTM_v2", train_preds)
            
            # Later, check drift
            current_preds = model.predict(X_current)
            drift = controller.check_drift(current_preds, train_preds)
        """
        self.baseline_stats[model_id] = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'count': len(predictions),
            'updated_at': datetime.now().isoformat()
        }
        
        self.logger.debug(
            f"Updated baseline for {model_id}: "
            f"μ={self.baseline_stats[model_id]['mean']:.4f}, "
            f"σ={self.baseline_stats[model_id]['std']:.4f}"
        )
    
    def get_baseline(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get baseline statistics for model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Baseline statistics or None if not found
        """
        return self.baseline_stats.get(model_id)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate quality control report.
        
        Returns:
            Dict with quality metrics and statistics
        
        Example:
            report = controller.generate_report()
            print(f"Baseline models: {report['total_baselines']}")
            print(f"Drift threshold: {report['drift_threshold']}")
        """
        return {
            'drift_threshold': self.drift_threshold,
            'baseline_models': list(self.baseline_stats.keys()),
            'total_baselines': len(self.baseline_stats),
            'baseline_stats': self.baseline_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def flag_anomalies(
        self,
        predictions: np.ndarray,
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Flag anomalous predictions using z-score.
        
        Args:
            predictions: Array of predictions
            threshold: Z-score threshold (default: 3.0)
        
        Returns:
            Boolean array (True = anomaly)
        
        Example:
            predictions = model.predict(X)
            anomalies = controller.flag_anomalies(predictions, threshold=3.0)
            if np.any(anomalies):
                logger.warning(f"Found {np.sum(anomalies)} anomalies")
        """
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        z_scores = np.abs((predictions - mean) / (std + 1e-6))
        anomalies = z_scores > threshold
        
        if np.any(anomalies):
            self.logger.warning(
                f"Found {np.sum(anomalies)} anomalies "
                f"(threshold={threshold}, max_z={np.max(z_scores):.2f})"
            )
        
        return anomalies
    
    def compare_models(
        self,
        model_a_preds: np.ndarray,
        model_b_preds: np.ndarray,
        actuals: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare two models' predictions.
        
        Args:
            model_a_preds: Model A predictions
            model_b_preds: Model B predictions
            actuals: Actual values (optional, for accuracy comparison)
        
        Returns:
            Comparison metrics
        
        Example:
            comparison = controller.compare_models(
                model_a.predict(X), model_b.predict(X), y_true
            )
            print(f"Correlation: {comparison['correlation']:.3f}")
            if actuals:
                print(f"Model A MAE: {comparison['model_a_mae']:.4f}")
                print(f"Model B MAE: {comparison['model_b_mae']:.4f}")
        """
        # Correlation between predictions
        correlation = np.corrcoef(model_a_preds, model_b_preds)[0, 1]
        
        # Mean absolute difference
        mad = np.mean(np.abs(model_a_preds - model_b_preds))
        
        result = {
            'correlation': float(correlation),
            'mean_absolute_difference': float(mad),
            'model_a_mean': float(np.mean(model_a_preds)),
            'model_b_mean': float(np.mean(model_b_preds)),
            'model_a_std': float(np.std(model_a_preds)),
            'model_b_std': float(np.std(model_b_preds))
        }
        
        # If actuals provided, calculate accuracy metrics
        if actuals is not None:
            model_a_mae = np.mean(np.abs(model_a_preds - actuals))
            model_b_mae = np.mean(np.abs(model_b_preds - actuals))
            
            result['model_a_mae'] = float(model_a_mae)
            result['model_b_mae'] = float(model_b_mae)
            result['better_model'] = float('A' if model_a_mae < model_b_mae else 'B')
            result['improvement'] = float(abs(model_a_mae - model_b_mae) / max(model_a_mae, model_b_mae))
        
        return result
