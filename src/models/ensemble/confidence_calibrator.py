#!/usr/bin/env python3
"""
Confidence Calibrator - Calibrates prediction confidence for ensemble models
Implements various calibration methods to ensure reliable probability estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ConfidenceCalibrator")

class ConfidenceCalibrator:
    """
    Confidence calibrator for ensemble models.
    
    This calibrator provides:
    - Platt scaling for binary classification
    - Isotonic regression for multi-class calibration
    - Temperature scaling for neural networks
    - Beta calibration for probability smoothing
    - Ensemble-specific calibration methods
    
    Critical for reliable probability estimates in trading decisions.
    """
    
    # Calibration methods
    CALIBRATION_METHODS = {
        'platt': {
            'description': 'Platt scaling (logistic regression)',
            'suitable_for': ['binary_classification'],
            'complexity': 'low'
        },
        'isotonic': {
            'description': 'Isotonic regression calibration',
            'suitable_for': ['binary_classification', 'multi_class'],
            'complexity': 'medium'
        },
        'temperature': {
            'description': 'Temperature scaling for neural networks',
            'suitable_for': ['neural_networks', 'deep_learning'],
            'complexity': 'low'
        },
        'beta': {
            'description': 'Beta calibration for probability smoothing',
            'suitable_for': ['ensemble', 'probabilistic_models'],
            'complexity': 'medium'
        },
        'ensemble': {
            'description': 'Ensemble-specific calibration',
            'suitable_for': ['ensemble_models'],
            'complexity': 'high'
        }
    }
    
    def __init__(self, 
                 method: str = "isotonic",
                 task_type: str = "classification",
                 n_bins: int = 10):
        """
        Initialize Confidence Calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature', 'beta', 'ensemble')
            task_type: Task type ('classification', 'regression')
            n_bins: Number of bins for reliability diagrams
        """
        self.logger = logger
        self.method = method
        self.task_type = task_type
        self.n_bins = n_bins
        
        # Calibration components
        self.calibrator = None
        self.is_fitted = False
        self.calibration_data = {}
        
        # Method-specific parameters
        self.temperature = 1.0  # For temperature scaling
        self.beta_params = {'alpha': 1.0, 'beta': 1.0}  # For beta calibration
        
        # Validation data for calibration
        self.validation_predictions = None
        self.validation_targets = None
        
        # Calibration history
        self.calibration_history = []
        
        self.logger.info(f"✅ ConfidenceCalibrator initialized with method: {method}")
    
    def fit(self, 
             predictions: np.ndarray,
             targets: np.ndarray,
             model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit calibrator on validation data.
        
        Args:
            predictions: Model predictions (probabilities for classification)
            targets: True target values
            model_type: Type of model being calibrated
            
        Returns:
            Dict with calibration results and metrics
        """
        self.logger.info(f"🎯 Fitting confidence calibrator with method: {self.method}")
        
        results = {
            'method': self.method,
            'task_type': self.task_type,
            'fitted_at': datetime.now(),
            'calibration_metrics': {},
            'is_fitted': False
        }
        
        try:
            # Store validation data
            self.validation_predictions = predictions
            self.validation_targets = targets
            
            # Validate inputs
            if len(predictions) != len(targets):
                raise ValueError("Predictions and targets must have same length")
            
            # Choose calibration method
            if self.method == 'platt':
                calibration_result = self._fit_platt_scaling(predictions, targets)
            elif self.method == 'isotonic':
                calibration_result = self._fit_isotonic_regression(predictions, targets)
            elif self.method == 'temperature':
                calibration_result = self._fit_temperature_scaling(predictions, targets)
            elif self.method == 'beta':
                calibration_result = self._fit_beta_calibration(predictions, targets)
            elif self.method == 'ensemble':
                calibration_result = self._fit_ensemble_calibration(predictions, targets, model_type)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            # Update results
            results.update(calibration_result)
            
            # Calculate calibration metrics
            calibration_metrics = self._calculate_calibration_metrics(predictions, targets)
            results['calibration_metrics'] = calibration_metrics
            
            # Store calibration data
            self.calibration_data = results
            self.is_fitted = True
            results['is_fitted'] = True
            
            # Store in history
            self.calibration_history.append(results)
            
            self.logger.info(f"✅ Calibration fitted. ECE: {calibration_metrics.get('ece', 0):.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error fitting calibrator: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _fit_platt_scaling(self, 
                           predictions: np.ndarray,
                           targets: np.ndarray) -> Dict[str, Any]:
        """Fit Platt scaling (logistic regression) calibrator."""
        
        try:
            if self.task_type != 'classification':
                raise ValueError("Platt scaling only for classification")
            
            # Use logistic regression to map scores to calibrated probabilities
            self.calibrator = LogisticRegression(random_state=42)
            
            # For binary classification, use positive class probabilities
            if predictions.ndim == 2:
                if predictions.shape[1] == 2:
                    scores = predictions[:, 1]  # Positive class
                else:
                    scores = np.max(predictions, axis=1)  # Max probability
            else:
                scores = predictions
            
            self.calibrator.fit(scores.reshape(-1, 1), targets)
            
            return {
                'calibrator_type': 'platt_scaling',
                'coefficients': self.calibrator.coef_.tolist(),
                'intercept': self.calibrator.intercept_.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting Platt scaling: {e}")
            return {'error': str(e)}
    
    def _fit_isotonic_regression(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray) -> Dict[str, Any]:
        """Fit isotonic regression calibrator."""
        
        try:
            # Create isotonic regression calibrator
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            
            if self.task_type == 'classification':
                if predictions.ndim == 2:
                    # Multi-class: calibrate each class separately
                    self.calibrators = {}
                    for class_idx in range(predictions.shape[1]):
                        class_calibrator = IsotonicRegression(out_of_bounds='clip')
                        class_calibrator.fit(predictions[:, class_idx], targets == class_idx)
                        self.calibrators[class_idx] = class_calibrator
                    
                    self.calibrator = self.calibrators
                else:
                    # Binary classification
                    self.calibrator.fit(predictions, targets)
            else:
                # Regression: calibrate predictions directly
                self.calibrator.fit(predictions, targets)
            
            return {
                'calibrator_type': 'isotonic_regression',
                'n_classes': predictions.shape[1] if predictions.ndim == 2 else 1
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting isotonic regression: {e}")
            return {'error': str(e)}
    
    def _fit_temperature_scaling(self, 
                               predictions: np.ndarray,
                               targets: np.ndarray) -> Dict[str, Any]:
        """Fit temperature scaling calibrator."""
        
        try:
            # Temperature scaling: optimize temperature parameter
            from scipy.optimize import minimize
            
            def nll_loss(temp):
                """Negative log likelihood loss for temperature scaling."""
                temp_predictions = predictions / temp
                if temp_predictions.ndim == 2:
                    # Multi-class: apply softmax with temperature
                    exp_preds = np.exp(temp_predictions - np.max(temp_predictions, axis=1, keepdims=True))
                    softmax_preds = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
                else:
                    # Binary: apply sigmoid with temperature
                    softmax_preds = 1 / (1 + np.exp(-temp_predictions / temp))
                
                # Avoid log(0)
                softmax_preds = np.clip(softmax_preds, 1e-15, 1 - 1e-15)
                
                if self.task_type == 'classification':
                    if targets.ndim == 1:  # One-hot encoded targets
                        nll = -np.sum(targets * np.log(softmax_preds)) / len(targets)
                    else:  # Class indices
                        nll = -np.mean(np.log(softmax_preds[np.arange(len(targets)), targets]))
                else:
                    # Regression: use MSE
                    nll = np.mean((softmax_preds - targets) ** 2)
                
                return nll
            
            # Optimize temperature
            result = minimize(nll_loss, x0=1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
            
            if result.success:
                self.temperature = result.x[0]
            else:
                self.logger.warning("Temperature scaling optimization failed, using default")
                self.temperature = 1.0
            
            return {
                'calibrator_type': 'temperature_scaling',
                'temperature': self.temperature,
                'optimization_success': result.success
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting temperature scaling: {e}")
            return {'error': str(e)}
    
    def _fit_beta_calibration(self, 
                            predictions: np.ndarray,
                            targets: np.ndarray) -> Dict[str, Any]:
        """Fit beta calibration for probability smoothing."""
        
        try:
            from scipy.stats import beta
            from scipy.optimize import minimize
            
            def beta_nll(params):
                """Negative log likelihood for beta calibration."""
                alpha, beta_param = params
                
                # Apply beta calibration
                calibrated = beta.cdf(predictions, alpha, beta_param)
                
                # Avoid log(0)
                calibrated = np.clip(calibrated, 1e-15, 1 - 1e-15)
                
                # Calculate NLL
                if self.task_type == 'classification':
                    nll = -np.mean(targets * np.log(calibrated) + 
                                   (1 - targets) * np.log(1 - calibrated))
                else:
                    nll = np.mean((calibrated - targets) ** 2)
                
                return nll
            
            # Optimize beta parameters
            result = minimize(beta_nll, x0=[1.0, 1.0], method='L-BFGS-B', 
                           bounds=[(0.1, 10.0), (0.1, 10.0)])
            
            if result.success:
                self.beta_params['alpha'] = result.x[0]
                self.beta_params['beta'] = result.x[1]
            else:
                self.logger.warning("Beta calibration optimization failed, using defaults")
            
            return {
                'calibrator_type': 'beta_calibration',
                'alpha': self.beta_params['alpha'],
                'beta': self.beta_params['beta'],
                'optimization_success': result.success
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting beta calibration: {e}")
            return {'error': str(e)}
    
    def _fit_ensemble_calibration(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 model_type: Optional[str] = None) -> Dict[str, Any]:
        """Fit ensemble-specific calibration."""
        
        try:
            # Ensemble calibration combines multiple methods
            ensemble_calibrators = {}
            
            # For ensembles, use isotonic regression as primary
            if self.task_type == 'classification':
                if predictions.ndim == 2:
                    ensemble_calibrators['isotonic'] = {}
                    for class_idx in range(predictions.shape[1]):
                        calibrator = IsotonicRegression(out_of_bounds='clip')
                        calibrator.fit(predictions[:, class_idx], targets == class_idx)
                        ensemble_calibrators['isotonic'][class_idx] = calibrator
                    
                    # Add Platt scaling as backup
                    ensemble_calibrators['platt'] = LogisticRegression(random_state=42)
                    scores = np.max(predictions, axis=1)
                    ensemble_calibrators['platt'].fit(scores.reshape(-1, 1), targets)
                else:
                    ensemble_calibrators['isotonic'] = IsotonicRegression(out_of_bounds='clip')
                    ensemble_calibrators['isotonic'].fit(predictions, targets)
            
            self.calibrator = ensemble_calibrators
            
            return {
                'calibrator_type': 'ensemble_calibration',
                'methods': list(ensemble_calibrators.keys()),
                'model_type': model_type
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting ensemble calibration: {e}")
            return {'error': str(e)}
    
    def transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply calibration to predictions.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Calibrated predictions
        """
        if not self.is_fitted:
            self.logger.warning("Calibrator not fitted, returning raw predictions")
            return predictions
        
        try:
            if self.method == 'platt':
                return self._transform_platt(predictions)
            elif self.method == 'isotonic':
                return self._transform_isotonic(predictions)
            elif self.method == 'temperature':
                return self._transform_temperature(predictions)
            elif self.method == 'beta':
                return self._transform_beta(predictions)
            elif self.method == 'ensemble':
                return self._transform_ensemble(predictions)
            else:
                return predictions
                
        except Exception as e:
            self.logger.error(f"Error applying calibration: {e}")
            return predictions
    
    def _transform_platt(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Platt scaling transformation."""
        
        if predictions.ndim == 2:
            if predictions.shape[1] == 2:
                scores = predictions[:, 1].reshape(-1, 1)
            else:
                scores = np.max(predictions, axis=1).reshape(-1, 1)
        else:
            scores = predictions.reshape(-1, 1)
        
        calibrated = self.calibrator.predict_proba(scores)
        
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            return calibrated
        elif predictions.ndim == 2:
            # Reconstruct multi-class probabilities
            max_probs = np.max(predictions, axis=1)
            max_indices = np.argmax(predictions, axis=1)
            calibrated_max = calibrated[:, 1]
            
            result = predictions.copy()
            for i, idx in enumerate(max_indices):
                result[i, idx] = calibrated_max[i]
            
            # Normalize
            result = result / np.sum(result, axis=1, keepdims=True)
            return result
        else:
            return calibrated[:, 1]
    
    def _transform_isotonic(self, predictions: np.ndarray) -> np.ndarray:
        """Apply isotonic regression transformation."""
        
        if predictions.ndim == 2:
            # Multi-class: apply to each class
            calibrated = np.zeros_like(predictions)
            for class_idx in range(predictions.shape[1]):
                if isinstance(self.calibrator, dict):
                    calibrated[:, class_idx] = self.calibrator[class_idx].transform(predictions[:, class_idx])
                else:
                    calibrated[:, class_idx] = self.calibrator.transform(predictions[:, class_idx])
            
            # Normalize
            calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
            return calibrated
        else:
            if isinstance(self.calibrator, dict):
                return self.calibrator[0].transform(predictions)
            else:
                return self.calibrator.transform(predictions)
    
    def _transform_temperature(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temperature scaling transformation."""
        
        temp_predictions = predictions / self.temperature
        
        if predictions.ndim == 2:
            # Multi-class: apply softmax with temperature
            exp_preds = np.exp(temp_predictions - np.max(temp_predictions, axis=1, keepdims=True))
            calibrated = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        else:
            # Binary: apply sigmoid with temperature
            calibrated = 1 / (1 + np.exp(-temp_predictions))
        
        return calibrated
    
    def _transform_beta(self, predictions: np.ndarray) -> np.ndarray:
        """Apply beta calibration transformation."""
        
        from scipy.stats import beta
        
        calibrated = beta.cdf(predictions, 
                          self.beta_params['alpha'], 
                          self.beta_params['beta'])
        
        return calibrated
    
    def _transform_ensemble(self, predictions: np.ndarray) -> np.ndarray:
        """Apply ensemble calibration transformation."""
        
        # Use isotonic regression as primary method
        if 'isotonic' in self.calibrator:
            return self._transform_isotonic(predictions)
        elif 'platt' in self.calibrator:
            return self._transform_platt(predictions)
        else:
            return predictions
    
    def _calculate_calibration_metrics(self, 
                                    predictions: np.ndarray,
                                    targets: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        
        try:
            metrics = {}
            
            if self.task_type == 'classification':
                # Expected Calibration Error (ECE)
                ece = self._calculate_ece(predictions, targets)
                metrics['ece'] = ece
                
                # Brier score
                if predictions.ndim == 2:
                    # Multi-class: use negative log likelihood
                    metrics['brier_score'] = log_loss(targets, predictions)
                else:
                    # Binary: use Brier score
                    metrics['brier_score'] = brier_score_loss(targets, predictions)
                
                # Reliability diagram data
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    targets, predictions if predictions.ndim == 1 else predictions[:, 1], 
                    n_bins=self.n_bins
                )
                metrics['reliability_data'] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating calibration metrics: {e}")
            return {}
    
    def _calculate_ece(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        
        try:
            if predictions.ndim == 2:
                # Multi-class: use max probability
                prob_pred = np.max(predictions, axis=1)
            else:
                prob_pred = predictions
            
            # Create bins
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
            bin_lowers = bin_edges[:-1]
            bin_uppers = bin_edges[1:]
            
            ece = 0.0
            
            for i in range(self.n_bins):
                # Find samples in this bin
                in_bin = (prob_pred > bin_lowers[i]) & (prob_pred <= bin_uppers[i])
                
                if np.sum(in_bin) > 0:
                    # Calculate accuracy in bin
                    if predictions.ndim == 2:
                        pred_class = np.argmax(predictions[in_bin], axis=1)
                        accuracy_in_bin = np.mean(pred_class == targets[in_bin])
                    else:
                        accuracy_in_bin = np.mean((predictions[in_bin] > 0.5) == targets[in_bin])
                    
                    # Average confidence in bin
                    avg_confidence_in_bin = np.mean(prob_pred[in_bin])
                    
                    # Weight by bin size
                    bin_weight = np.sum(in_bin) / len(targets)
                    
                    ece += bin_weight * abs(accuracy_in_bin - avg_confidence_in_bin)
            
            return ece
            
        except Exception as e:
            self.logger.error(f"Error calculating ECE: {e}")
            return 0.0
    
    def evaluate_calibration(self, 
                           predictions: np.ndarray,
                           targets: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate calibration quality with comprehensive metrics.
        
        Args:
            predictions: Model predictions to evaluate
            targets: True target values
            
        Returns:
            Dict with calibration evaluation results
        """
        if not self.is_fitted:
            return {'error': 'Calibrator not fitted'}
        
        try:
            # Apply calibration
            calibrated_predictions = self.transform(predictions)
            
            # Calculate metrics for both raw and calibrated predictions
            raw_metrics = self._calculate_calibration_metrics(predictions, targets)
            calibrated_metrics = self._calculate_calibration_metrics(calibrated_predictions, targets)
            
            # Calculate improvement
            improvement = {}
            for metric in raw_metrics:
                if metric in calibrated_metrics:
                    improvement[metric] = raw_metrics[metric] - calibrated_metrics[metric]
            
            results = {
                'raw_metrics': raw_metrics,
                'calibrated_metrics': calibrated_metrics,
                'improvement': improvement,
                'calibration_method': self.method,
                'evaluated_at': datetime.now()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating calibration: {e}")
            return {'error': str(e)}
    
    def plot_reliability_diagram(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot reliability diagram for calibration visualization.
        
        Args:
            predictions: Model predictions
            targets: True target values
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if predictions.ndim == 2:
                prob_pred = np.max(predictions, axis=1)
            else:
                prob_pred = predictions
            
            # Get calibration curve data
            fraction_of_positives, mean_predicted_value = calibration_curve(
                targets, prob_pred, n_bins=self.n_bins
            )
            
            # Plot reliability diagram
            plt.figure(figsize=(8, 6))
            
            # Perfect calibration line
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            # Model calibration
            plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                    label=f'{self.method.capitalize()} Calibration')
            
            # Bins histogram
            bin_counts = np.histogram(prob_pred, bins=self.n_bins, range=(0, 1))[0]
            bin_centers = (np.arange(self.n_bins) + 0.5) / self.n_bins
            
            plt.twinx()
            plt.bar(bin_centers, bin_counts, width=1/self.n_bins, alpha=0.3, 
                    color='gray', label='Sample Count')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Reliability Diagram - {self.method.capitalize()} Calibration')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Reliability diagram saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting reliability diagram: {e}")
    
    def save_calibrator(self, filepath: str) -> bool:
        """Save calibrator to file."""
        
        try:
            save_data = {
                'method': self.method,
                'task_type': self.task_type,
                'n_bins': self.n_bins,
                'calibration_data': self.calibration_data,
                'is_fitted': self.is_fitted,
                'saved_at': datetime.now().isoformat()
            }
            
            # Add method-specific data
            if self.method == 'temperature':
                save_data['temperature'] = self.temperature
            elif self.method == 'beta':
                save_data['beta_params'] = self.beta_params
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Calibrator saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving calibrator: {e}")
            return False
    
    def load_calibrator(self, filepath: str) -> bool:
        """Load calibrator from file."""
        
        try:
            with open(filepath, 'r') as f:
                load_data = json.load(f)
            
            self.method = load_data.get('method', self.method)
            self.task_type = load_data.get('task_type', self.task_type)
            self.n_bins = load_data.get('n_bins', self.n_bins)
            self.calibration_data = load_data.get('calibration_data', {})
            self.is_fitted = load_data.get('is_fitted', False)
            
            # Restore method-specific data
            if self.method == 'temperature':
                self.temperature = load_data.get('temperature', 1.0)
            elif self.method == 'beta':
                self.beta_params = load_data.get('beta_params', {'alpha': 1.0, 'beta': 1.0})
            
            # Note: Actual calibrator objects need to be reconstructed
            # This is a simplified version - in production, you'd need proper serialization
            
            self.logger.info(f"Calibrator loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading calibrator: {e}")
            return False


# Factory function for easy instantiation
def get_confidence_calibrator(method: str = "isotonic",
                            task_type: str = "classification",
                            n_bins: int = 10) -> ConfidenceCalibrator:
    """Factory function to get ConfidenceCalibrator instance."""
    return ConfidenceCalibrator(method, task_type, n_bins)


# Convenience function for quick calibration
def calibrate_confidence_quick(predictions: np.ndarray,
                           targets: np.ndarray,
                           method: str = "isotonic",
                           task_type: str = "classification") -> Dict[str, Any]:
    """
    Quick confidence calibration.
    
    Args:
        predictions: Model predictions to calibrate
        targets: True target values
        method: Calibration method
        task_type: Task type
        
    Returns:
        Calibration result dictionary
    """
    calibrator = get_confidence_calibrator(method, task_type)
    return calibrator.fit(predictions, targets)
