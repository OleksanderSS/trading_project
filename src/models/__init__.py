"""
Models module - machine learning models
"""

# Model analysis and selection
# Advanced model analysis
from .analysis.baseline_dominance_detector import BaselineDominanceDetector
from .analysis.overfitting_detector import OverfittingDetector, detect_overfitting_quick, get_overfitting_detector
from .analysis.regime_winner_analyzer import RegimeWinnerAnalyzer

# Advanced ensemble components
from .ensemble.confidence_calibrator import ConfidenceCalibrator, calibrate_confidence_quick, get_confidence_calibrator
from .ensemble.model_correlation_analyzer import (
    ModelCorrelationAnalyzer,
    analyze_model_correlation_quick,
    get_model_correlation_analyzer,
)
from .ensemble.weight_stability_monitor import (
    WeightStabilityMonitor,
    get_weight_stability_monitor,
    monitor_weight_stability_quick,
)

# Integrated model management
from .integrated_model_manager import IntegratedModelManager, get_integrated_model_manager
from .model_selector import SmartModelSelector

# Model monitoring
from .monitoring.prediction_drift_monitor import PredictionDriftMonitor, get_prediction_drift_monitor
from .quality.controller import ModelQualityController

__all__ = [
    'BaselineDominanceDetector',
    'OverfittingDetector',
    'detect_overfitting_quick',
    'get_overfitting_detector',
    'RegimeWinnerAnalyzer',
    'ConfidenceCalibrator',
    'calibrate_confidence_quick',
    'get_confidence_calibrator',
    'ModelCorrelationAnalyzer',
    'analyze_model_correlation_quick',
    'get_model_correlation_analyzer',
    'WeightStabilityMonitor',
    'get_weight_stability_monitor',
    'monitor_weight_stability_quick',
    'IntegratedModelManager',
    'get_integrated_model_manager',
    'SmartModelSelector',
    'PredictionDriftMonitor',
    'get_prediction_drift_monitor',
    'ModelQualityController',
]
