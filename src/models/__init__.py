"""
Models module - machine learning models
"""

# Model analysis and selection
from .model_selector import SmartModelSelector
from .quality.controller import ModelQualityController

# Advanced model analysis
from .analysis.baseline_dominance_detector import BaselineDominanceDetector, get_baseline_dominance_detector
from .analysis.regime_winner_analyzer import RegimeWinnerAnalyzer, get_regime_winner_analyzer
from .analysis.overfitting_detector import OverfittingDetector, get_overfitting_detector

# Model monitoring
from .monitoring.prediction_drift_monitor import PredictionDriftMonitor, get_prediction_drift_monitor

# Advanced ensemble components
from .ensemble.confidence_calibrator import (
    ConfidenceCalibrator, get_confidence_calibrator, calibrate_confidence_quick
)
from .ensemble.model_correlation_analyzer import (
    ModelCorrelationAnalyzer, get_model_correlation_analyzer, analyze_model_correlation_quick
)
from .ensemble.weight_stability_monitor import (
    WeightStabilityMonitor, get_weight_stability_monitor, monitor_weight_stability_quick
)

# Integrated model management
from .integrated_model_manager import IntegratedModelManager, get_integrated_model_manager
