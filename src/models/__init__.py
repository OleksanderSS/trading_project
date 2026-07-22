"""
Models module - machine learning models
"""

# Lazy imports to avoid heavy side effects (storage dir creation, monitoring initialization) on package import
# This allows import src.models without triggering constructor side effects

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


def __getattr__(name: str):
    """Lazy import for heavy model components."""
    # Model analysis
    if name == "BaselineDominanceDetector":
        from .analysis.baseline_dominance_detector import BaselineDominanceDetector
        return BaselineDominanceDetector
    elif name == "OverfittingDetector" or name == "detect_overfitting_quick" or name == "get_overfitting_detector":
        from .analysis.overfitting_detector import (
            OverfittingDetector,
            detect_overfitting_quick,
            get_overfitting_detector,
        )
        if name == "OverfittingDetector":
            return OverfittingDetector
        elif name == "detect_overfitting_quick":
            return detect_overfitting_quick
        elif name == "get_overfitting_detector":
            return get_overfitting_detector
    elif name == "RegimeWinnerAnalyzer":
        from .analysis.regime_winner_analyzer import RegimeWinnerAnalyzer
        return RegimeWinnerAnalyzer

    # Ensemble components
    elif name == "ConfidenceCalibrator" or name == "calibrate_confidence_quick" or name == "get_confidence_calibrator":
        from .ensemble.confidence_calibrator import (
            ConfidenceCalibrator,
            calibrate_confidence_quick,
            get_confidence_calibrator,
        )
        if name == "ConfidenceCalibrator":
            return ConfidenceCalibrator
        elif name == "calibrate_confidence_quick":
            return calibrate_confidence_quick
        elif name == "get_confidence_calibrator":
            return get_confidence_calibrator
    elif name == "ModelCorrelationAnalyzer" or name == "analyze_model_correlation_quick" or name == "get_model_correlation_analyzer":
        from .ensemble.model_correlation_analyzer import (
            ModelCorrelationAnalyzer,
            analyze_model_correlation_quick,
            get_model_correlation_analyzer,
        )
        if name == "ModelCorrelationAnalyzer":
            return ModelCorrelationAnalyzer
        elif name == "analyze_model_correlation_quick":
            return analyze_model_correlation_quick
        elif name == "get_model_correlation_analyzer":
            return get_model_correlation_analyzer
    elif name == "WeightStabilityMonitor" or name == "get_weight_stability_monitor" or name == "monitor_weight_stability_quick":
        from .ensemble.weight_stability_monitor import (
            WeightStabilityMonitor,
            get_weight_stability_monitor,
            monitor_weight_stability_quick,
        )
        if name == "WeightStabilityMonitor":
            return WeightStabilityMonitor
        elif name == "get_weight_stability_monitor":
            return get_weight_stability_monitor
        elif name == "monitor_weight_stability_quick":
            return monitor_weight_stability_quick

    # Integrated model management
    elif name == "IntegratedModelManager" or name == "get_integrated_model_manager":
        from .integrated_model_manager import IntegratedModelManager, get_integrated_model_manager
        if name == "IntegratedModelManager":
            return IntegratedModelManager
        elif name == "get_integrated_model_manager":
            return get_integrated_model_manager
    elif name == "SmartModelSelector":
        from .model_selector import SmartModelSelector
        return SmartModelSelector

    # Model monitoring
    elif name == "PredictionDriftMonitor" or name == "get_prediction_drift_monitor":
        from .monitoring.prediction_drift_monitor import PredictionDriftMonitor, get_prediction_drift_monitor
        if name == "PredictionDriftMonitor":
            return PredictionDriftMonitor
        elif name == "get_prediction_drift_monitor":
            return get_prediction_drift_monitor
    elif name == "ModelQualityController":
        from .quality.controller import ModelQualityController
        return ModelQualityController

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
