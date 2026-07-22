"""
Models Ensemble Module - Advanced Ensemble Components
============================================

This module provides comprehensive ensemble management and optimization capabilities.

Key Components:
- DynamicWeightCalculator: Dynamic weight calculation for ensemble models
- ConfidenceCalibrator: Calibrates prediction confidence for reliable probability estimates
- ModelCorrelationAnalyzer: Analyzes model correlation and diversity for ensemble optimization
- WeightStabilityMonitor: Monitors and ensures ensemble weight stability

These components enable building effective, diverse, and stable ensemble models.
"""

from .confidence_calibrator import ConfidenceCalibrator, calibrate_confidence_quick, get_confidence_calibrator
from .dynamic_weights import DynamicWeightCalculator
from .model_correlation_analyzer import (
    ModelCorrelationAnalyzer,
    analyze_model_correlation_quick,
    get_model_correlation_analyzer,
)
from .weight_stability_monitor import (
    WeightStabilityMonitor,
    get_weight_stability_monitor,
    monitor_weight_stability_quick,
)

__all__ = [
    # Core ensemble classes
    'DynamicWeightCalculator',
    'ConfidenceCalibrator',
    'ModelCorrelationAnalyzer',
    'WeightStabilityMonitor',

    # Factory functions
    'get_confidence_calibrator',
    'get_model_correlation_analyzer',
    'get_weight_stability_monitor',

    # Quick analysis functions
    'calibrate_confidence_quick',
    'analyze_model_correlation_quick',
    'monitor_weight_stability_quick'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Advanced ensemble components for adaptive trading systems'
