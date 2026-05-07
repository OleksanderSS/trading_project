"""
Models Monitoring Module - Real-time Model Performance Monitoring
====================================================

This module provides comprehensive monitoring capabilities for model performance
and prediction drift detection in production environments.

Key Components:
- PredictionDriftMonitor: Real-time prediction drift monitoring and retraining triggers

These components enable proactive model maintenance and ensure production reliability.
"""

from .prediction_drift_monitor import (
    PredictionDriftMonitor,
    get_prediction_drift_monitor,
    monitor_predictions_quick
)

__all__ = [
    # Core monitoring classes
    'PredictionDriftMonitor',
    
    # Factory functions
    'get_prediction_drift_monitor',
    
    # Quick monitoring functions
    'monitor_predictions_quick'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Real-time model monitoring for adaptive trading systems'
