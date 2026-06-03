#!/usr/bin/env python3
"""
Overfitting Detector - Facade for the modular Overfitting-Detection System.
This module maintains backward compatibility while delegating to the new modular structure.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .overfitting_detection.manager import OverfittingDetector as ModularOverfittingDetector

class OverfittingDetector(ModularOverfittingDetector):
    """
    Facade for OverfittingDetector.
    Maintains the original API but uses modular components internally.
    """
    pass

def get_overfitting_detector(config: Optional[Dict[str, Any]] = None) -> OverfittingDetector:
    """Factory function to get OverfittingDetector instance."""
    return OverfittingDetector(config)

async def detect_overfitting_quick(model: Any,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick overfitting detection.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary
        
    Returns:
        Overfitting detection result dictionary
    """
    detector = get_overfitting_detector(config)
    return await detector.detect_overfitting(model, X_train, y_train, X_val, y_val)
