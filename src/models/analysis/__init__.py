"""
Models Analysis Module - Advanced Model Analysis Components
==================================================

This module provides advanced analysis capabilities for model performance and selection.

Key Components:
- BaselineDominanceDetector: Detects when simple baselines outperform complex models
- RegimeWinnerAnalyzer: Analyzes model winner consistency across market regimes

These components enable sophisticated model analysis and adaptation to changing market conditions.
"""

from .baseline_dominance_detector import BaselineDominanceDetector
from .overfitting_detector import OverfittingDetector
from .regime_winner_analyzer import RegimeWinnerAnalyzer

__all__ = [
    'BaselineDominanceDetector',
    'RegimeWinnerAnalyzer',
    'OverfittingDetector'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Advanced model analysis for adaptive trading systems'
