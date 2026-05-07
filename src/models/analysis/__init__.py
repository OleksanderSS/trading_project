"""
Models Analysis Module - Advanced Model Analysis Components
==================================================

This module provides advanced analysis capabilities for model performance and selection.

Key Components:
- BaselineDominanceDetector: Detects when simple baselines outperform complex models
- RegimeWinnerAnalyzer: Analyzes model winner consistency across market regimes

These components enable sophisticated model analysis and adaptation to changing market conditions.
"""

from .baseline_dominance_detector import (
    BaselineDominanceDetector,
    get_baseline_dominance_detector,
    analyze_baseline_dominance_quick
)

from .regime_winner_analyzer import (
    RegimeWinnerAnalyzer,
    get_regime_winner_analyzer,
    analyze_regime_consistency_quick
)

from .overfitting_detector import (
    OverfittingDetector,
    get_overfitting_detector,
    detect_overfitting_quick
)

__all__ = [
    # Core analysis classes
    'BaselineDominanceDetector',
    'RegimeWinnerAnalyzer',
    'OverfittingDetector',
    
    # Factory functions
    'get_baseline_dominance_detector',
    'get_regime_winner_analyzer',
    'get_overfitting_detector',
    
    # Quick analysis functions
    'analyze_baseline_dominance_quick',
    'analyze_regime_consistency_quick',
    'detect_overfitting_quick'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Advanced model analysis for adaptive trading systems'
