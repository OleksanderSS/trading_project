"""
Features Analysis Module - Advanced Feature Analysis Components
==================================================

This module provides advanced analysis capabilities for feature engineering.

Key Components:
- RegimeImportanceTracker: Dynamic feature importance tracking across market regimes
- NewsDecayModeler: ML-optimized news impact decay modeling

These components enable sophisticated feature analysis and adaptation to changing market conditions.
"""

from .regime_importance_tracker import (
    RegimeImportanceTracker,
    get_regime_importance_tracker,
    track_regime_importance_quick
)

from .news_decay_modeler import (
    NewsDecayModeler,
    get_news_decay_modeler,
    fit_news_decay_model_quick
)

__all__ = [
    # Core analysis classes
    'RegimeImportanceTracker',
    'NewsDecayModeler',
    
    # Factory functions
    'get_regime_importance_tracker',
    'get_news_decay_modeler',
    
    # Quick analysis functions
    'track_regime_importance_quick',
    'fit_news_decay_model_quick'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Advanced feature analysis for adaptive trading systems'
