"""
Model Selector Module
Provides smart model selection based on context and performance history.
"""

# Primary selector (performance history-based)
# Adaptive selector
from .adaptive_selector import AdaptiveModelSelector

# Fingerprint-based selector
from .fingerprint_selector import FingerprintModelSelector
from .smart_selector import PerformanceHistorySelector

# Backward compatibility aliases
SmartModelSelector = PerformanceHistorySelector

__all__ = [
    'SmartModelSelector',
    'PerformanceHistorySelector',
    'FingerprintModelSelector',
    'AdaptiveModelSelector',
]
