"""
Model Selector Module
Provides smart model selection based on context and performance history.
"""

# Primary selector (context-based)
# Adaptive selector
from .adaptive_selector import AdaptiveModelSelector

# Fingerprint-based selector (backward compatibility)
from .fingerprint_selector import SmartModelSelector as FingerprintModelSelector
from .smart_selector import SmartModelSelector

__all__ = [
    'SmartModelSelector',
    'FingerprintModelSelector',
    'AdaptiveModelSelector',
]
