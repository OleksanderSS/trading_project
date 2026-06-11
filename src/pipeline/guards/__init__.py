"""
Pipeline Guards Module - Temporal Safety Protection
==============================================

This module provides comprehensive temporal safety guards for trading pipelines.

Key Components:
- TimeframeAlignmentGuard: Validates multi-timeframe temporal compatibility
- SafeFeatureCombiner: Temporal-safe feature combination
- TemporalTargetGuard: Safe target generation with time constraints
- TemporalLeakageGuard: Protection against rolling windows leakage
- MacroReleaseTimingGuard: Prevents early macro data access
- DataFreshnessMonitor: Real-time data freshness monitoring

These guards prevent the most common and dangerous temporal leakage issues
in trading systems, ensuring realistic backtest results and reliable live performance.
"""

from .macro_release_timing_guard import (
    MacroReleaseTimingGuard,
    get_macro_release_timing_guard,
    validate_macro_timing_quick,
)
from .safe_feature_combiner import SafeFeatureCombiner, combine_timeframes_quick, get_safe_feature_combiner
from .temporal_leakage_guard import TemporalLeakageGuard, get_temporal_leakage_guard, validate_temporal_leakage_quick
from .temporal_target_guard import TemporalTargetGuard, generate_targets_quick, get_temporal_target_guard
from .timeframe_alignment_guard import TimeframeAlignmentGuard, get_timeframe_alignment_guard, validate_timeframes_quick

__all__ = [
    # Core guard classes
    'TimeframeAlignmentGuard',
    'SafeFeatureCombiner',
    'TemporalTargetGuard',
    'TemporalLeakageGuard',
    'MacroReleaseTimingGuard',

    # Factory functions
    'get_timeframe_alignment_guard',
    'get_safe_feature_combiner',
    'get_temporal_target_guard',
    'get_temporal_leakage_guard',
    'get_macro_release_timing_guard',

    # Quick validation functions
    'validate_timeframes_quick',
    'combine_timeframes_quick',
    'generate_targets_quick',
    'validate_temporal_leakage_quick',
    'validate_macro_timing_quick'
]

# Module version
__version__ = '1.0.0'
__author__ = 'Trading System Team'
__description__ = 'Temporal safety guards for trading pipeline protection'
