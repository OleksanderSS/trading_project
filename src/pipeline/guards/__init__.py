"""
Pipeline Guards Module - Temporal Safety Protection
==============================================

Runtime temporal safety checks for the feature pipeline.

Key Components:
- TemporalLeakageGuard: rolling-window budgets and future-value checks

FOUR GUARDS WERE REMOVED FROM HERE on 2026-08-02, to
src/archive/guards_superseded/. None of their methods had a single call
site in src/; each had been overtaken by machinery built later, and their
presence in this namespace advertised protection the pipeline did not have:

- TimeframeAlignmentGuard, SafeFeatureCombiner -> superseded by
  src/pipeline/stages/feature_engineering/timeframe_context.py
  (BackwardTimeframeContextAssembler), which is wired, produces the ctx_1d_*
  columns visible in the export, and additionally records
  ctx_<tf>_available_at / _source_datetime per joined column while excluding
  target-like columns.

- TemporalTargetGuard -> superseded by src/targets/. It emitted targets under
  the SAME NAMES the live builder uses (target_return_1d, target_return_5d)
  from a raw shift(-n) in BARS, with no timeframe contract and no boundary
  masking, so on 60m data its "target_return_1d" was a one-hour return. The
  name collision is what made it dangerous rather than merely unused.

- MacroReleaseTimingGuard -> its hardcoded release schedule (GDP 08:30 ET at
  quarter end, CPI 08:30 monthly, ...) is superseded by reading the real
  publication timestamp at collection time. It pointed at a real gap --
  fred_data.realtime_start is a DATE, so an intraday model could read a
  figure hours before publication -- and that gap is now closed in
  CollectionStage._defer_date_only_availability, conservatively and without
  a schedule table that drifts.

Feature-vs-target leakage is checked by
src/features/validation/feature_leakage_guard.py, which detects by
CORRELATION rather than by naming.
"""

from .temporal_leakage_guard import (
    TemporalLeakageGuard,
    get_temporal_leakage_guard,
    validate_temporal_leakage_quick,
)

__all__ = [
    'TemporalLeakageGuard',
    'get_temporal_leakage_guard',
    'validate_temporal_leakage_quick',
]

# Module version
__version__ = '2.0.0'
__author__ = 'Trading System Team'
__description__ = 'Temporal safety guards for trading pipeline protection'
