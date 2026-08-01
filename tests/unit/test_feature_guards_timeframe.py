"""The leakage guard's window budget is looked up by timeframe key.

market_data_raw stores interval='1h' (9,549 rows, verified against the live
database), while TemporalLeakageGuard.SAFE_ROLLING_CONFIGS is keyed '15m',
'60m', '1d'. Handing the raw value straight through made

    config = self.SAFE_ROLLING_CONFIGS.get(timeframe, {})
    max_periods = config.get('max_periods', 100)

miss for hourly data, so the budget fell back to 100 instead of 168 -- and
"Rolling window too large" is one of the conditions FeatureGuards treats as
fatal, raising ValueError. price_filter.py already carried a local
workaround accepting both spellings, which is what a split convention looks
like before anyone names it.

Recorded here, not fixed here: with the spelling aligned, this check still
inspects nothing in practice. Measured against the 713 real feature names in
diagnostic_reports/feature_lineage_report.json:

    rolling_<N> window pattern      0 matches
    future_price patterns           0
    future_volume patterns          0
    future_high_low patterns        0
    lookahead_indicators patterns   0

The patterns are written for a naming convention this project does not use
(its features are AATR_14, ACCELERATION_10, ABB_Upper). Making them match
real leakage is a design decision, not a rename, and belongs with the
target-column contract rather than in this file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.guards import FeatureGuards


@pytest.fixture()
def guards():
    return FeatureGuards(mode='prepare')


def _frame(column: str, value) -> pd.DataFrame:
    return pd.DataFrame({
        column: [value] * 4,
        'datetime': pd.date_range('2026-01-01', periods=4, freq='h'),
        'close': [1.0, 2.0, 3.0, 4.0],
    })


@pytest.mark.parametrize("stored,expected", [
    ("1h", "60m"),      # what market_data_raw actually holds
    ("60m", "60m"),
    ("15m", "15m"),
    ("1d", "1d"),
    ("daily", "1d"),
    ("30min", "30m"),
])
def test_the_stored_spelling_is_normalised(guards, stored, expected):
    assert guards._infer_timeframe(_frame('interval', stored)) == expected


def test_the_normalised_key_finds_a_window_budget(guards):
    """The point of normalising: '1h' used to miss and fall back to 100."""
    from src.pipeline.guards.temporal_leakage_guard import TemporalLeakageGuard

    configs = TemporalLeakageGuard().SAFE_ROLLING_CONFIGS
    resolved = guards._infer_timeframe(_frame('interval', '1h'))

    assert resolved in configs
    assert configs[resolved]['max_periods'] == 168


def test_a_timeframe_column_is_used_when_interval_is_absent(guards):
    assert guards._infer_timeframe(_frame('timeframe', '15m')) == '15m'


def test_a_mixed_frame_yields_no_timeframe(guards):
    """A combined feature set spans timeframes, and no single window budget
    applies to it -- which is also why this check is dormant in the real
    pipeline."""
    frame = pd.DataFrame({
        'interval': ['15m', '15m', '1h', '1d'],
        'datetime': pd.date_range('2026-01-01', periods=4, freq='h'),
        'close': [1.0, 2.0, 3.0, 4.0],
    })
    assert guards._infer_timeframe(frame) is None


def test_a_frame_without_either_column_yields_no_timeframe(guards):
    frame = pd.DataFrame({'close': [1.0, 2.0]})
    assert guards._infer_timeframe(frame) is None


def test_guards_still_pass_a_clean_frame_through(guards):
    """apply_guards is on the Stage 3 path; normalising must not change what
    it returns for ordinary input."""
    frame = _frame('interval', '1h')
    result = guards.apply_guards(frame)

    assert len(result) == len(frame)
    assert 'close' in result.columns
