"""CriticalSignalDetector: price shocks, volume spikes, volatility explosions.

Restored from src/archive/ after commit dabe5540 archived it as having "zero
callers". True of the direct-instantiation path it examined, but analysis.yaml
enables it (name: critical_signals, data_mapping: ['price_data']) and
UnifiedAnalyticsEngine loads it from there -- and Stage 7 constructs that
engine. Archiving it took the engine from 3 registered analyzers to 2 with
only a log line.

It had no test coverage, which is part of why it was archived. That is fixed
here: these check the arithmetic of each detector, not merely that it runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector


def _prices(n=120, start=100.0, drift=0.0002, vol=0.005, seed=0):
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(drift, vol, n))
    return pd.DataFrame(
        {
            "close": close,
            "volume": rng.integers(900_000, 1_100_000, n).astype(float),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="D"),
    )


def test_it_is_an_analyzer():
    from src.analytics.interfaces import IAnalyzer

    assert isinstance(CriticalSignalDetector(), IAnalyzer)


def test_a_crash_over_the_window_is_flagged():
    df = _prices()
    df.iloc[60:, df.columns.get_loc("close")] *= 0.85  # -15% step

    flags = CriticalSignalDetector().detect_price_shock(df)

    assert flags.iloc[60:65].any(), "a 15% drop across the window went unnoticed"


def test_a_calm_series_produces_no_price_shock():
    assert not CriticalSignalDetector().detect_price_shock(_prices()).any()


def test_a_jump_is_a_shock_too():
    """It used to be `returns < threshold` with a negative threshold, so a
    melt-up of the same size was invisible in a column called
    price_shock_detected."""
    df = _prices()
    df.iloc[60:, df.columns.get_loc("close")] *= 1.15

    assert CriticalSignalDetector().detect_price_shock(df).iloc[60:65].any()


@pytest.mark.parametrize("threshold", [-0.05, 0.05])
def test_the_threshold_is_read_as_a_magnitude(threshold):
    """Configured as -0.07 for years; both spellings must mean the same."""
    df = _prices()
    df.iloc[60:, df.columns.get_loc("close")] *= 0.85
    detector = CriticalSignalDetector({"price_shock": {"window": 5, "threshold": threshold}})

    assert detector.detect_price_shock(df).iloc[60:65].any()


def test_direction_separates_a_crash_from_a_spike():
    """Symmetry must not cost the information the old behaviour carried."""
    down, up = _prices(), _prices()
    down.iloc[60:, down.columns.get_loc("close")] *= 0.85
    up.iloc[60:, up.columns.get_loc("close")] *= 1.15
    detector = CriticalSignalDetector()

    assert detector.price_shock_direction(down).iloc[60:65].min() == -1
    assert detector.price_shock_direction(up).iloc[60:65].max() == 1
    assert (detector.price_shock_direction(_prices()) == 0).all()


def test_the_configured_thresholds_actually_reach_the_detector():
    """They lived in unified_config.yaml, which the engine never reads for
    this: it passes the analyzer entry's `params`, and there were none, so
    the detector silently ran on its hardcoded defaults."""
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
    from src.config.unified_config_manager import get_current_config

    detector = UnifiedAnalyticsEngine(get_current_config()).analyzers["critical_signals"]

    assert detector.config["price_shock"]["window"] == 3
    assert abs(detector.config["price_shock"]["threshold"]) == pytest.approx(0.07)
    assert detector.config["volume_spike"]["multiplier"] == 4.0


def test_a_volume_spike_is_flagged():
    df = _prices()
    df.iloc[80, df.columns.get_loc("volume")] *= 5

    assert CriticalSignalDetector().detect_volume_spike(df).iloc[80]


def test_ordinary_volume_is_not_a_spike():
    assert not CriticalSignalDetector().detect_volume_spike(_prices()).any()


def test_a_volatility_explosion_is_flagged():
    df = _prices(vol=0.003)
    rng = np.random.default_rng(7)
    idx = df.columns.get_loc("close")
    # A burst of much larger moves late in the series.
    df.iloc[100:, idx] = df.iloc[99, idx] * np.cumprod(
        1 + rng.normal(0, 0.06, len(df) - 100)
    )

    assert CriticalSignalDetector().detect_volatility_explosion(df).iloc[100:].any()


@pytest.mark.parametrize("missing", ["close", "volume"])
def test_a_missing_column_yields_no_signal_rather_than_an_error(missing):
    df = _prices().drop(columns=[missing])
    detector = CriticalSignalDetector()

    for check in (
        detector.detect_price_shock,
        detector.detect_volume_spike,
        detector.detect_volatility_explosion,
    ):
        assert not check(df).any()


def test_analyze_adds_all_three_columns_without_dropping_data():
    df = _prices()
    result = CriticalSignalDetector().analyze(df)

    for column in (
        "price_shock_detected",
        "price_shock_direction",
        "volume_spike_detected",
        "volatility_explosion_detected",
    ):
        assert column in result.columns
    assert len(result) == len(df)
    assert "close" in result.columns


def test_thresholds_come_from_configuration():
    strict = CriticalSignalDetector({"price_shock": {"window": 2, "threshold": -0.01}})
    lax = CriticalSignalDetector({"price_shock": {"window": 2, "threshold": -0.50}})
    df = _prices()
    df.iloc[60:, df.columns.get_loc("close")] *= 0.95

    assert strict.detect_price_shock(df).any()
    assert not lax.detect_price_shock(df).any()
