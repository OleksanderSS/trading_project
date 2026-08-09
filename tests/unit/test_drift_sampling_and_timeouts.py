"""Two ways Stage 7 reported checks it had not performed.

The 2026-08-09 `continue` run finished fast after a five-hour hang was
"fixed" by capping the drift check at 100 features and cutting the
per-context analyzer timeout to 30 seconds. Both changes were reported as
clean. The artifacts disagreed:

    feature_drift: failed   54 of 66 contexts
    feature_drift: OK       12

and every one of the 54 carried `error: ""`. An empty message is the
signature of concurrent.futures.TimeoutError, which has no text -- so the
54 were timeouts, indistinguishable in the results from crashes, and
summarised upstream as "checked without errors". Recovering that took
reading the raw artifacts.

The 12 that did run sampled `valid_common[:100]` after an alphabetical
sort: AATR_14_15m, AATR_14_1d, AATR_14_60m, ABB_Lower_15m... -- the first
hundred names out of ~1,940. Whole families (volume_*, sentiment_*,
state_*) could drift untouched while the report said none was detected.

Neither change was wrong to make. Both stated more than they did.
"""
from __future__ import annotations

import logging
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from src.monitoring.feature_drift_monitor import FeatureDriftMonitor


# --------------------------------------------------- which features are checked


def _sample(names):
    """The selection the monitor performs, isolated from Evidently."""
    names = sorted(names)
    cap = FeatureDriftMonitor.MAX_DRIFT_FEATURES
    if len(names) <= cap:
        return names
    step = len(names) / cap
    return [names[int(i * step)] for i in range(cap)]


def test_the_sample_spans_the_whole_feature_space():
    """An alphabetical prefix reaches 'B'; a stride reaches 'z'."""
    names = [f"{chr(ord('a') + i // 100)}_feature_{i:04d}" for i in range(1940)]

    chosen = _sample(names)

    assert len(chosen) == FeatureDriftMonitor.MAX_DRIFT_FEATURES
    assert chosen[0].startswith("a_")
    assert chosen[-1] > "s_", (
        f"the sample stops at {chosen[-1]}; it is a prefix again, not a spread"
    )


def test_every_region_of_the_sorted_space_is_represented():
    names = [f"f{i:04d}" for i in range(1940)]

    chosen = set(_sample(names))

    for start in range(0, 1940, 200):
        window = {f"f{i:04d}" for i in range(start, min(start + 200, 1940))}
        assert chosen & window, f"nothing sampled from rows {start}..{start+200}"


def test_a_small_feature_set_is_taken_whole():
    names = [f"f{i}" for i in range(20)]

    assert _sample(names) == sorted(names)


def test_the_cap_is_a_named_constant():
    """It governs what the drift number means, so it is not an inline 100."""
    assert isinstance(FeatureDriftMonitor.MAX_DRIFT_FEATURES, int)
    assert FeatureDriftMonitor.MAX_DRIFT_FEATURES > 0


def test_the_result_declares_that_it_sampled():
    """"No drift over 100 of 1,940 features" is not "no drift"."""
    import inspect

    source = inspect.getsource(FeatureDriftMonitor.check_drift)

    for field in ("features_available", "features_sampled", "features_checked"):
        assert field in source, f"the result does not report {field}"


# ------------------------------------------------------------ timeouts speak


class _StuckAnalyzer:
    def analyze(self, data, **kwargs):
        import time

        time.sleep(30)
        return {"status": "OK"}


def _engine_with(analyzer, monkeypatch):
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    engine = object.__new__(UnifiedAnalyticsEngine)
    engine.analyzers = {"stuck": analyzer}
    engine.analyzer_data_map = {"stuck": ["price_data"]}
    engine.analyzer_registration_report = {"stuck": {"status": "registered"}}
    engine.analyzer_configs = []
    engine.logger = logging.getLogger("engine-test")
    return engine


def test_a_timeout_says_it_timed_out(monkeypatch):
    """The defect that made 54 failures unreadable: str(TimeoutError) is ''."""
    assert str(FuturesTimeoutError()) == "", (
        "if this ever gains a message, the empty-error diagnosis below "
        "needs revisiting"
    )

    import inspect

    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    source = inspect.getsource(UnifiedAnalyticsEngine._run_analyzers_parallel) \
        if hasattr(UnifiedAnalyticsEngine, "_run_analyzers_parallel") \
        else inspect.getsource(UnifiedAnalyticsEngine)

    assert "FuturesTimeoutError" in source, (
        "timeouts are still handled by the generic except, so they are "
        "recorded with an empty error message"
    )
    assert "timed out after" in source
    assert "timeout_seconds" in source


def test_a_message_less_exception_still_names_its_type():
    """Any exception with an empty str() would have produced the same
    unreadable record, not only TimeoutError."""
    import inspect

    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    source = inspect.getsource(UnifiedAnalyticsEngine)

    assert "no message" in source, (
        "an exception whose str() is empty is still recorded as error=''"
    )


# ----------------------------------------------------- constants cannot drift


def test_a_constant_column_is_excluded():
    """Evidently does not ignore a constant column -- it raises.

        Too many bins for data range. Cannot create 10 finite-sized bins

    and the exception fails the whole context, not just that column. All 66
    contexts died that way on the 2026-08-09 evening run. They had survived
    the morning only because the selection took the first hundred names
    alphabetically (AATR_*, ABB_*, all continuous) and never reached the
    constants; sampling across the whole space found them at once.
    """
    import inspect

    source = inspect.getsource(FeatureDriftMonitor.check_drift)

    assert "nunique" in source, (
        "nothing excludes zero-variance columns, so one constant feature "
        "still fails an entire context"
    )


def test_the_variance_filter_runs_before_the_sample_is_drawn():
    """Filtering after sampling would keep the failure and merely shrink the
    sample: a constant column drawn into the 100 would still raise."""
    import inspect

    source = inspect.getsource(FeatureDriftMonitor.check_drift)
    filter_at = source.index("nunique")
    sample_at = source.index("MAX_DRIFT_FEATURES")

    assert filter_at < sample_at


def test_the_real_batch_has_constant_columns_to_exclude():
    """The measurement the filter rests on: 198 of 774 columns that pass the
    non-null test do not vary."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(path)
    rows = frame[(frame.ticker == "XLF") & (frame.interval == "60m")]
    if rows.empty:
        pytest.skip("XLF 60m not in this batch")

    ref, cur = rows.iloc[: len(rows) // 2], rows.iloc[len(rows) // 2 :]
    numeric = [
        c for c in rows.select_dtypes(include=[np.number]).columns
        if not c.startswith("target_") and c not in ("hash", "interval")
    ]
    populated = [
        c for c in numeric if ref[c].count() >= 10 and cur[c].count() >= 10
    ]
    varying = [
        c for c in populated
        if ref[c].nunique(dropna=True) > 1 and cur[c].nunique(dropna=True) > 1
    ]

    assert len(populated) > len(varying), (
        "no constant columns left to trip Evidently -- if this is genuinely "
        "true now, the filter is harmless, but check before removing it"
    )


# ------------------------------------------- a hash that failed is not a key


def test_a_failed_fingerprint_never_reuses_a_cached_result():
    """The fallback identifies a frame by key + shape + head(3), so two
    different datasets of the same shape with the same first rows collide --
    and the engine would return one's analysis for the other.

    Found because an AttributeError sent every call down that path on
    2026-08-09 and two deliberately different data maps hashed identically.
    The fallback is now a guaranteed cache MISS: recomputing is cheap,
    serving the wrong context's analysis is not.
    """
    import pandas as pd

    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    engine = object.__new__(UnifiedAnalyticsEngine)
    engine.analyzer_configs = []
    engine.analyzers = {}
    engine.analyzer_data_map = {}
    engine.analyzer_registration_report = {}

    # A column of lists: hash_pandas_object raises "unhashable type: 'list'"
    # and the fallback takes over. A stray object() would NOT do it -- the
    # main path stringifies unknown values, so str(object()) sails through.
    # The first version of this test used one and passed for the wrong
    # reason, asserting a difference that never had to exist.
    frame = pd.DataFrame({"a": [[1, 2], [3, 4], [5, 6]]})
    data_map = {"price_data": frame}

    first = engine._generate_data_hash(data_map)
    second = engine._generate_data_hash(data_map)

    assert first != second, (
        "an uncomputable fingerprint still produces a stable key, so a "
        "cached result can be served for data nobody could identify"
    )


def test_the_contract_hash_survives_a_bare_engine():
    """_contract_timeout is set in __init__, but every construction path has
    to reach the hash -- an AttributeError here does not surface, it drops
    into the colliding fallback."""
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    engine = object.__new__(UnifiedAnalyticsEngine)
    engine.analyzer_configs = []
    engine.analyzers = {}
    engine.analyzer_data_map = {}
    engine.analyzer_registration_report = {}

    assert engine._analysis_contract_hash()
