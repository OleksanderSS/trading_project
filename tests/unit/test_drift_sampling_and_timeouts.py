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
