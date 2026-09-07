"""
Cache key policy.

The cache stores whole DataFrames keyed by a SHA-256 of (key, params, db_salt).
Anything the key fails to distinguish is a frame served from one context into
another — which is how a stale frame reaches the pipeline while looking fresh.
"""
import pandas as pd

from src.core.cache.cache_manager import CacheManager


class _SaltStub:
    """Stands in for a constructed CacheManager: _get_cache_key needs only the salt."""

    def __init__(self, salt="SALT"):
        self.db_salt = salt


def _key(stub, *args, **kwargs):
    return CacheManager._get_cache_key(stub, *args, **kwargs)


def test_empty_params_do_not_collide_with_absent_params():
    """
    ``params={}`` must not read back what was stored under no params at all.

    The guard used to be ``if params:``, so every falsy params value — {}, 0,
    "" — dropped out of the key and aliased onto the no-params entry.
    """
    stub = _SaltStub()
    absent = _key(stub, "run", None)

    assert _key(stub, "run", {}) != absent
    assert _key(stub, "run", 0) != absent
    assert _key(stub, "run", "") != absent
    assert _key(stub, "run", {"ticker": "AAPL"}) != absent


def test_different_params_produce_different_keys():
    stub = _SaltStub()
    assert _key(stub, "run", {"ticker": "AAPL"}) != _key(stub, "run", {"ticker": "MSFT"})


def test_param_order_does_not_change_the_key():
    """Params are a mapping, so key order must not split one entry into two."""
    stub = _SaltStub()
    assert _key(stub, "run", {"a": 1, "b": 2}) == _key(stub, "run", {"b": 2, "a": 1})


def test_db_salt_separates_entries_when_tracked_tables_change():
    assert _key(_SaltStub("before"), "run", {"x": 1}) != _key(_SaltStub("after"), "run", {"x": 1})


def test_collectors_namespace_is_deliberately_salt_free():
    """
    The collectors namespace drops the DB salt on purpose: a collector's job is
    to detect what is new, so its cache must not be invalidated by the very
    rows it just wrote. This test pins that as intended, not accidental.
    """
    same = _key(_SaltStub("before"), "run", {"x": 1}, namespace="collectors")
    other = _key(_SaltStub("after"), "run", {"x": 1}, namespace="collectors")
    assert same == other


def test_version_separates_entries_written_by_different_code():
    """
    The point of the version component: a collector that changed its parsing
    must not read back the payload its previous version wrote.
    """
    stub = _SaltStub()
    before = _key(stub, "run", {"x": 1}, namespace="collectors", version="aaaaaaaaaaaa")
    after = _key(stub, "run", {"x": 1}, namespace="collectors", version="bbbbbbbbbbbb")

    assert before != after


def test_no_version_is_not_the_same_entry_as_some_version():
    stub = _SaltStub()
    assert _key(stub, "run", {"x": 1}) != _key(stub, "run", {"x": 1}, version="aaaaaaaaaaaa")


def test_same_version_is_stable():
    stub = _SaltStub()
    assert _key(stub, "run", {"x": 1}, version="v1") == _key(stub, "run", {"x": 1}, version="v1")


def test_collector_cache_version_is_per_class_and_stable():
    """
    cache_version fingerprints the collector's own source, so a collector
    invalidates its own entries when it changes — and only its own.
    """
    from src.data.collectors.fred_collector import FredCollector
    from src.data.collectors.huggingface_collector import HuggingfaceCollector

    hf = object.__new__(HuggingfaceCollector)
    fred = object.__new__(FredCollector)

    assert hf.cache_version != fred.cache_version
    assert hf.cache_version == object.__new__(HuggingfaceCollector).cache_version
    assert len(hf.cache_version) == 12


def test_cached_frame_columns_are_not_checked_against_the_caller():
    """
    Documents the open gap behind "stale legacy cache columns".

    Collector payloads are now versioned by the collector's source
    fingerprint. What is still unguarded is the *value* side: get() does not
    compare a returned frame's columns against what the caller expects, so a
    caller that passes no version still gets whatever was stored. This pins
    that remaining gap.
    """
    stub = _SaltStub()
    key_v1 = _key(stub, "enriched_features", {"ticker": "AAPL"})
    key_v2 = _key(stub, "enriched_features", {"ticker": "AAPL"})

    # Same inputs, same key — regardless of which code version produced the frame.
    assert key_v1 == key_v2

    old_frame = pd.DataFrame({"close": [1.0], "LEGACY_COL": [2.0]})
    new_frame = pd.DataFrame({"close": [1.0], "SHARPE_RATIO": [2.0]})
    assert list(old_frame.columns) != list(new_frame.columns)
