"""The cache key must not cost more than the work it caches.

Stage 7 partitions prices by (ticker, cadence) and calls `run_full_analysis`
once per partition, handing it the same features frame each time. Every call
began by hashing the whole data map -- including that unchanged frame -- to
look up a persistent cache.

Measured 2026-09-01: `hash_pandas_object` on the real features frame
(1,243,783 x 439) takes 28.23 s, while the analysis it guards takes 0.05 s.
Over ~330 contexts that is about two and a half hours, spent re-deriving the
same digest, for a lookup that cannot hit -- `price_data` differs on every
call by construction, so the composite key is new every time (REGISTER #220).

These tests pin the fix at the level that matters: the same frame object is
hashed once, a different frame is hashed again, and the digest for the same
content is unchanged by the caching.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def engine():
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine

    instance = UnifiedAnalyticsEngine.__new__(UnifiedAnalyticsEngine)
    instance._hash_cache = {}
    return instance


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    b = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    return a, b


def test_the_same_frame_object_is_hashed_once(engine, frames, monkeypatch):
    a, _ = frames
    calls = {"n": 0}
    real = pd.util.hash_pandas_object

    def counting(obj, **kwargs):
        calls["n"] += 1
        return real(obj, **kwargs)

    monkeypatch.setattr(pd.util, "hash_pandas_object", counting)

    first = engine._frame_content_hash(a)
    for _ in range(9):
        engine._frame_content_hash(a)

    assert calls["n"] == 1, (
        f"the frame was hashed {calls['n']} times; Stage 7 does this once per "
        "context and there were ~330 contexts"
    )
    assert engine._frame_content_hash(a) == first


def test_a_different_frame_is_hashed_separately(engine, frames):
    a, b = frames
    assert engine._frame_content_hash(a) != engine._frame_content_hash(b)


def test_the_digest_matches_the_uncached_computation(engine, frames):
    """Caching must not change the answer, only how often it is computed."""
    import hashlib

    a, _ = frames
    expected = hashlib.sha256(
        pd.util.hash_pandas_object(a, index=True, categorize=True).values.tobytes()
    ).hexdigest()
    assert engine._frame_content_hash(a) == expected


def test_the_cache_does_not_grow_without_bound(engine):
    rng = np.random.default_rng(1)
    for _ in range(engine._HASH_CACHE_SIZE * 3):
        engine._frame_content_hash(
            pd.DataFrame(rng.normal(size=(20, 3)), columns=list("xyz"))
        )
    assert len(engine._hash_cache) <= engine._HASH_CACHE_SIZE


def test_the_cache_holds_its_frames_so_ids_cannot_be_reused(engine):
    """`id()` is only stable while the object is alive.

    Without holding a reference, a frame could be collected, a new one could
    land on the same address, and the cache would return the dead frame's
    digest for it.
    """
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("xyz"))
    engine._frame_content_hash(frame)
    held = engine._hash_cache[id(frame)][0]
    assert held is frame
