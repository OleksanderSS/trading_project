"""`--mode light` rebuilt a batch that was already built and already gated.

On 2026-08-18 that was not merely wasteful. The batch had been prepared and
passed verify_batch that morning; rebuilding it a second time inside the
training run died with `MemoryError: unable to allocate 4.17 GiB` in stage 3,
so the training it was meant to precede never ran. Two rebuilds, three hours,
zero champions -- for a frame sitting on disk the whole time.

These tests pin the reuse and, just as importantly, the fallback: a machine
with no prepared batch must still be able to build one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.hybrid_orchestrator import HybridOrchestrator  # noqa: E402


class Bare(HybridOrchestrator):
    """Reach the loader without constructing the whole orchestrator."""
    def __init__(self, root: Path):
        import logging
        self.logger = logging.getLogger('test')
        self._PREPARED_BATCH_DIR = root


def _write_batch(root: Path, name: str = 'main_database') -> None:
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(d / 'features.parquet')
    pd.DataFrame({'target_x': [0, 1, 0]}).to_parquet(d / 'targets.parquet')


def test_a_prepared_batch_is_reused(tmp_path):
    _write_batch(tmp_path)
    f, t = Bare(tmp_path)._load_prepared_batch('main_database')
    assert f is not None and t is not None
    assert len(f) == 3 and 'target_x' in t.columns


def test_the_default_batch_name_is_used_when_none_is_given(tmp_path):
    _write_batch(tmp_path)
    f, _ = Bare(tmp_path)._load_prepared_batch(None)
    assert f is not None


def test_a_missing_batch_falls_back_to_rebuilding(tmp_path):
    # A clean machine must still work; returning None is the signal to build.
    assert Bare(tmp_path)._load_prepared_batch('main_database') == (None, None)


def test_half_a_batch_is_not_a_batch(tmp_path):
    d = tmp_path / 'main_database'
    d.mkdir(parents=True)
    pd.DataFrame({'a': [1]}).to_parquet(d / 'features.parquet')
    assert Bare(tmp_path)._load_prepared_batch('main_database') == (None, None)


def test_an_unreadable_batch_falls_back_rather_than_killing_the_run(tmp_path):
    d = tmp_path / 'main_database'
    d.mkdir(parents=True)
    (d / 'features.parquet').write_text('not a parquet file', encoding='utf-8')
    (d / 'targets.parquet').write_text('nor is this', encoding='utf-8')
    assert Bare(tmp_path)._load_prepared_batch('main_database') == (None, None)


def test_a_named_batch_does_not_pick_up_a_different_one(tmp_path):
    _write_batch(tmp_path, 'main_database')
    assert Bare(tmp_path)._load_prepared_batch('some_other_batch') == (None, None)
