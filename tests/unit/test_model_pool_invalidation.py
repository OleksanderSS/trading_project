"""Tests that retraining/re-promoting a model actually invalidates the
stale copy a long-running process may still hold in ModelPool.

Found during a same-session correctness audit: BaseTrainer._promote_champion_file
and _save_model_candidate always write to fixed filenames
(CHAMP_{ticker}_{target}.joblib / model_{ticker}_{target}_{model_type}.joblib),
and model_resolver.py's ModelResolver keys ModelPool.get_model() by that
same filename stem. ModelPool never re-invokes loader_fn on a cache hit —
without invalidation, a process that retrains and then predicts again in
the same interpreter would silently keep serving the pre-retrain object.
"""
import pytest

from src.models.model_pool import ModelPool, get_model_pool
import src.models.model_pool as model_pool_module
from src.training.base_trainer import BaseTrainer


@pytest.fixture(autouse=True)
def _isolated_global_pool():
    """The module-level singleton pool must not leak between tests."""
    model_pool_module._pool = None
    yield
    model_pool_module._pool = None


class _ConcreteTrainerStub(BaseTrainer):
    """BaseTrainer is abstract (ABC) — object.__new__ still enforces
    __abstractmethods__ even bypassing __init__, so a minimal concrete
    subclass is needed to exercise its concrete methods in isolation."""

    def _prepare_ticker_groups(self, plan):
        raise NotImplementedError

    def _train_ticker_group(self, ticker_group, data_context):
        raise NotImplementedError


def test_promote_champion_file_evicts_stale_pool_entry(tmp_path, monkeypatch):
    pool = get_model_pool()
    stem = "CHAMP_AAPL_target_up_1d"
    pool.add_model(stem, model="stale_pre_retrain_model_object")
    assert pool.has_model(stem)

    winner_path = tmp_path / "candidate.joblib"
    winner_path.write_bytes(b"fake model bytes")

    stage = object.__new__(_ConcreteTrainerStub)
    stage.output_dir = tmp_path
    stage.logger = type("L", (), {"warning": lambda *a, **k: None})()

    BaseTrainer._promote_champion_file(stage, winner_path, ticker="AAPL", target="target_up_1d")

    assert not pool.has_model(stem), "stale CHAMP entry must be evicted after re-promotion"


def test_invalidate_model_pool_entry_is_a_safe_noop_when_not_cached():
    # Must not raise even if the model was never in the pool.
    BaseTrainer._invalidate_model_pool_entry("never_cached_model_id")


def test_pool_hit_after_invalidation_forces_loader_fn_to_run_again():
    """End-to-end proof this actually fixes the bug: after invalidation,
    get_model() calls the loader again instead of returning the stale
    cached object."""
    pool = get_model_pool()
    stem = "CHAMP_AAPL_target_up_1d"
    pool.add_model(stem, model="OLD_MODEL")

    # Simulate retrain + re-promotion.
    BaseTrainer._invalidate_model_pool_entry(stem)

    load_calls = []

    def loader():
        load_calls.append(1)
        return "NEW_MODEL"

    result = pool.get_model(stem, loader_fn=loader)

    assert result == "NEW_MODEL"
    assert len(load_calls) == 1  # loader_fn was actually invoked, not skipped
