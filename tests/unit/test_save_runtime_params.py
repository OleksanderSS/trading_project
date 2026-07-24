"""Tests for run_hybrid_pipeline._save_runtime_params's full-mode cleanup.

Context: a full-mode run must never be silently trained with a stale
runtime_params.json left over from an earlier --epochs/--test-* invocation.
colab_clean_cell.py's ConfigLoader only checks whether the file *exists* --
not whether the CURRENT run is actually in test mode -- so a leftover file
(e.g. epochs=1 from a quick smoke test) forces every model in a later "full"
batch to train with that old, near-untrained configuration while
batch_metadata.json still reports test_mode: false, making the run look
like real training when it wasn't. Reproduced live: data/colab/accumulated/
main_database/runtime_params.json (epochs=1, max_iterations=1, created
2026-07-23T17:10) silently poisoned a same-day "full mode" Colab training
batch -- 75% of models were skipped as already-trained and the rest trained
for a single epoch.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run_hybrid_pipeline import _save_runtime_params


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        mode="continue",
        test_ticker=None,
        test_target=None,
        test_model=None,
        epochs=None,
        max_iterations=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_full_mode_removes_stale_test_mode_runtime_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch_dir = tmp_path / "data" / "colab" / "accumulated" / "main_database"
    batch_dir.mkdir(parents=True)
    stale = batch_dir / "runtime_params.json"
    stale.write_text(json.dumps({"epochs": 1, "max_iterations": 1}), encoding="utf-8")

    await _save_runtime_params(_args(), "main_database")

    assert not stale.exists()


@pytest.mark.asyncio
async def test_full_mode_is_a_no_op_when_no_stale_file_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch_dir = tmp_path / "data" / "colab" / "accumulated" / "main_database"
    batch_dir.mkdir(parents=True)

    await _save_runtime_params(_args(), "main_database")

    assert not (batch_dir / "runtime_params.json").exists()


@pytest.mark.asyncio
async def test_test_mode_still_creates_runtime_params_with_requested_epochs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    await _save_runtime_params(_args(epochs=2, max_iterations=3), "main_database")

    written = tmp_path / "data" / "colab" / "accumulated" / "main_database" / "runtime_params.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["epochs"] == 2
    assert payload["max_iterations"] == 3


@pytest.mark.asyncio
async def test_test_ticker_alone_still_creates_runtime_params(tmp_path, monkeypatch):
    """has_test_params must trigger on any test signal, not just epochs."""
    monkeypatch.chdir(tmp_path)

    await _save_runtime_params(_args(test_ticker="AAPL"), "main_database")

    written = tmp_path / "data" / "colab" / "accumulated" / "main_database" / "runtime_params.json"
    assert written.exists()
