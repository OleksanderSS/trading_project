"""Continue mode reached for champions two weeks older than the batch.

`--mode light` appends every run to `light_models_results.json` under `runs[]`.
`load_colab_results` never looked at that name -- it reads
`trained_models_metadata.json`, `colab_results.json` and
`evaluation_results.json` -- so it fell through to `colab_results.json`, which
is written once and never again.

Measured on 2026-08-23: the freshest run had just written 97 champions, and
continue mode would have carried 660 models dated 2026-08-08 into stages 5 to
7. No error, no warning, and a result that reads as current.

The run is chosen by its own timestamp, not by which key happened to be
assigned last. Two entries in `files_to_load` already map to `models_metadata`,
so ordering inside a dict literal was deciding which artifact won.
"""

from __future__ import annotations

import json

import pytest

from src.pipeline.hybrid.colab_manager import ColabManager


@pytest.fixture
def manager():
    import logging
    instance = ColabManager.__new__(ColabManager)
    instance.logger = logging.getLogger("colab-probe")
    return instance


def _write(tmp_path, runs):
    (tmp_path / "light_models_results.json").write_text(
        json.dumps({"batch_name": "b", "runs": runs}), encoding="utf-8"
    )


def test_the_newest_run_wins_over_an_older_artifact(manager, tmp_path):
    _write(tmp_path, [
        {"timestamp": "20260808_185131", "models_metadata": {f"old{i}": {} for i in range(660)}},
        {"timestamp": "20260823_062255", "models_metadata": {f"new{i}": {} for i in range(97)}},
    ])
    results = {"models_metadata": {f"stale{i}": {} for i in range(660)}}
    manager._prefer_latest_local_run(tmp_path, results)

    assert len(results["models_metadata"]) == 97
    assert all(key.startswith("new") for key in results["models_metadata"])


def test_order_in_the_file_does_not_decide_it(manager, tmp_path):
    """Newest last is a convention, not a guarantee; the timestamp decides."""
    _write(tmp_path, [
        {"timestamp": "20260823_062255", "models_metadata": {"new": {}}},
        {"timestamp": "20260808_185131", "models_metadata": {"old": {}}},
    ])
    results: dict = {}
    manager._prefer_latest_local_run(tmp_path, results)
    assert list(results["models_metadata"]) == ["new"]


def test_a_run_with_no_models_is_not_chosen(manager, tmp_path):
    """An empty run is a run that produced nothing, not the newest truth."""
    _write(tmp_path, [
        {"timestamp": "20260801_000000", "models_metadata": {"real": {}}},
        {"timestamp": "20260823_062255", "models_metadata": {}},
    ])
    results: dict = {}
    manager._prefer_latest_local_run(tmp_path, results)
    assert list(results["models_metadata"]) == ["real"]


def test_nothing_is_touched_when_the_file_is_absent(manager, tmp_path):
    """A Colab-only batch has no local runs, and must keep what it loaded."""
    results = {"models_metadata": {"from_colab": {}}}
    manager._prefer_latest_local_run(tmp_path, results)
    assert list(results["models_metadata"]) == ["from_colab"]


def test_unreadable_json_leaves_the_loaded_metadata_alone(manager, tmp_path):
    (tmp_path / "light_models_results.json").write_text("{ not json", encoding="utf-8")
    results = {"models_metadata": {"from_colab": {}}}
    manager._prefer_latest_local_run(tmp_path, results)
    assert list(results["models_metadata"]) == ["from_colab"]
