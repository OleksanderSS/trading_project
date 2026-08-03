"""Stage 5 looked for a Colab results file that nothing produces.

colab_clean_cell.py._save_results_summary writes colab_results.json.
ModelResolver._load_heavy_models_from_disk looked only for
colab_results_summary.json -- a name no writer in the project uses -- and
when it was absent did nothing, silently. Every heavy model Colab trained
was invisible to prediction, which then ran on light models alone and
reported success.

hybrid/results_processor._find_results_file had always tried all three
names. Two readers of one artifact disagreeing about its filename is
precisely how this survived.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.pipeline.stages.prediction.model_resolver import ModelResolver


@pytest.fixture()
def resolver():
    instance = object.__new__(ModelResolver)
    instance.logger = logging.getLogger("model-resolver-test")
    return instance


def _write(batch_dir: Path, name: str, payload: dict) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / name).write_text(json.dumps(payload), encoding="utf-8")


METADATA = {"models_metadata": {"AAPL_target_up_1d_lstm": {"winner": "lstm"}}}


def test_the_name_colab_actually_writes_is_found(tmp_path, resolver):
    """The defect: this is the file that exists after a real Colab run."""
    _write(tmp_path, "colab_results.json", METADATA)
    found: dict = {}

    resolver._load_heavy_models_from_disk(tmp_path, found)

    assert "AAPL_target_up_1d_lstm" in found


def test_the_historical_name_still_wins_when_both_exist(tmp_path, resolver):
    _write(tmp_path, "colab_results_summary.json",
           {"models_metadata": {"from_summary": {}}})
    _write(tmp_path, "colab_results.json", {"models_metadata": {"from_results": {}}})
    found: dict = {}

    resolver._load_heavy_models_from_disk(tmp_path, found)

    assert "from_summary" in found and "from_results" not in found


def test_a_missing_file_is_announced_rather_than_ignored(tmp_path, resolver, caplog):
    """"Colab produced nothing" and "Colab's output is under a name nobody
    looked for" are indistinguishable from the outside, and only the second
    is a defect."""
    found: dict = {}

    with caplog.at_level(logging.WARNING):
        resolver._load_heavy_models_from_disk(tmp_path, found)

    assert found == {}
    assert any("No Colab results file" in r.message for r in caplog.records)


def test_the_ticker_results_shape_still_routes_to_its_handler(tmp_path, resolver):
    _write(tmp_path, "colab_results.json", {
        "ticker_results": {
            "AAPL": {"timeframes": {"1d": {"results": {
                "target_up_1d": {"models": {"lstm": {"metrics": {"accuracy": 0.6}}}}
            }}}}
        }
    })
    found: dict = {}

    resolver._load_heavy_models_from_disk(tmp_path, found)

    assert "AAPL_target_up_1d_lstm" in found
    assert found["AAPL_target_up_1d_lstm"]["model_category"] == "heavy"


def test_a_corrupt_file_does_not_take_down_the_stage(tmp_path, resolver, caplog):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "colab_results.json").write_text("{not json", encoding="utf-8")
    found: dict = {}

    with caplog.at_level(logging.WARNING):
        resolver._load_heavy_models_from_disk(tmp_path, found)

    assert found == {}


def test_both_readers_agree_on_the_candidate_filenames():
    """The two readers drifting apart is the root cause, not the symptom."""
    import inspect

    from src.pipeline.hybrid.results_processor import ResultsProcessor

    processor_source = inspect.getsource(ResultsProcessor._find_results_file)

    for name in ModelResolver._COLAB_RESULT_FILENAMES:
        assert name in processor_source, (
            f"{name} is accepted by ModelResolver but not by ResultsProcessor"
        )
