"""Out-of-sample evidence must come from more than the tail of the data.

REGISTER #47. `prepare_data_for_models` splits once and chronologically:
`x_test = X.iloc[test_start:]`. So every number this project has called "out of
sample" was measured on ONE market episode -- whichever one the batch happens
to end in. A Sharpe from 2021-2023 and a Sharpe from four disjoint windows are
different claims, and only the second survives the question "does it hold in
another period".

The walk-forward evaluator has kept `validation_predictions` per fold since it
was written, with a comment saying exactly why. Measured 2026-09-05: NOTHING
READ THEM. Produced, unit-tested, zero consumers -- the third time in one day
that shape appeared, after `apply_seal` (#264) and `universe_as_of` (Р46). A
mechanism nobody calls is indistinguishable from one that does not exist.

So the fold rows now travel: `_walk_forward_stability` carries them out as
`fold_predictions`, the champion payload keeps them under
`walk_forward_stability`, and `_write_holdout_predictions` writes them beside
the tail with a `window` column naming where each row came from.

WHAT THIS FILE FORBIDS: the artifact silently returning to one episode. That
is the failure mode -- not a crash, just a file that looks the same and answers
less.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage


def _record(stamp: str, prediction: float, actual: float) -> dict:
    return {"datetime": stamp, "prediction": prediction,
            "probability": prediction, "actual": actual}


def _champion(with_folds: bool = True) -> dict:
    payload = {
        "ticker": "AAPL",
        "timeframe": "1d",
        "target_name": "target_return_1d",
        "model_type": "lightgbm",
        "holdout_predictions": [
            _record("2023-01-03", 1.0, 0.4),
            _record("2023-01-04", 0.0, -0.2),
        ],
    }
    if with_folds:
        payload["walk_forward_stability"] = {
            "passed": True,
            "fold_count": 2,
            "fold_predictions": [
                {"fold": 1, "rows": [_record("2019-05-02", 1.0, 0.1)]},
                {"fold": 2, "rows": [_record("2021-05-03", 0.0, -0.3)]},
            ],
        }
    return payload


def _write(champions: dict, tmp_path: Path, monkeypatch) -> pd.DataFrame:
    monkeypatch.chdir(tmp_path)
    path = ModelingStage._write_holdout_predictions(champions)
    assert path is not None, "nothing was written at all"
    return pd.read_parquet(path)


def test_the_artifact_carries_the_folds_not_only_the_tail(tmp_path, monkeypatch):
    frame = _write({"AAPL::target_return_1d": _champion()}, tmp_path, monkeypatch)

    assert set(frame["window"]) == {"holdout", "fold_1", "fold_2"}, (
        "the artifact is back to a single contiguous tail, so everything it "
        "supports was measured on one market episode (#47)"
    )
    assert len(frame) == 4


def test_every_row_names_the_window_it_came_from(tmp_path, monkeypatch):
    frame = _write({"AAPL::target_return_1d": _champion()}, tmp_path, monkeypatch)

    assert frame["window"].notna().all()
    assert (frame["window"].astype(str).str.len() > 0).all(), (
        "a row whose window nobody can name is a row nobody can weigh"
    )


def test_the_windows_are_disjoint_in_time(tmp_path, monkeypatch):
    """Four numbers from one stretch are not four independent looks. The point
    of keeping folds is that they sit in different periods."""
    frame = _write({"AAPL::target_return_1d": _champion()}, tmp_path, monkeypatch)
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    years = frame.groupby("window")["datetime"].min().dt.year
    assert years.nunique() == len(years), (
        f"the windows share a year, so they are not separate episodes: "
        f"{years.to_dict()}"
    )


def test_a_champion_without_folds_still_writes_its_tail(tmp_path, monkeypatch):
    """The fold rows are an addition, not a precondition. A context too short
    to build folds must still contribute its holdout, or wiring #47 would have
    silently deleted evidence."""
    frame = _write({"AAPL::target_return_1d": _champion(with_folds=False)},
                   tmp_path, monkeypatch)

    assert set(frame["window"]) == {"holdout"}
    assert len(frame) == 2


def test_nothing_at_all_is_reported_rather_than_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ModelingStage._write_holdout_predictions({}) is None
    assert ModelingStage._write_holdout_predictions(
        {"AAPL::t": {"holdout_predictions": []}}) is None


def test_the_fold_series_has_a_consumer_at_all():
    """The defect this closes, stated as a check rather than as prose.

    `validation_predictions` was produced and unit-tested and read by nobody
    for as long as it existed. If the name stops appearing on the consuming
    side, it is back to being a mechanism that only its own tests can see.
    """
    import inspect

    source = inspect.getsource(ModelingStage._walk_forward_stability)
    assert "validation_predictions" in source, (
        "the stability payload no longer picks up the fold series, so the "
        "walk-forward rows are unread again (#47)"
    )
    writer = inspect.getsource(ModelingStage._write_holdout_predictions)
    assert "fold_predictions" in writer and "window" in writer, (
        "the artifact writer no longer reads the fold series"
    )
