"""Everything measured out of sample has been measured on one market episode.

`prepare_data_for_models` splits chronologically and hands the champion
`x_test = X.iloc[test_start:]` -- a single contiguous tail. Whatever period
the data happens to end in is the period every out-of-sample number describes.

The walk-forward evaluator already walks several disjoint windows and already
predicts on each of them. It kept only the metrics. A metric cannot be
re-aggregated, priced against a cost model, or turned into an equity curve
afterwards, so "does this hold up in more than one period" was unanswerable
without retraining.

The rows are kept now. This does not change what is trained or how anything is
selected; it stops throwing away the evidence.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
)


def _frame(rows: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(rows)
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=rows, freq="h"),
        "ticker": ["AAPL"] * rows,
        "interval": ["60m"] * rows,
        "f1": signal,
        "f2": rng.standard_normal(rows),
        "target_hourly_up_1h": (signal + rng.normal(0, 0.5, rows) > 0).astype(int),
    })


@pytest.fixture(scope="module")
def result():
    evaluator = PipelineWalkForwardValidationEvaluator(
        WalkForwardValidationConfig(max_folds=3, min_train_rows=400,
                                    validation_rows=120, step_rows=120)
    )
    return evaluator.evaluate(
        _frame(), ticker="AAPL", timeframe="60m",
        target_name="target_hourly_up_1h",
    )


def test_more_than_one_window_is_evaluated(result):
    assert len(result["folds"]) > 1


def test_every_fold_carries_its_own_rows(result):
    for fold in result["folds"]:
        rows = fold.get("validation_predictions")
        assert rows, f"fold {fold['fold']} kept no out-of-sample rows"
        assert len(rows) == fold["validation_window"]["sample_count"]


def test_each_row_says_when_what_was_said_and_what_happened(result):
    row = result["folds"][0]["validation_predictions"][0]
    assert set(row) == {"datetime", "prediction", "actual"}
    assert row["datetime"] is not None
    assert isinstance(row["prediction"], float)
    assert isinstance(row["actual"], float)


def test_the_windows_do_not_overlap(result):
    """Several periods, not one period counted several times."""
    spans = []
    for fold in result["folds"]:
        stamps = [row["datetime"] for row in fold["validation_predictions"]]
        spans.append((min(stamps), max(stamps)))
    spans.sort()
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] < later[0], f"{earlier} overlaps {later}"


def test_the_rows_stay_in_time_order(result):
    for fold in result["folds"]:
        stamps = [row["datetime"] for row in fold["validation_predictions"]]
        assert stamps == sorted(stamps)


def test_a_fold_never_predicts_on_its_own_training_rows(result):
    """The purge gap has to be visible in the timestamps, not only in indices."""
    for fold in result["folds"]:
        train_end = fold["train_window"]["end"]
        validation_start = fold["validation_window"]["start"]
        assert validation_start > train_end, (
            f"fold {fold['fold']} validates from {validation_start}, "
            f"but trained through {train_end}"
        )
        assert fold["purge_window"]["row_count"] >= 1
