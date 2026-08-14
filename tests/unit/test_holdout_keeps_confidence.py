"""Stage 4 threw away how sure the model was.

`predict` collapses a probability into 0 or 1, and 0 or 1 was all that
reached the artifact: the 2026-08-14 batch held exactly two distinct values
across 4,844 out-of-sample rows. A sweep of decision thresholds over that file
returned the identical count at every cut-off from 0.5 to 0.9, because there
was nothing between them to move.

What it cost is measurable on that batch. On target_hourly_breakout_1h the
winner fired 22 times in NORMAL and was right 91% of the time, and fired 138
times in TRENDING_UP and was right 48% -- at balanced accuracies of 0.841 and
0.837. The model ranks equally well in both regimes; only the cut-off is
wrong for one of them, and no cut-off can be chosen without the probability.

Every wrapper under src/models already implements `predict_proba`, and
SVMModel passes `probability=True` to SVC specifically so that it can. Nothing
in the training path had ever asked.

Two places had to change, which is the point of the last test here: the
trainer produces the column, and the writer enumerates its columns by name --
so producing it was not by itself enough to persist it. The same pairing hid
the keyword counts for four rounds of fixes.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage
from src.training.base_trainer import BaseTrainer


class _Probabilistic:
    """A binary classifier that is sure about some rows and not others."""

    classes_ = np.array([0, 1])

    def __init__(self, positive: list[float]):
        self._positive = np.asarray(positive, dtype=float)

    def predict(self, X):
        return (self._positive > 0.5).astype(float)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._positive, self._positive])


class _Blunt:
    """A model with no probabilities to give."""

    def predict(self, X):
        return np.zeros(3)


@pytest.fixture
def holdout():
    index = pd.date_range("2026-07-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=index)


def test_the_confidence_behind_each_call_survives(holdout):
    positive = [0.97, 0.51, 0.04, 0.62]
    model = _Probabilistic(positive)

    proba = BaseTrainer._holdout_probabilities(model, holdout, is_classif=True)
    series = BaseTrainer._holdout_prediction_series(
        holdout, np.array([1.0, 0.0, 0.0, 1.0]), model.predict(holdout), proba
    )

    stored = [r["probability"] for r in series]
    assert stored == pytest.approx(positive), (
        "a 0.97 and a 0.51 are both 'yes' and must not arrive as the same number"
    )
    assert len({round(p, 3) for p in stored}) == 4, (
        "the batch that surfaced this held two distinct values across 4,844 rows"
    )


def test_a_barely_positive_call_is_distinguishable_from_a_certain_one(holdout):
    """The whole point: a threshold must be able to separate them."""
    model = _Probabilistic([0.97, 0.51, 0.04, 0.62])
    proba = BaseTrainer._holdout_probabilities(model, holdout, is_classif=True)
    series = BaseTrainer._holdout_prediction_series(
        holdout, np.array([1.0, 0.0, 0.0, 1.0]), model.predict(holdout), proba
    )

    frame = pd.DataFrame(series)
    said_yes = frame[frame["prediction"] == 1.0]
    assert len(said_yes) == 3
    assert len(said_yes[said_yes["probability"] > 0.9]) == 1, (
        "raising the cut-off must change how often the model fires"
    )


def test_a_model_without_probabilities_reports_none_not_certainty(holdout):
    """An absent confidence must stay absent rather than be invented as 1.0."""
    series = BaseTrainer._holdout_prediction_series(
        holdout.iloc[:3], np.array([0.0, 0.0, 0.0]), np.zeros(3),
        BaseTrainer._holdout_probabilities(_Blunt(), holdout.iloc[:3], is_classif=True),
    )

    assert all(r["probability"] is None for r in series)


def test_a_regression_winner_is_not_asked_for_probabilities(holdout):
    model = _Probabilistic([0.9, 0.9, 0.9, 0.9])

    assert BaseTrainer._holdout_probabilities(model, holdout, is_classif=False) is None


def test_the_positive_class_is_found_where_sklearn_put_it(holdout):
    """Column 1 is the positive class only because `classes_` says so."""
    model = _Probabilistic([0.8, 0.2, 0.7, 0.1])
    model.classes_ = np.array([0, 1])

    proba = BaseTrainer._holdout_probabilities(model, holdout, is_classif=True)

    assert proba == pytest.approx([0.8, 0.2, 0.7, 0.1])


def test_the_writer_carries_the_column_to_the_artifact(tmp_path, monkeypatch):
    """Producing the column is not enough -- the writer names its columns.

    This is the pairing that hid the keyword counts: the enricher computed
    them and the merge listed the columns it kept, so the fix upstream showed
    no effect downstream for four rounds.
    """
    monkeypatch.chdir(tmp_path)
    champions = {
        "AAPL_1h_target_hourly_breakout_1h_NORMAL": {
            "ticker": "AAPL", "timeframe": "1h",
            "target_name": "target_hourly_breakout_1h", "model_type": "catboost",
            "holdout_predictions": [
                {"datetime": "2026-07-01T00:00:00+00:00", "prediction": 1.0,
                 "actual": 1.0, "probability": 0.93},
                {"datetime": "2026-07-01T01:00:00+00:00", "prediction": 1.0,
                 "actual": 0.0, "probability": 0.52},
            ],
        }
    }

    path = ModelingStage._write_holdout_predictions(champions)

    assert path is not None
    frame = pd.read_parquet(path)
    assert "probability" in frame.columns, (
        "the artifact is what every downstream consumer reads"
    )
    assert frame["probability"].tolist() == pytest.approx([0.93, 0.52])
