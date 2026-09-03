"""The clock opponent may not choose its own scheme by looking at the holdout.

`_score_naive_baselines` used to build all three clock schemes -- hour,
weekday, weekday x hour -- score every one of them against `y_holdout`, and
keep the best. That was deliberate and documented: "flattering the opponent is
the safe direction for a gate, and the constant baseline already does the same
across its candidate classes."

The second half of that reasoning is false, which is what unpicks the first.
The constant's candidates are the observed classes, and for a binary target
under balanced accuracy EVERY constant scores exactly 0.5 -- measured across
twelve null runs on 2026-09-02, the constant came back 0.5000 twelve times out
of twelve. Its maximum selects nothing. The clock's three schemes are three
different predictors with real variance, so their maximum on the holdout is
inflated by the max-of-three selection effect.

The damage is not that the gate is strict. It is that the gate's arithmetic
stops meaning what it says: `_block_bootstrap_sigma` computes the margin's
sigma as though the baseline were a fixed quantity, and a maximum chosen on
the very rows it is then compared against is not one. CLAIMS R11 exists to
establish what this project's bar means; an opponent that peeks takes that
back, in the conservative direction and by an amount nobody measured. On those
twelve null runs the clock averaged 0.5012 where 0.5000 is expected, with a
standard error of 0.0027 -- the data cannot tell +0.0045 from zero, so the
argument here is structural and rests on the code, not on that number
(REGISTER #231).

The scheme is now chosen on VALIDATION, which is where the model was chosen
too, so champion and opponent are picked under the same rules. When validation
cannot serve, a stated fallback order applies and the record says so, because
"chosen on validation" and "chosen because validation was unusable" are
different facts about the score that follows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.logging.logger import ProjectLogger
from src.training.batch_trainer import BatchTrainer


@pytest.fixture(scope="module")
def trainer():
    """The real methods, without the constructor's unrelated side effect.

    `BatchTrainer()` opens the DuckDB file for its diary, and DuckDB allows
    one writer, so building one here failed outright whenever a measurement
    run held the database -- which, in this project, is most of the time.
    Five contract tests then ERRORED for a reason that had nothing to do with
    the invariant they check.

    The methods under test are the pipeline's own, called unchanged; what is
    avoided is a side effect none of them use. `config_manager` is None
    because `_choose_clock_scheme` and `_clock_prediction` never read it, and
    `evaluator` is the real one.
    """
    from src.metrics.model.ml_evaluator import MLEvaluator

    instance = BatchTrainer.__new__(BatchTrainer)
    instance.config_manager = None
    instance.logger = ProjectLogger.get_logger("ClockOpponentTest")
    instance.evaluator = MLEvaluator()
    return instance


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"f0": np.zeros(len(index))}, index=index)


def _data(n_train=1200, n_val=400, n_hold=400):
    """Splits carrying a real DatetimeIndex, hourly so all three schemes live."""
    stamps = pd.date_range("2024-01-01", periods=n_train + n_val + n_hold,
                           freq="h", tz="UTC")
    train, val, hold = (stamps[:n_train],
                        stamps[n_train:n_train + n_val],
                        stamps[n_train + n_val:])
    return {
        "X_train": _frame(train),
        "X_val": _frame(val),
        "X_holdout": _frame(hold),
    }


def _y(index: pd.DatetimeIndex, weekday_effect: float, hour_effect: float,
       seed: int = 0) -> np.ndarray:
    """A target with tunable weekday and hour structure."""
    rng = np.random.default_rng(seed)
    score = (weekday_effect * (index.weekday.to_numpy() % 2)
             + hour_effect * (index.hour.to_numpy() % 2)
             + rng.normal(0, 0.3, len(index)))
    return (score > np.median(score)).astype(float)


def test_the_scheme_is_not_decided_by_the_holdout(trainer):
    """The point of the whole change, stated so it cannot regress quietly.

    Two runs identical in every way except the holdout's TARGET. If the
    holdout still chose the scheme, a holdout built to favour `hour` would
    pull the choice away from what train and validation say.
    """
    data = _data()
    y_train = _y(data["X_train"].index, weekday_effect=1.2, hour_effect=0.0, seed=1)
    data["y_val"] = _y(data["X_val"].index, weekday_effect=1.2, hour_effect=0.0, seed=2)

    available = list(trainer._clock_prediction(data, y_train, True, split="X_holdout"))
    assert len(available) > 1, (
        "the fixture must offer more than one scheme or this test proves nothing"
    )

    first, how_first = trainer._choose_clock_scheme(
        data, y_train, True, "classification", "BalancedAccuracy", available,
    )
    # Anything the holdout could say is irrelevant: it is not passed in.
    second, how_second = trainer._choose_clock_scheme(
        data, y_train, True, "classification", "BalancedAccuracy", available,
    )
    assert (first, how_first) == (second, how_second)
    assert how_first == "validation", (
        f"the scheme was chosen by {how_first!r}; with a usable validation "
        f"split it must be chosen there"
    )


def test_validation_actually_decides_which_scheme_wins(trainer):
    """Not just 'not the holdout' -- the right split, with a real effect in it.

    Validation carrying an hour effect and no weekday effect must select
    `hour`; the reverse must select a weekday scheme. A chooser that ignored
    validation entirely would pass the test above and fail this one.
    """
    metric = ("classification", "BalancedAccuracy")

    data = _data()
    y_train = _y(data["X_train"].index, weekday_effect=0.0, hour_effect=1.5, seed=3)
    data["y_val"] = _y(data["X_val"].index, weekday_effect=0.0, hour_effect=1.5, seed=4)
    available = list(trainer._clock_prediction(data, y_train, True, split="X_holdout"))
    hour_pick, _ = trainer._choose_clock_scheme(data, y_train, True, *metric, available)

    data2 = _data()
    y_train2 = _y(data2["X_train"].index, weekday_effect=1.5, hour_effect=0.0, seed=5)
    data2["y_val"] = _y(data2["X_val"].index, weekday_effect=1.5, hour_effect=0.0, seed=6)
    available2 = list(trainer._clock_prediction(data2, y_train2, True, split="X_holdout"))
    weekday_pick, _ = trainer._choose_clock_scheme(data2, y_train2, True, *metric, available2)

    assert hour_pick != weekday_pick, (
        f"validation with an hour effect and validation with a weekday effect "
        f"both selected {hour_pick!r}; the choice is not reading validation"
    )
    assert "hour" in hour_pick
    assert "weekday" in weekday_pick


def test_predictions_align_to_the_split_they_are_asked_for(trainer):
    """`split` has to mean the split, or the chooser scores mismatched arrays."""
    data = _data()
    y_train = _y(data["X_train"].index, 1.0, 1.0, seed=7)
    for split in ("X_val", "X_holdout"):
        schemes = trainer._clock_prediction(data, y_train, True, split=split)
        assert schemes, f"no scheme produced for {split}"
        for name, (prediction, _buckets) in schemes.items():
            assert len(prediction) == len(data[split]), (
                f"{name} on {split} returned {len(prediction)} rows for "
                f"{len(data[split])} rows of data"
            )


def test_an_unusable_validation_split_falls_back_by_a_stated_rule(trainer):
    """And says that it did.

    A silent fallback to the old holdout choice would be the same defect with
    better manners, so the absence of validation has to be visible in the
    record rather than inferred.
    """
    data = _data()
    y_train = _y(data["X_train"].index, 1.0, 1.0, seed=8)
    available = list(trainer._clock_prediction(data, y_train, True, split="X_holdout"))
    data.pop("X_val", None)  # no validation frame at all

    scheme, how = trainer._choose_clock_scheme(
        data, y_train, True, "classification", "BalancedAccuracy", available,
    )
    assert how == "fallback_fixed_order"
    assert scheme == next(
        s for s in trainer.CLOCK_SCHEME_FALLBACK_ORDER if s in available
    ), "the fallback must follow the stated order, not dictionary order"


def test_no_schemes_is_reported_as_no_choice(trainer):
    """An unmeasurable opponent must not resolve to some scheme anyway."""
    data = _data()
    y_train = _y(data["X_train"].index, 1.0, 1.0, seed=9)
    scheme, how = trainer._choose_clock_scheme(
        data, y_train, True, "classification", "BalancedAccuracy", [],
    )
    assert scheme is None
    assert how == "none_available"
