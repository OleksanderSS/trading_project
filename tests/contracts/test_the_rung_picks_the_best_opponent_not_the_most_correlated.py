"""The strongest single-column opponent is not the most correlated column.

REGISTER #188, filed 31.08, marked закрито, and the fix never made -- caught
on 04.09 by `stale_state_scan.py` rule G, which reads a closed row that ends
"**Виправлення:** щабель має брати 5-10 найкорельованіших кандидатів".

Rung 5 of the opponent ladder asks whether ONE column and a straight line
already do what the model does. It ranked candidates by |Pearson| with the
target on train and fitted a line to the winner. Those are different
questions, and the project has a measured case of the gap: on
`target_hourly_breakout_1h` the rung chose `CCI_15m` (0.7915) while #171 had
already measured distance to the upper Bollinger band at AUC 0.9666 on the
same target, earning the same money as the model at matched selectivity. A
better opponent existed, was documented, and the rung never tried it -- because
it correlated less.

Why correlation is the wrong ranker HERE specifically: for a classification
target the prediction is `column >= cut`, so the score depends only on the
column's ORDERING. Pearson is magnitude-sensitive. A column whose ordering is
perfect near the decision boundary and noisy far from it scores worse on
Pearson and better as an opponent, which is exactly the Bollinger shape --
the band matters when price is near it.

DEVIATION FROM THE RECORDED FIX, stated because it is a deviation. #188 said
to score the candidates "керівною метрикою **на відкладеному наборі**".
Choosing one of ten on the holdout makes the opponent's reported score the
maximum of ten draws on the very data it is reported against -- an opponent
biased upward by its own selection, which can block a good model for a reason
that has nothing to do with the model. The candidates are ranked on TRAIN
instead. Each is one column and a straight line, two parameters, so there is
no meaningful overfitting to rank away, and the holdout stays what it exists
for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.model.ml_evaluator import MLEvaluator
from src.training.base_trainer import BaseTrainer


def _two_columns(n: int = 6000, seed: int = 5):
    """`most_correlated` wins on Pearson; `best_opponent` wins as an opponent.

    Measured on this construction: Pearson 0.768 against 0.705, thresholded
    accuracy 0.907 against 1.000. The old rule takes the first, the gate
    needs the second.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    target = (latent > 0).astype(int)

    most_correlated = latent + rng.normal(scale=0.30, size=n)
    # Ordering preserved exactly -- the noise only stretches magnitudes, and
    # it stretches them most where the answer is least in doubt.
    best_opponent = latent * (1.0 + 3.0 * np.abs(rng.normal(size=n)))

    index = pd.date_range("2010-01-04", periods=n, freq="D", tz="UTC")
    index.name = "model_datetime"
    frame = pd.DataFrame(
        {"most_correlated": most_correlated, "best_opponent": best_opponent},
        index=index,
    )
    return frame, target


def _score(frame, target, **kwargs):
    return BaseTrainer._score_single_feature_baseline(
        {"X_train": frame, "y_train": target,
         "X_holdout": frame, "y_holdout": target},
        True, "classification", "F1", MLEvaluator(), **kwargs,
    )


def test_the_construction_really_does_disagree():
    """If the two rankers agreed here, the test below would pass for the wrong
    reason and keep passing after a regression."""
    frame, target = _two_columns()
    pearson = frame.corrwith(pd.Series(target, index=frame.index)).abs()
    assert pearson["most_correlated"] > pearson["best_opponent"]

    def thresholded_accuracy(column):
        cut = np.quantile(column, 1.0 - target.mean())
        return float(((column >= cut).astype(int) == target).mean())

    assert (thresholded_accuracy(frame["best_opponent"].to_numpy())
            > thresholded_accuracy(frame["most_correlated"].to_numpy()))


def test_the_rung_chooses_the_better_opponent():
    frame, target = _two_columns()
    out = _score(frame, target)

    assert out["single_feature_status"] == "measured"
    assert out["single_feature_name"] == "best_opponent", (
        "rung 5 still ranks by |Pearson|, so it fields a weaker opponent than "
        "one it could have measured -- the #171/#188 shape"
    )


def test_the_number_of_opponents_tried_is_reported():
    """The project counts attempts everywhere else. A rung that quietly
    searches ten columns and reports one is the same omission in a smaller
    place."""
    frame, target = _two_columns()
    out = _score(frame, target)
    assert out["single_feature_candidates"] == 2, (
        "the count of candidates actually tried is missing or wrong"
    )
    assert "single_feature_train_score" in out, (
        "the score the choice was made on is not reported, so the selection "
        "cannot be checked"
    )


def test_the_candidate_count_is_a_named_number():
    assert isinstance(BaseTrainer.SINGLE_FEATURE_CANDIDATES, int)
    assert 5 <= BaseTrainer.SINGLE_FEATURE_CANDIDATES <= 20, (
        "the prefilter is either too narrow to contain the better opponent or "
        "wide enough to be a search in its own right"
    )


def test_a_single_column_frame_still_works():
    """The commonest case must not be broken by the loop."""
    frame, target = _two_columns()
    out = _score(frame[["most_correlated"]], target)
    assert out["single_feature_status"] == "measured"
    assert out["single_feature_name"] == "most_correlated"
    assert out["single_feature_candidates"] == 1


def test_an_unusable_frame_still_says_so_rather_than_scoring_zero():
    """"Could not measure" must never read as "the opponent scored nothing" --
    REGISTER #202."""
    index = pd.date_range("2010-01-04", periods=200, freq="D", tz="UTC")
    constant = pd.DataFrame({"flat": np.ones(200)}, index=index)
    out = _score(constant, np.zeros(200, dtype=int))
    assert out["single_feature_score"] is None
    assert out["single_feature_status"] in {"no_usable_feature", "shape_mismatch"}
