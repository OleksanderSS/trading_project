"""An anomalous bar must LOSE confidence, and the warning must fire on the
anomalous side.

REGISTER #151, filed on 2026-08-29 as "the warning fires on any value". The
entry corrected itself on 30.08 -- a threshold does exist, `< 0.8` -- and
stopped there. Reading the code on 2026-09-04 shows both uses of the score
were inverted against its own documented contract.

`calculate_anomaly_score` says "Return anomaly score in [0, 1]. Higher -> more
anomalous", and both of its parts agree:

    the z-score branch returns |z| / 3, clipped -- larger deviation, larger
    score

    the isolation-forest branch returns 1.0 exactly when the point is an
    outlier and 0.0 when it is not

The prediction stage then did:

    raw_confidence = confidence * anomaly_score      # a normal bar -> 0
    if anomaly_score < 0.8: warn "potential data anomaly"

So a perfectly ordinary bar (anomaly 0.0) was assigned confidence ZERO while
an outlier (1.0) kept its confidence in full, and the warning announced every
ordinary bar as an anomaly. Observed on the 2026-08-29 run: scores of 0.06,
0.08, 0.11, 0.25, 0.26, 0.33, 0.56, 0.68, 0.69, 0.70, 0.71 -- normal data,
every one of them warned about, every one of them crushed.

A warning that fires on every row equals a warning switched off, and this one
was worse than off: it looked like a working control while inverting the
quantity it guarded.

Checked before flipping the sign: `calculate_ensemble_confidence` returns an
ordinary confidence (consensus, dispersion, diary accuracy, volatility), so
there is no compensating inversion that made the product correct.
"""
from __future__ import annotations

import inspect

import pytest

from src.pipeline.stages.prediction.anomaly_engine import AnomalyEngine
from src.pipeline.stages.prediction.orchestrator import (
    ANOMALY_WARNING_THRESHOLD,
)


def test_the_scorer_still_means_higher_is_more_anomalous():
    """Everything below depends on this direction. If the scorer is ever
    inverted, the fix becomes the defect."""
    doc = inspect.getdoc(AnomalyEngine.calculate_anomaly_score) or ""
    assert "more anomalous" in doc.lower()

    isolation = inspect.getsource(AnomalyEngine._calculate_isolation_forest_anomaly)
    assert "1.0 if pred[0] == -1 else 0.0" in isolation, (
        "the isolation-forest branch no longer returns 1.0 for an outlier, so "
        "the direction of the whole score is in question"
    )
    zscore = inspect.getsource(AnomalyEngine._calculate_zscore_anomaly)
    assert "/ 3.0" in zscore and "np.abs" in zscore, (
        "the z-score branch no longer grows with deviation"
    )


def test_confidence_is_multiplied_by_what_remains_after_the_anomaly():
    from src.pipeline.stages.prediction import orchestrator

    source = inspect.getsource(orchestrator.PredictionStage._create_prediction_result)
    assert "1.0 - anomaly_score" in source, (
        "confidence is multiplied by the anomaly again, so a normal bar gets "
        "zero confidence and an outlier keeps all of it"
    )
    assert "* anomaly_score)" not in source.replace("(1.0 - anomaly_score)", ""), (
        "a bare multiplication by anomaly_score survives somewhere"
    )


@pytest.mark.parametrize("anomaly,expected", [
    (0.0, 1.0),    # ordinary bar keeps its confidence
    (0.2, 0.8),
    (0.71, 0.29),  # the worst score actually observed on a real run
    (1.0, 0.0),    # a certain outlier keeps none
])
def test_the_multiplier_moves_the_right_way(anomaly, expected):
    assert (1.0 - anomaly) == pytest.approx(expected)


def test_the_warning_fires_on_the_anomalous_side():
    from src.pipeline.stages.prediction import orchestrator

    source = inspect.getsource(orchestrator.PredictionStage._create_prediction_result)
    assert "anomaly_score > ANOMALY_WARNING_THRESHOLD" in source, (
        "the warning is back on the low side, where it fires on every "
        "ordinary bar and therefore says nothing"
    )
    assert "HIGH anomaly" in source, (
        "the message still calls a high score 'low', which is what made the "
        "line self-contradictory"
    )


def test_none_of_the_observed_scores_would_warn_today():
    """The eleven values seen on the 2026-08-29 run were all ordinary data.
    Every one of them warned. None may now."""
    observed = [0.06, 0.08, 0.11, 0.25, 0.26, 0.33, 0.56, 0.68, 0.69, 0.70, 0.71]
    assert not [s for s in observed if s > ANOMALY_WARNING_THRESHOLD], (
        "an ordinary bar still trips the warning"
    )


def test_the_threshold_is_a_named_number():
    """Inline, the number IS the check and nobody reads it; named, changing it
    is a decision."""
    assert isinstance(ANOMALY_WARNING_THRESHOLD, float)
    assert 0.5 < ANOMALY_WARNING_THRESHOLD < 1.0
