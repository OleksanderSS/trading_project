"""The most expensive step in stage 3, and two reasons it was expensive.

RedundancyDetector takes 88 to 158 minutes per run. Two things in it were
measured rather than guessed:

  * the low-variance filter compared `var() < threshold`, and an all-NaN
    column has `var() == NaN`, so every comparison was False and the column
    survived. A CONSTANT column was dropped and an EMPTY one was kept -- the
    emptier the column, the safer it was from the filter meant to catch it.
    Those survivors went on into correlation clustering and VIF.

  * removal dropped one column at a time, copying the frame on each. Measured:
    5,000 x 1,200 dropping 800 took 16.44s against 0.014s for one drop.

  * VIF fitted one LinearRegression per column against a copy of the frame
    minus that column: O(n * p^3). It is the diagonal of the inverted
    correlation matrix, which is an identity, not an approximation. Agreement
    checked at 3.17e-12 maximum relative difference before the swap.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.validation.redundancy_detector import RedundancyDetector


@pytest.fixture
def detector():
    return RedundancyDetector()


def test_a_column_with_no_values_is_removed(detector):
    """It was the one shape the filter let through."""
    frame = pd.DataFrame({
        "live": np.linspace(0, 10, 200),
        "empty": [np.nan] * 200,
        "constant": [7.0] * 200,
    })
    result = detector._remove_low_variance_features(frame)

    assert set(result["removed_features"]) == {"empty", "constant"}
    assert list(result["remaining_features"].columns) == ["live"]


def test_a_column_with_one_value_among_nans_is_removed(detector):
    """var() of a single observation is NaN too."""
    column = [np.nan] * 199 + [3.0]
    frame = pd.DataFrame({"live": np.linspace(0, 10, 200), "almost_empty": column})
    result = detector._remove_low_variance_features(frame)
    assert "almost_empty" in result["removed_features"]


def test_a_varying_column_survives(detector):
    frame = pd.DataFrame({"a": np.linspace(0, 10, 200), "b": np.linspace(5, -5, 200)})
    result = detector._remove_low_variance_features(frame)
    assert result["removed_features"] == []
    assert list(result["remaining_features"].columns) == ["a", "b"]


def test_the_caller_s_frame_is_not_mutated(detector):
    frame = pd.DataFrame({"live": np.linspace(0, 1, 200), "empty": [np.nan] * 200})
    detector._remove_low_variance_features(frame)
    assert list(frame.columns) == ["live", "empty"]


def test_vif_matches_the_regression_definition(detector):
    """1/(1 - R^2) of each column on the others, computed the literal way."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    base = rng.standard_normal((800, 4))
    frame = pd.DataFrame(
        np.hstack([base, base @ rng.standard_normal((4, 8)) + 0.4 * rng.standard_normal((800, 8))]),
        columns=[f"f{i}" for i in range(12)],
    )
    target = pd.Series(rng.standard_normal(800))

    result = detector._calculate_vif_analysis(frame, target)
    scores = result["vif_scores"]
    assert len(scores) == 12

    for name in frame.columns:
        others = frame.drop(columns=[name])
        r2 = LinearRegression().fit(others, frame[name]).score(others, frame[name])
        expected = float("inf") if r2 >= 0.999 else 1 / (1 - r2)
        if np.isfinite(expected) and np.isfinite(scores[name]):
            assert scores[name] == pytest.approx(expected, rel=1e-6)
        else:
            assert not np.isfinite(scores[name]) and not np.isfinite(expected)


def test_perfectly_collinear_columns_do_not_raise(detector):
    """A singular correlation matrix is a normal state here, not an error."""
    values = np.linspace(0, 1, 300)
    frame = pd.DataFrame({"a": values, "b": values * 2.0, "c": np.sin(values * 7)})
    result = detector._calculate_vif_analysis(frame, pd.Series(values))

    assert "error" not in result
    assert not np.isfinite(result["vif_scores"]["a"])
    assert "a" in result["high_vif_features"]


def test_an_independent_column_has_vif_near_one(detector):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.standard_normal((2000, 5)), columns=list("abcde"))
    result = detector._calculate_vif_analysis(frame, pd.Series(rng.standard_normal(2000)))
    for name in "abcde":
        assert result["vif_scores"][name] == pytest.approx(1.0, abs=0.05)
    assert result["high_vif_features"] == []
