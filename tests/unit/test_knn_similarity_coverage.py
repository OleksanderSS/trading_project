"""A context we barely know must not match confidently.

KnnSimilarityFinder fills missing features with historical medians. That is
the right direction (history informs the present, not the reverse), but it
turns a mostly-unknown row into an average-looking one, which then sits
close to every other average row.

The only guard was "at least one feature present". Measured before the fix:
a context with 1 real value out of 20 returned three neighbours with
similarity scores 0.173, 0.158, 0.150 and no warning. Those are the rows
nearest the median, not the rows nearest this context -- and the caller has
no way to see the difference.

min_feature_coverage (0.5 by default, configurable) now sets the bar.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.core.exceptions import DataProcessingError

COLUMNS = [f"f{i}" for i in range(20)]


def _history(rows=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(rows, len(COLUMNS))),
        columns=COLUMNS,
        index=[f"h{i}" for i in range(rows)],
    )


def _target(known):
    row = pd.DataFrame([[np.nan] * len(COLUMNS)], columns=COLUMNS, index=["t0"])
    for i in range(known):
        row.iloc[0, i] = 0.5
    return row


def test_a_well_covered_context_still_matches():
    result = KnnSimilarityFinder(config={"n_neighbors": 3}).analyze(
        {"historical_features": _history(), "target_features": _target(20)}
    )

    assert len(result["similarities"]["t0"]) == 3


def test_half_coverage_is_accepted():
    result = KnnSimilarityFinder(config={"n_neighbors": 3}).analyze(
        {"historical_features": _history(), "target_features": _target(10)}
    )

    assert result["similarities"]["t0"]


def test_a_barely_known_context_is_refused(caplog):
    """The regression: 1 of 20 used to return three confident neighbours."""
    finder = KnnSimilarityFinder(config={"n_neighbors": 3})

    with caplog.at_level(logging.WARNING), pytest.raises(DataProcessingError):
        finder.analyze(
            {"historical_features": _history(), "target_features": _target(1)}
        )


def test_the_refusal_says_how_many_features_were_missing(caplog):
    finder = KnnSimilarityFinder(config={"n_neighbors": 3})

    with caplog.at_level(logging.WARNING):
        with pytest.raises(DataProcessingError):
            finder.analyze(
                {"historical_features": _history(), "target_features": _target(2)}
            )

    assert any("fewer than" in r.getMessage() for r in caplog.records)


def test_the_threshold_is_configurable():
    """A caller that knowingly works with sparse contexts can lower it."""
    finder = KnnSimilarityFinder(config={"n_neighbors": 3, "min_feature_coverage": 0.05})

    result = finder.analyze(
        {"historical_features": _history(), "target_features": _target(1)}
    )

    assert result["similarities"]["t0"]


def test_find_similar_situations_applies_the_same_bar():
    finder = KnnSimilarityFinder(config={"n_neighbors": 3})
    finder.fit(_history())

    sparse = pd.Series({column: np.nan for column in COLUMNS})
    sparse["f0"] = 0.5

    with pytest.raises(DataProcessingError, match="at least"):
        finder.find_similar_situations(sparse)


def test_find_similar_situations_works_with_a_full_context():
    finder = KnnSimilarityFinder(config={"n_neighbors": 3})
    finder.fit(_history())

    positions, distances = finder.find_similar_situations(
        pd.Series({column: 0.5 for column in COLUMNS})
    )

    assert len(positions) == 3
    assert len(distances) == 3


def test_the_default_bar_is_recorded():
    assert KnnSimilarityFinder().min_feature_coverage == 0.5


def test_fully_populated_vectors_are_unaffected():
    """KnnContextFinder feeds fingerprint vectors, which have no gaps at all;
    this must keep working exactly as before."""
    finder = KnnSimilarityFinder(config={"n_neighbors": 1})
    columns = ["fp_0", "fp_1", "fp_2"]
    history = pd.DataFrame(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        columns=columns, index=["a", "b", "c"],
    )
    target = pd.DataFrame([[1.0, 1.0, 1.0]], columns=columns, index=["target"])

    result = finder.analyze(
        {"historical_features": history, "target_features": target}
    )

    assert result["similarities"]["target"][0]["id"] == "c"
