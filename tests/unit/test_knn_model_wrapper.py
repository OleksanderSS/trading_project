"""Neighbour positions are not index labels.

KnnModelWrapper.fit() drops non-numeric and null rows before fitting, so
kneighbors returns positions into the CLEANED array. A caller that maps them
back through the frame it passed in gets a different row for every neighbour
after the first dropped one -- silently, with ids that look plausible.

Not a live defect today, and that was checked rather than assumed:
KnnSimilarityFinder._prepare_feature_matrices imputes gaps from historical
medians before calling fit, so nothing is dropped and the two indices agree.
The trap is in the wrapper's contract, not in its current use --
fitted_data_index was stored for exactly this and nothing consumed it.

fit() now says when it drops rows, and neighbor_labels() does the mapping.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.analytics.utils.knn_model_wrapper import KnnModelWrapper


def _frame(n=6, gap_at=None, start=0.0):
    frame = pd.DataFrame(
        {"f0": np.arange(n, dtype=float) + start,
         "f1": np.arange(n, dtype=float) + start},
        index=[f"row{i}" for i in range(n)],
    )
    if gap_at is not None:
        frame.iloc[gap_at, 0] = np.nan
    return frame


def test_neighbour_labels_come_from_the_fitted_rows():
    wrapper = KnnModelWrapper(n_neighbors=1)
    wrapper.fit(_frame())

    _, indices = wrapper.find_neighbors(pd.DataFrame(
        {"f0": [5.0], "f1": [5.0]}, index=["t"]
    ))

    assert wrapper.neighbor_labels(indices) == [["row5"]]


def test_positions_and_labels_diverge_once_a_row_is_dropped():
    """The whole point: after a gap, position i is no longer row i."""
    wrapper = KnnModelWrapper(n_neighbors=1)
    frame = _frame(gap_at=2)
    wrapper.fit(frame)

    _, indices = wrapper.find_neighbors(pd.DataFrame(
        {"f0": [5.0], "f1": [5.0]}, index=["t"]
    ))
    position = int(np.asarray(indices)[0][0])

    assert wrapper.neighbor_labels(indices) == [["row5"]]
    assert frame.index[position] == "row4", (
        "indexing the input frame by position gives the wrong row -- which is "
        "exactly what a caller doing historical_index[idx] would get"
    )


def test_dropping_rows_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        KnnModelWrapper(n_neighbors=1).fit(_frame(gap_at=2))

    assert any("dropped 1 of 6" in r.getMessage() for r in caplog.records)


def test_a_clean_frame_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        KnnModelWrapper(n_neighbors=1).fit(_frame())

    assert not [r for r in caplog.records if "dropped" in r.getMessage()]


def test_labels_before_fitting_are_refused():
    with pytest.raises(RuntimeError):
        KnnModelWrapper().neighbor_labels([[0]])


def test_finding_neighbours_before_fitting_is_refused():
    with pytest.raises(RuntimeError):
        KnnModelWrapper().find_neighbors(_frame())


def test_an_empty_fit_frame_is_refused():
    with pytest.raises(ValueError):
        KnnModelWrapper().fit(pd.DataFrame())


def test_a_frame_with_no_numeric_columns_is_refused():
    with pytest.raises(ValueError):
        KnnModelWrapper().fit(pd.DataFrame({"tag": ["a", "b"]}))


def test_a_non_positive_neighbour_count_is_refused():
    with pytest.raises(ValueError):
        KnnModelWrapper(n_neighbors=0)


def test_an_empty_target_returns_no_neighbours():
    wrapper = KnnModelWrapper(n_neighbors=1)
    wrapper.fit(_frame())

    assert wrapper.find_neighbors(pd.DataFrame({"tag": ["x"]})) == ([], [])


def test_the_live_caller_still_agrees_with_the_labels():
    """KnnSimilarityFinder imputes before fitting, so its own mapping is
    correct; this pins that the two agree while that stays true."""
    from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

    historical = _frame(n=5)
    target = pd.DataFrame({"f0": [4.0], "f1": [4.0]}, index=["t0"])

    result = KnnSimilarityFinder(config={"n_neighbors": 1}).analyze(
        {"historical_features": historical, "target_features": target}
    )

    assert result["similarities"]["t0"][0]["id"] == "row4"
