"""The block imputer must behave identically on a frame and on an array.

`_BlockImputer` replaced `SimpleImputer` inside training on 2026-08-31 to stop
a MemoryError (REGISTER #158). Its `transform` indexed by column NAME, which
is right for the training caller and an IndexError for Stage 5, which hands it
the numpy array it built by reindexing to the fit-time columns.

Nothing caught that for a day, because Stage 5 had never run in the project's
history -- the prediction path had no caller, so it had no test either. This
file tests both callers against each other: identical statistics, identical
output, whichever shape goes in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.adapters.data_preparation import _BlockImputer


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 7))
    data[::3, 2] = np.nan
    data[1::5, 5] = np.nan
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(7)])


def test_array_and_frame_transform_to_the_same_matrix(frame):
    imputer = _BlockImputer(block=3)
    imputer.fit_transform(frame)

    from_frame = imputer.transform(frame)
    from_array = imputer.transform(frame.to_numpy(dtype=float))

    np.testing.assert_allclose(from_frame, from_array)


def test_transform_of_an_array_is_what_fit_transform_produced(frame):
    imputer = _BlockImputer(block=3)
    fitted = imputer.fit_transform(frame)
    np.testing.assert_allclose(imputer.transform(frame.to_numpy(dtype=float)), fitted)


def test_blocking_does_not_change_the_medians(frame):
    """Byte-identical statistics is the whole justification for blocking."""
    from sklearn.impute import SimpleImputer

    blocked = _BlockImputer(block=2)
    blocked.fit_transform(frame)
    whole = SimpleImputer(strategy="median").fit(frame.to_numpy(dtype=float))

    np.testing.assert_allclose(blocked.statistics_, whole.statistics_)


def test_a_wrong_width_array_is_refused_rather_than_silently_misaligned(frame):
    imputer = _BlockImputer(block=3)
    imputer.fit_transform(frame)
    with pytest.raises(ValueError, match="fitted on 7 columns"):
        imputer.transform(frame.to_numpy(dtype=float)[:, :5])


def test_an_imputer_pickled_before_this_change_still_transforms(frame):
    """Champions on disk were pickled by the previous version of this class.

    The first attempt at the array path stored a new `slices` attribute in
    `__init__`. Unpickling restores `__dict__` and never calls `__init__`, so
    every champion already saved came back without it and Stage 5 refused all
    seven contexts with `'_BlockImputer' object has no attribute 'slices'`.
    This reproduces that object exactly: fitted, then stripped of anything the
    old version did not have.
    """
    imputer = _BlockImputer(block=3)
    expected = imputer.fit_transform(frame)

    legacy = _BlockImputer.__new__(_BlockImputer)
    legacy.__dict__ = {"block": imputer.block, "parts": imputer.parts}

    np.testing.assert_allclose(
        legacy.transform(frame.to_numpy(dtype=float)), expected
    )
    np.testing.assert_allclose(legacy.transform(frame), expected)
