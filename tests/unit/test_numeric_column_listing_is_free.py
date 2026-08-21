"""Listing numeric column NAMES must not allocate the numeric data.

`_initial_feature_columns` asked which columns are numeric via
`frame.select_dtypes(include="number").columns`. select_dtypes consolidates and
COPIES the matching columns into one block, then hands back their names — so on
the 2026-08-21 batch it allocated

    2,192 columns x 258,397 rows x float64 = 4.22 GiB

to answer a question about metadata. It failed the rebuild with MemoryError
twice, once at 256,208 rows and once at 258,397. Deepening the daily history
from two years to thirty is what pushed it over, but the allocation was always
pointless: casting to float32 would have halved a number that should be zero.

The tests pin equivalence with the old behaviour — including the cases where a
naive rewrite differs — and pin that the listing does not touch the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.stages.feature_engineering.orchestrator import (  # noqa: E402
    FeatureEngineeringStage,
)

# an instance method, but it reads nothing from self
def listing(frame):
    return FeatureEngineeringStage._initial_feature_columns(None, frame)


def mixed() -> pd.DataFrame:
    return pd.DataFrame({
        'close': [1.5, 2.5],
        'volume': [10, 20],
        'small_uint': np.array([1, 2], dtype='uint8'),
        'name': ['x', 'y'],
        'flag': [True, False],
        'when': pd.to_datetime(['2020-01-01', '2020-01-02']),
        'datetime': pd.to_datetime(['2020-01-01', '2020-01-02']),
        'ticker': ['A', 'B'],
        'interval': ['1d', '1d'],
    })


class TestItMatchesTheOldBehaviour:
    def test_the_same_columns_as_select_dtypes(self):
        d = mixed()
        old = [c for c in d.select_dtypes(include='number').columns
               if c not in {'datetime', 'date', 'timestamp', 'ticker',
                            'interval', 'timeframe'}]
        assert listing(d) == old

    def test_every_numeric_width_is_included(self):
        d = pd.DataFrame({'i8': np.array([1], 'int8'), 'u16': np.array([1], 'uint16'),
                          'f32': np.array([1.0], 'float32'), 'f64': np.array([1.0], 'float64')})
        assert set(listing(d)) == {'i8', 'u16', 'f32', 'f64'}

    def test_booleans_are_excluded_as_select_dtypes_excludes_them(self):
        # A naive `dtype != object` rewrite would let bool through and quietly
        # widen the feature pool.
        assert 'flag' not in listing(mixed())

    def test_datetimes_are_excluded(self):
        assert 'when' not in listing(mixed())

    def test_metadata_columns_are_excluded(self):
        out = listing(mixed())
        for c in ('datetime', 'ticker', 'interval'):
            assert c not in out


class TestItDoesNotTouchTheData:
    def test_no_block_consolidation_happens(self):
        """The point: metadata only, whatever the frame's size."""
        d = mixed()
        blocks_before = len(d._mgr.blocks)
        listing(d)
        assert len(d._mgr.blocks) == blocks_before, (
            'the listing consolidated the frame, which is the allocation this '
            'exists to avoid')

    def test_a_frame_too_wide_to_materialise_is_still_listable(self):
        # 3,000 columns of float64 over 200k rows would be ~4.8 GiB if
        # consolidated. Built lazily here: only the dtypes are ever read.
        n = 3000
        d = pd.DataFrame({f'f{i}': pd.array([1.0], dtype='float64') for i in range(n)})
        assert len(listing(d)) == n
