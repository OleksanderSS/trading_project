"""A row already in the table has to be recognised as already in the table.

`upsert` filtered out rows whose unique key already existed by comparing
`str(value)` on both sides. For a date column that is not one key but three:
DuckDB returns a Timestamp printing as "2026-06-01 00:00:00", a collector may
hand over a `datetime.date` printing as "2026-06-01", and numpy's datetime64
prints "2026-06-01T00:00:00". Same day, three strings, none equal.

So the row survived the filter and hit the unique index on insert:

    Constraint Error: Duplicate key "date: 2026-06-01 00:00:00"

`vix_data` failed outright on every run this way, and the collector was
reported dead while its data was already stored.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.management.data_manager import DataManager

_key = DataManager._comparable_key


def test_the_same_day_in_four_shapes_is_one_key():
    values = pd.Series([
        pd.Timestamp("2026-06-01"),
        dt.date(2026, 6, 1),
        "2026-06-01",
        np.datetime64("2026-06-01T00:00:00"),
    ], dtype=object)
    assert _key(values).nunique() == 1


def test_str_alone_would_have_seen_four_different_days():
    """Pins why the fix is needed, not just that it works."""
    values = [
        pd.Timestamp("2026-06-01"),
        dt.date(2026, 6, 1),
        "2026-06-01",
        np.datetime64("2026-06-01T00:00:00"),
    ]
    assert len({str(v) for v in values}) > 1


def test_a_stored_timestamp_matches_an_incoming_date():
    """The exact shape of the vix_data failure."""
    stored = pd.Series(pd.to_datetime(["2026-06-01", "2026-06-02"]))
    incoming = pd.Series([dt.date(2026, 6, 1), dt.date(2026, 6, 3)], dtype=object)

    already_there = set(_key(stored).tolist())
    is_new = ~_key(incoming).isin(already_there)

    assert is_new.tolist() == [False, True]


def test_different_days_stay_different():
    values = pd.Series(pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]))
    assert _key(values).nunique() == 3


def test_timezones_do_not_split_one_instant_in_two():
    aware = pd.Series(pd.to_datetime(["2026-06-01T12:00:00+02:00"]))
    naive = pd.Series(pd.to_datetime(["2026-06-01T10:00:00"]))
    assert _key(aware).iloc[0] == _key(naive).iloc[0]


def test_a_text_key_is_still_compared_as_text():
    hashes = pd.Series(["a1b2c3", "d4e5f6", "a1b2c3"])
    keys = _key(hashes)
    assert keys.tolist() == ["a1b2c3", "d4e5f6", "a1b2c3"]


def test_a_column_that_is_only_partly_dates_is_not_half_turned_into_nat():
    mixed = pd.Series(["2026-06-01", "not-a-date", "2026-06-02"])
    keys = _key(mixed)
    assert keys.tolist() == ["2026-06-01", "not-a-date", "2026-06-02"]
    assert keys.nunique() == 3


def test_the_index_survives_so_the_caller_can_align_it():
    """drop_duplicates upstream leaves a non-contiguous index."""
    values = pd.Series(pd.to_datetime(["2026-06-01", "2026-06-03"]), index=[0, 7])
    assert _key(values).index.tolist() == [0, 7]


@pytest.mark.parametrize("empty", [pd.Series([], dtype="datetime64[ns]"),
                                   pd.Series([], dtype=object)])
def test_an_empty_column_does_not_raise(empty):
    assert len(_key(empty)) == 0
