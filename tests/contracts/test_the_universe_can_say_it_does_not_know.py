"""A point-in-time universe must refuse a date it cannot cover.

REGISTER #169, CLAIMS Р41 and Р42.

Measured on our own panel: of 110 names, ZERO stop trading before the panel
ends, across twenty-seven years. Nothing can die in that universe, so a
permanent long in the survivors pays by construction -- and Р41 measured the
size of it: a fitted cross-sectional book scored Sharpe 1.417, and 0.675 once
every persistent name bet was removed. More than half of the result was the
composition of the list.

`universe_membership` exists to remove that. The way such a fix fails is not by
crashing: it is by answering ANYWAY, returning today's survivors wearing a
point-in-time label, so the bias comes back through the code written to remove
it and nothing looks wrong.

THE SECOND EDGE, WHICH IS WHY THIS FILE EXISTS AT ALL. The first version
guarded only the early side -- a date before the earliest delisting on record.
Running it showed the other end was the one that mattered: built from Alpha
Vantage's single demo snapshot, the store's latest known delisting is
2014-07-09, so every date in our real test window (2015-2023) would have been
answered with today's survivors and nothing else. Coverage is a WINDOW, and a
window has two ends.

These tests use frames built in-process. The store on disk is a fetched
artefact whose contents change with the key available, and a contract that
depends on today's fetch is a contract that fails for a reason unrelated to
the behaviour it pins.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data.universe_membership import (
    COLUMNS, NoCoverage, coverage, died_between, load, save, universe_as_of,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NaT if column.endswith("_date") else ""
    for column in ("ipo_date", "delisting_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame[list(COLUMNS)]


def _store() -> pd.DataFrame:
    return _frame([
        {"ticker": "OLD", "ipo_date": "1990-01-02",
         "delisting_date": "2005-06-30", "status": "Delisted"},
        {"ticker": "DEAD", "ipo_date": "1998-03-02",
         "delisting_date": "2012-11-15", "status": "Delisted"},
        {"ticker": "ALIVE", "ipo_date": "1995-05-05", "status": "Active"},
        {"ticker": "LATE", "ipo_date": "2010-08-01", "status": "Active"},
    ])


def test_a_date_inside_the_covered_window_is_answered():
    names = universe_as_of("2008-01-02", _store())
    assert names == {"DEAD", "ALIVE"}, (
        "OLD died in 2005 and LATE had not listed; the answer must be the "
        "names actually trading that day"
    )


def test_a_name_is_held_on_its_last_day_and_not_after():
    """Needs its own store: in `_store()` the last known death IS 2012-11-15,
    so the day after falls outside coverage and is refused -- correctly. The
    first version of this test asked the module to answer a date it had just
    been taught to refuse, which is a badly built test rather than a defect.
    """
    later = pd.concat([_store(), _frame([
        {"ticker": "LAST", "ipo_date": "2001-01-02",
         "delisting_date": "2014-01-06", "status": "Delisted"},
    ])], ignore_index=True)
    assert "DEAD" in universe_as_of("2012-11-15", later)
    assert "DEAD" not in universe_as_of("2012-11-16", later)


def test_a_date_before_the_earliest_known_delisting_is_refused():
    """Answering would assert that nothing had died yet, which is false of any
    real market and is exactly the bias being removed."""
    with pytest.raises(NoCoverage) as raised:
        universe_as_of("2001-01-02", _store())
    assert "earliest" in str(raised.value)


def test_a_date_after_the_latest_known_delisting_is_refused():
    """The edge that actually bit. With one snapshot the store knows nothing
    that died afterwards, so this date would return today's survivors."""
    with pytest.raises(NoCoverage) as raised:
        universe_as_of("2020-01-02", _store())
    message = str(raised.value)
    assert "latest" in message
    assert "survivors" in message, (
        "the refusal does not say WHY, so the next reader will widen the "
        "guard rather than fetch the data"
    )


def test_an_empty_store_refuses_rather_than_returning_nothing():
    """An empty set would read as 'no name existed', which is a measurement."""
    with pytest.raises(NoCoverage):
        universe_as_of("2008-01-02", _frame([]))


def test_a_store_with_no_delistings_refuses_every_date():
    """This is the current universe in miniature: 110 names, zero deaths. If
    such a store answered, it would be `assets.yaml` with extra steps."""
    only_alive = _frame([
        {"ticker": "A", "ipo_date": "1990-01-02", "status": "Active"},
        {"ticker": "B", "ipo_date": "1995-01-03", "status": "Active"},
    ])
    with pytest.raises(NoCoverage) as raised:
        universe_as_of("2008-01-02", only_alive)
    assert "#169" in str(raised.value)


def test_the_dead_are_findable_by_window():
    dead = died_between("2005-01-01", "2013-01-01", _store())
    assert list(dead["ticker"]) == ["OLD", "DEAD"]
    assert died_between("2015-01-01", "2016-01-01", _store()).empty


def test_coverage_states_both_ends_and_the_source():
    known = coverage(_store())
    assert known.rows == 4
    assert known.delisted == 2
    assert str(known.earliest_delisting.date()) == "2005-06-30"
    assert str(known.latest_delisting.date()) == "2012-11-15"
    described = known.describe()
    assert "2005-06-30" in described and "2012-11-15" in described


def test_a_write_that_loses_delistings_is_audible(tmp_path, caplog):
    """The store's whole purpose is the dead names. A refetch that returned
    only the active list would look like a healthy 14,000-row file while
    deleting the point of it -- the #138 shape, in a new place."""
    path = tmp_path / "universe.parquet"
    save(_store(), path)

    thinner = _frame([
        {"ticker": "ALIVE", "ipo_date": "1995-05-05", "status": "Active"},
        {"ticker": "LATE", "ipo_date": "2010-08-01", "status": "Active"},
    ])
    with caplog.at_level(logging.ERROR):
        save(thinner, path)

    said = " ".join(r.getMessage() for r in caplog.records
                    if r.levelno >= logging.ERROR)
    assert "FEWER DELISTINGS" in said
    assert "#169" in said


def test_a_growing_store_does_not_cry_wolf(tmp_path, caplog):
    path = tmp_path / "universe.parquet"
    save(_store(), path)
    richer = pd.concat([_store(), _frame([
        {"ticker": "GONE", "ipo_date": "2000-01-03",
         "delisting_date": "2011-02-02", "status": "Delisted"},
    ])], ignore_index=True)
    with caplog.at_level(logging.ERROR):
        save(richer, path)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_round_trip_keeps_the_dates(tmp_path):
    path = tmp_path / "universe.parquet"
    save(_store(), path)
    back = load(path)
    assert len(back) == 4
    assert str(back.set_index("ticker").loc["DEAD", "delisting_date"].date()) \
        == "2012-11-15"
    assert universe_as_of("2008-01-02", store=path) == {"DEAD", "ALIVE"}
