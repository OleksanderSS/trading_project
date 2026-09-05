"""Collecting the same number twice must not store it twice.

REGISTER #142. The FRED table's dedup hash was
["series_id", "date", "realtime_start", "value"], and for an UNREVISED daily
series `realtime_start` is the day we asked -- #130 measured that FRED stamps
a non-vintage request with the request date. So every collection minted a
fresh hash for an observation whose value had not moved, `filter_new_records`
filtered nothing, and the table grew by one whole copy per run.

MEASURED ON THE LIVE TABLE, 2026-09-05:

    485,010 rows for 97,130 distinct (series, date) pairs -- 4.99x
    85,478 rows added on 2026-09-02; 85,469 and 85,467 on the two runs before
    of the 18 unrevised series' 398,042 rows, 308,200 (77.4%) repeat a value
      that already existed under a different collection date
    corrected key: the table would hold 176,800 rows, 63.5% smaller

WHY THE BRANCH IS ON THE SERIES AND NEVER ON THE TABLE. #131 restored the
vintage structure after a blanket dedup collapsed 314,062 rows to 97,090 and
destroyed it. Applying the shorter key to CPI or GDP would undo that fix. For
a revised series the vintage IS the record; for an unrevised one it is the
date we happened to ask.

And `value` stays in the key either way: for a series FRED does not revise, a
changed value is news, not a duplicate.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.collectors.fred_collector import FredCollector

UNREVISED = sorted(FredCollector.UNREVISED_DAILY_SERIES)


@pytest.fixture()
def collector():
    made = FredCollector.__new__(FredCollector)
    made.hash_keys = ["series_id", "date", "realtime_start", "value"]
    return made


def _row(series_id: str, asked_on: str, value: str = "1.5") -> pd.Series:
    return pd.Series({
        "series_id": series_id, "date": "2020-01-02",
        "realtime_start": asked_on, "value": value,
    })


def test_an_unrevised_series_keys_on_the_fact_not_the_asking(collector):
    series = UNREVISED[0]
    monday = collector._generate_hash(_row(series, "2026-09-02"))
    friday = collector._generate_hash(_row(series, "2026-09-05"))
    assert monday == friday, (
        "the same observation collected on two days still produces two hashes, "
        "so the table grows by a full copy on every run"
    )


def test_a_changed_value_is_still_new(collector):
    """Dropping the collection date must not drop the observation."""
    series = UNREVISED[0]
    assert collector._generate_hash(_row(series, "2026-09-05", "1.5")) != \
        collector._generate_hash(_row(series, "2026-09-05", "1.6"))


@pytest.mark.parametrize("series", ["CPIAUCSL", "GDP", "UNRATE"])
def test_a_revised_series_keeps_its_vintages(collector, series):
    """#131 restored this structure after a blanket dedup destroyed it. The
    shorter key must never reach these."""
    if series in FredCollector.UNREVISED_DAILY_SERIES:
        pytest.skip(f"{series} is in the unrevised set; nothing to protect")
    monday = collector._generate_hash(_row(series, "2026-09-02"))
    friday = collector._generate_hash(_row(series, "2026-09-05"))
    assert monday != friday, (
        f"{series} is revised, so two vintages of one observation collapsed "
        "into one row -- which is the #131 defect returning"
    )


def test_the_key_differs_only_by_the_collection_date(collector):
    """The two keys must not drift apart in any other way: a second difference
    would make the two paths two decisions."""
    unrevised = collector._hash_keys_for(UNREVISED[0])
    revised = collector._hash_keys_for("CPIAUCSL")
    assert set(revised) - set(unrevised) == {"realtime_start"}
    assert set(unrevised) - set(revised) == set()


def test_the_branch_reads_the_series_not_the_table(collector):
    """A table-wide switch is how #131 happened. The decision has to be per
    row."""
    import inspect

    source = inspect.getsource(FredCollector._generate_hash)
    assert "_hash_keys_for" in source
    assert 'row.get("series_id"' in source, (
        "the hash no longer looks at which series it is hashing, so one rule "
        "is being applied to both kinds"
    )


def test_every_unrevised_series_takes_the_short_key(collector):
    """Named as a set rather than trusted: a series added to the frozenset
    later must be covered by the same rule without another edit."""
    for series in UNREVISED:
        assert "realtime_start" not in collector._hash_keys_for(series), series
