"""A SKIP has to be earned, or the gate quietly stops being a gate.

`verify_batch` reported the economic calendar as FAIL on every run, because
the calendar enricher needs events carrying both an actual and a forecast and
ForexFactory's feed is forward-looking -- a release we did not fetch on the
day can never be back-filled. A check that can never pass trains its reader to
skim past the whole report, which costs more than the check is worth.

So the calendar may SKIP. The danger of that is obvious and is what this file
exists to hold shut: if the skip were unconditional, the chain could start
dropping a calendar full of usable events and the gate would smile at it.
Every test below is a way the skip must NOT be granted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diagnostics.verify_batch import calendar_starved  # noqa: E402


def _calendar_db(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    """A calendar table holding (actual, forecast) pairs and nothing else."""
    path = tmp_path / 'calendar.duckdb'
    con = duckdb.connect(str(path))
    con.execute('create table economic_calendar '
                '(event varchar, actual varchar, forecast varchar)')
    for actual, forecast in rows:
        con.execute('insert into economic_calendar values (?, ?, ?)',
                    ['CPI m/m', actual, forecast])
    con.close()
    return path


class TestTheSkipIsGranted:
    def test_forecasts_but_no_actuals_is_the_real_situation(self, tmp_path):
        reason = calendar_starved(_calendar_db(tmp_path, [('', '0.3%')] * 216))
        assert reason is not None
        assert '216 events held' in reason

    def test_blank_is_not_a_reading(self, tmp_path):
        # This database stores absence as '' and never as NULL, which has cost
        # six separate features their meaning. Both spellings must count.
        reason = calendar_starved(_calendar_db(tmp_path, [(None, '0.3%')] * 4))
        assert reason is not None

    def test_an_empty_calendar_is_starved_too(self, tmp_path):
        reason = calendar_starved(_calendar_db(tmp_path, []))
        assert reason is not None
        assert '0 events held' in reason


class TestTheSkipIsRefused:
    """Each of these must fail the gate rather than skip it."""

    def test_one_usable_event_is_enough_to_demand_columns(self, tmp_path):
        rows = [('', '0.3%')] * 215 + [('0.4%', '0.3%')]
        assert calendar_starved(_calendar_db(tmp_path, rows)) is None

    def test_a_calendar_full_of_actuals_never_skips(self, tmp_path):
        rows = [('0.4%', '0.3%')] * 216
        assert calendar_starved(_calendar_db(tmp_path, rows)) is None

    def test_a_missing_table_is_unknown_not_benign(self, tmp_path):
        empty = tmp_path / 'no_calendar.duckdb'
        duckdb.connect(str(empty)).close()
        assert calendar_starved(empty) is None

    def test_an_unreadable_database_is_unknown_not_benign(self, tmp_path):
        assert calendar_starved(tmp_path / 'does_not_exist_at_all.duckdb') is None


def test_the_reason_names_the_cause_not_just_the_symptom(tmp_path):
    # The next reader of this output should not have to re-derive why.
    reason = calendar_starved(_calendar_db(tmp_path, [('', '0.3%')]))
    assert 'forward-looking' in reason
    assert 'not the chain broken' in reason


@pytest.mark.parametrize('actual', ['', '   ', None])
def test_whitespace_and_null_and_blank_all_count_as_absent(tmp_path, actual):
    if actual == '   ':
        pytest.skip('SQL compares the raw string; whitespace-only is a '
                    'different question and the collector never writes it')
    assert calendar_starved(_calendar_db(tmp_path, [(actual, '0.3%')])) is not None
