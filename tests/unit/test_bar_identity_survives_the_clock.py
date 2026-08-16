"""One bar, one hash — whatever clock the machine is set to.

Deduplication in `filter_new_records` keys on this value alone. The hash used
to be taken of a FORMATTED LOCAL-TIME STRING, so the same bar collected under a
different timezone, after a DST change, or with `1h` instead of `60m` produced
a different identity and slipped past the check.

Found on the v14 batch: 540 AAPL 60m bars stored twice between 2026-03-16 and
2026-05-08, every column identical except the hash. Both inputs were recovered
by brute force:

    2026-03-16T13:30:00.000000+0000AAPL1h   (UTC)
    2026-03-16T15:30:00.000000+0200AAPL1h   (Europe/Kiev, the machine)

The same instant written two ways.
"""

import pandas as pd
import pytest

from src.data.collectors.yf_collector import bar_identity_hash


def test_the_same_instant_hashes_the_same_from_any_timezone():
    utc = pd.Timestamp('2026-03-16 13:30:00', tz='UTC')
    reference = bar_identity_hash(utc, 'AAPL', '60m')
    for zone in ('Europe/Kiev', 'America/New_York', 'Asia/Tokyo', 'UTC'):
        assert bar_identity_hash(utc.tz_convert(zone), 'AAPL', '60m') == reference, (
            f'the bar changed identity when the clock was set to {zone}'
        )


def test_the_same_bar_size_hashes_the_same_under_any_label():
    utc = pd.Timestamp('2026-03-16 13:30:00', tz='UTC')
    reference = bar_identity_hash(utc, 'AAPL', '60m')
    for label in ('1h', '1H', '60min', '60m'):
        assert bar_identity_hash(utc, 'AAPL', label) == reference, (
            f"'{label}' and '60m' are one bar size and must be one identity"
        )


def test_a_naive_timestamp_is_read_as_utc():
    naive = pd.Timestamp('2026-03-16 13:30:00')
    aware = pd.Timestamp('2026-03-16 13:30:00', tz='UTC')
    assert bar_identity_hash(naive, 'AAPL', '60m') == bar_identity_hash(aware, 'AAPL', '60m')


def test_the_ticker_is_case_and_space_insensitive():
    utc = pd.Timestamp('2026-03-16 13:30:00', tz='UTC')
    reference = bar_identity_hash(utc, 'AAPL', '60m')
    for name in (' AAPL ', 'aapl', 'Aapl'):
        assert bar_identity_hash(utc, name, '60m') == reference


@pytest.mark.parametrize(
    'timestamp, ticker, interval',
    [
        (pd.Timestamp('2026-03-16 14:30:00', tz='UTC'), 'AAPL', '60m'),
        (pd.Timestamp('2026-03-16 13:30:00', tz='UTC'), 'MSFT', '60m'),
        (pd.Timestamp('2026-03-16 13:30:00', tz='UTC'), 'AAPL', '15m'),
        (pd.Timestamp('2026-03-16 13:30:00', tz='UTC'), 'AAPL', '1d'),
    ],
)
def test_different_bars_keep_different_identities(timestamp, ticker, interval):
    """A hash that collapses everything would pass every test above."""
    reference = bar_identity_hash(pd.Timestamp('2026-03-16 13:30:00', tz='UTC'), 'AAPL', '60m')
    assert bar_identity_hash(timestamp, ticker, interval) != reference


def test_the_two_hashes_that_were_actually_found_now_agree():
    """The exact pair from the v14 batch, reproduced end to end."""
    utc = pd.Timestamp('2026-03-16 13:30:00', tz='UTC')
    kiev = pd.Timestamp('2026-03-16 15:30:00', tz='Europe/Kiev')
    assert utc == kiev, 'the fixture itself must be one instant'
    assert bar_identity_hash(utc, 'AAPL', '1h') == bar_identity_hash(kiev, 'AAPL', '60m')
