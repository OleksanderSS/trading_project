"""The earliest intraday bar may move backwards or stay; it may never move forward.

A daily bar deleted today can be downloaded again tomorrow. An intraday bar
cannot: Yahoo serves at most 60 days of 15m and 730 days of 60m per request
(`yf_collector._INTRADAY_HISTORY_LIMIT_DAYS`), so intraday history is
ACCUMULATED forward and never refetched. Deleting one destroys it.

So the guard is on the DATA, not on the code -- there is no code to guard. The
earliest bar per intraday cadence is pinned below; a run that starts later than
the pin has lost history that cannot be bought back.

READ THE PIN'S OWN COMMENT BEFORE TRUSTING THIS FILE. It was written on
2026-09-01 to defend 44,315 bars that a manual operation had removed on
2026-08-05, on the reasoning that the deletion was unexplained and therefore
accidental. Measured 2026-09-02: the deletion was a deliberate cleanup, the
bars were corrupt, and this test spent a day pinning them in place. The
explanation was in this repository the whole time, in the docstring of
`tests/unit/test_price_filter_drop_reporting.py`.

What survives of the original argument is the arithmetic and nothing else: the
length of an intraday record is the only thing that sets what an intraday
result can ever prove, and at any length reachable here the smallest annualised
Sharpe 15m can tell apart from zero is around 6 (CLAIMS.md R8). That was true
of the restored record too, which is why restoring it was never going to
rescue anything even if the bars had been clean.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"

#: Earliest bar per cadence. These may only be moved EARLIER without a reason
#: written down. Moving one later means bars were deleted, and for an intraday
#: cadence that is normally permanent -- so a later pin needs an entry in
#: REGISTER saying why, which is what happened here.
#:
#: THE 15m PIN MOVED FORWARD ON 2026-09-02, from 2026-03-16 to 2026-05-26, and
#: the reason is that the earlier pin was protecting corrupt data (REGISTER
#: #228). The restore this file was written to defend put back 44,315 bars whose
#: TICKER LABELS were scrambled: 43% of them carried an OHLCV bar that also
#: appeared under a different ticker -- 8,448 distinct bars worn by two, three,
#: and twice by four names at once -- and KO's restored 15m closes spanned
#: 41.0 to 999.0 against its own daily range of 73.6 to 82.1. Seventeen of the
#: 24 tickers were outside their own daily range and eighteen carried bars
#: stamped 00:00 UTC, which is a daily bar wearing a 15m label.
#:
#: Only whole tickers could be judged: a bar that received exactly one wrong
#: label leaves no trace, so 43% was a lower bound and no filter could separate
#: good rows from bad inside an affected name. Six ETFs survived every check --
#: MOO, XHB, XLE, XLF, XLK, XLV, 1,560 rows, 3.5% of the block -- and their
#: earliest bar is the new pin. `scripts/maintenance/remove_scrambled_intraday_bars.py`
#: removed the other 42,755, after which the filter keeps 15m: duplicate ratio
#: 0.000000, extreme return ratio 0.000006, 173,048 rows over 112 tickers.
#:
#: The lesson this file now carries is not the one it was written with. An
#: invariant that pins DATA rather than code can end up defending the defect it
#: was meant to prevent, and this one did for a day.
EARLIEST = {
    "15m": datetime(2026, 5, 26, tzinfo=timezone.utc),
    "1h": datetime(2024, 8, 19, tzinfo=timezone.utc),
}

#: Daily is excluded on purpose: it is refetchable, so an earlier bar going
#: missing is a nuisance rather than a loss, and pinning it would make this
#: test fail for a reason it is not about.


@pytest.fixture(scope="module")
def spans():
    duckdb = pytest.importorskip("duckdb")
    if not DB_PATH.exists():
        pytest.skip(f"no database at {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """select interval, min(datetime) as first_bar, max(datetime) as last_bar,
                  count(*) as bars
           from market_data_raw group by interval"""
    ).fetchall()
    return {row[0]: {"first": row[1], "last": row[2], "bars": row[3]} for row in rows}


@pytest.mark.parametrize("cadence", sorted(EARLIEST))
def test_intraday_history_was_not_shortened(spans, cadence):
    if cadence not in spans:
        pytest.skip(f"no {cadence} bars stored")

    first = spans[cadence]["first"]
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)
    pinned = EARLIEST[cadence]

    # A day of slack: the pin is a date and the first bar is a session open.
    assert first.date() <= pinned.date(), (
        f"the earliest {cadence} bar is {first.date()}, later than the pinned "
        f"{pinned.date()}. Intraday bars cannot be refetched -- Yahoo serves "
        f"60 days of 15m and 730 of 60m -- so history that moved forward is "
        f"gone. If this was deliberate, say why in REGISTER and move the pin. "
        f"If it was not, a backup table may still have the rows -- but CHECK "
        f"THEM BEFORE PUTTING THEM BACK. On 2026-09-01 rows were restored from "
        f"exactly such a table without checking, and 43% of them carried "
        f"another ticker's prices (REGISTER #228). "
        f"`scripts/diagnostics/why_15m_is_dropped.py` is the check."
    )


def test_the_backup_that_held_the_lost_bars_is_still_there(spans):
    """Until the restore is confirmed in a rebuilt batch, keep the evidence.

    It now holds the 42,755 scrambled bars that were deleted on 2026-09-02 as
    well as the 1,560 that were kept, so it is the only remaining copy of what
    the restore put in the live table. Keep it: if the judgement in #228 is ever
    shown wrong, this is what it would be re-checked against.
    """
    duckdb = pytest.importorskip("duckdb")
    if not DB_PATH.exists():
        pytest.skip(f"no database at {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = {row[0] for row in con.execute(
        "select table_name from information_schema.tables"
    ).fetchall()}
    assert "market_data_raw_prepurge_20260805" in tables, (
        "the pre-purge backup is gone. It held the only copy of 44,315 15m "
        "bars for four weeks; drop it only after a batch has been rebuilt "
        "from the restored live table and the span confirmed."
    )
