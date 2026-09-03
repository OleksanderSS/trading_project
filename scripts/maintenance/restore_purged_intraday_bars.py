"""Put back the intraday bars a manual purge dropped, because they cannot be refetched.

REGISTER #218. On 2026-08-05 something deleted 15m bars from `market_data_raw`
and left `market_data_raw_prepurge_20260805` behind. No code in this repository
performs that deletion, so why it happened cannot be recovered -- which is the
first half of the lesson: an untraceable manual operation on the database lost
data, so this one is a script.

The second half is why it matters more than an ordinary data loss. Yahoo serves
at most 60 days of 15m history per request (`yf_collector`
`_INTRADAY_HISTORY_LIMIT_DAYS`), so intraday history can only be ACCUMULATED
forward, never refetched. A daily bar deleted today can be downloaded again
tomorrow; a 15m bar older than 60 days is gone for good. Measured 2026-09-01:

    live      15m   112 tickers   2026-06-09 -> 2026-08-28   (80 days)
    prepurge  15m    24 tickers   2026-03-16 -> 2026-07-30

    rows in the backup and not in the live table:
        44,944 over 24 tickers, 2026-03-16 -> 2026-06-08

That is roughly a doubling of the 15m record for those 24 names, and the
record's length is the only thing that sets what an intraday result could ever
prove: the minimum annualised Sharpe this frame can tell apart from zero is
6.01 at 80 days and about 4.2 at 165 (CLAIMS.md R8). Restoring does not rescue
intraday research -- 4.2 is still out of reach. It is done because measured
bars that cannot be re-obtained are not something to leave in a backup table.

WHAT IS CHECKED BEFORE ANYTHING IS WRITTEN:

  * the backup agrees with the live table where they overlap. Measured on the
    22,459 shared rows: 126 differ in close (0.56%), largest gap 2.31. Those
    are vendor revisions, not a different series.
  * rows with no usable price are excluded. 629 of the 44,944 carry a null or
    non-positive close and a null volume; they are dropped rather than
    restored, and the count is reported.
  * every restored row is hashed with `bar_identity_hash`, the same function
    the collector uses, so `filter_new_records` sees them as the bars they are
    and the next collection does not store them again.

Dry run by default.

    python scripts/maintenance/restore_purged_intraday_bars.py
    python scripts/maintenance/restore_purged_intraday_bars.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.collectors.yf_collector import bar_identity_hash  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"
LIVE = "market_data_raw"
BACKUP = "market_data_raw_prepurge_20260805"

MISSING_ROWS = f"""
select b.datetime, b.close, b.high, b.low, b.open, b.volume, b.ticker, b.interval
from {BACKUP} b
left join {LIVE} l
  on l.ticker = b.ticker and l.datetime = b.datetime and l.interval = b.interval
where l.ticker is null
"""


def summarise(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return con.execute(
        f"""select interval, count(*) as rows, count(distinct ticker) as tickers,
                   min(datetime) as first_bar, max(datetime) as last_bar
            from {table} group by interval order by rows desc"""
    ).fetchdf()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the rows; without it nothing is touched")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=not args.apply)

    print("before:")
    print(summarise(con, LIVE).to_string(index=False))

    missing = con.execute(MISSING_ROWS).fetchdf()
    if missing.empty:
        print("\nnothing in the backup is absent from the live table.")
        return 0

    usable = missing[
        missing["close"].notna() & (missing["close"] > 0)
        & missing["volume"].notna() & (missing["volume"] >= 0)
    ].copy()
    dropped = len(missing) - len(usable)

    print(
        f"\nmissing from the live table: {len(missing):,} rows over "
        f"{missing['ticker'].nunique()} tickers, "
        f"{missing['datetime'].min()} -> {missing['datetime'].max()}"
    )
    print(f"  unusable (no price or no volume), not restored: {dropped:,}")
    print(f"  to restore: {len(usable):,}")
    print(usable.groupby("interval").size().to_string())

    if not args.apply:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    usable["hash"] = [
        bar_identity_hash(stamp, ticker, interval)
        for stamp, ticker, interval in zip(
            usable["datetime"], usable["ticker"], usable["interval"]
        )
    ]
    con.register("usable", usable)

    clash = con.execute(
        f"select count(*) as n from {LIVE} l join usable u on l.hash = u.hash"
    ).fetchone()[0]
    if clash:
        print(
            f"\nrefusing to write: {clash} of the rows already exist in "
            f"{LIVE} under the same identity hash. The join on "
            "(ticker, datetime, interval) said they were absent, so the two "
            "disagree and one of them is wrong -- resolve that first."
        )
        return 1

    con.execute("begin transaction")
    try:
        con.execute(
            f"""insert into {LIVE} (datetime, close, high, low, open, volume,
                                    ticker, interval, hash)
                select datetime, close, high, low, open, volume,
                       ticker, interval, hash
                from usable"""
        )
        con.execute("commit")
    except Exception:
        con.execute("rollback")
        raise

    print("\nafter:")
    print(summarise(con, LIVE).to_string(index=False))
    print(f"\nrestored {len(usable):,} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
