"""Remove the restored 15m bars whose ticker labels are scrambled. Undoes REGISTER #218.

WHAT HAPPENED, in order, because the order is the whole argument:

    2026-08-04  a prepare run drops the whole 15m timeframe:
                cross_ticker_duplicate_ohlcv, extreme_return_contamination
    2026-08-05  someone deletes 44,315 15m bars by hand, over 24 tickers,
                covering 2026-03-16 -> 2026-06-08
    2026-08-30  15m passes the filter and the ladder grades seven 15m champions
    2026-09-01  I restore those bars (REGISTER #218), reading the unexplained
                deletion as an accident
    2026-09-02  run 14 drops the whole 15m timeframe again, for the same two
                reasons, and stage 3 spends two hours on 1d and 60m alone

The explanation for the deletion was in this repository the entire time, in the
docstring of `tests/unit/test_price_filter_drop_reporting.py`: "4,668 rows of
15m carry prices belonging to another instrument (KO above 200, INTC above
300), 16 of 24 tickers have a 15m range inconsistent with their own daily
range". I wrote that the deletion could not be explained without looking for
the explanation.

MEASURED 2026-09-02, on the live table, by the price filter's own function:

    the table WITHOUT the restored rows   171,488 rows, 112 tickers
        cross_ticker_duplicate_ratio  0.000000     extreme_return_ratio 0.000006
        -> kept

    the restored rows ALONE                44,315 rows,  24 tickers
        cross_ticker_duplicate_ratio  0.429629     extreme_return_ratio 0.107268
        -> dropped on both

43% of the restored rows carry an OHLCV bar that also appears under a different
ticker -- 8,448 distinct bars, each worn by two, three, and in two cases four
names at once. KO's restored 15m closes span 41.0 to 999.0 while KO's own daily
range over those dates is 73.6 to 82.1. Seventeen of the 24 tickers are outside
their own daily range that way, and eighteen carry bars stamped 00:00 UTC --
daily bars wearing a 15m label.

WHY THE ROWS CANNOT BE FILTERED INSTEAD, which is what I planned before
measuring: the corruption is in the association between a bar and a ticker. A
mislabelled bar is detectable only when two tickers happen to collide on the
same OHLCV. A bar that received exactly one wrong label leaves no trace and
looks like ordinary data. So the duplicate count is a lower bound on the damage
and there is no test that separates good rows from bad ones inside an affected
ticker -- only whole tickers can be judged.

WHAT SURVIVES: six ETFs (MOO, XHB, XLE, XLF, XLK, XLV) with no duplicate rows,
no midnight stamps, no move above 50%, and prices inside their own daily range.
260 bars each: 1,560 rows, 3.5% of the block, covering 2026-05-26 to
2026-06-08. Thirteen days for six names. They are kept because they are correct,
not because they are worth much -- the intraday power verdict does not move
(CLAIMS R8: the smallest annualised Sharpe 15m can distinguish from zero is
about 6, whatever we do to the record's length).

The judgement is made by re-deriving the four criteria here rather than by a
hard-coded list of tickers, so the script says what it decided and why on every
run instead of asserting a conclusion someone reached once.

REVERSIBILITY: `market_data_raw_prepurge_20260805` still holds every one of
these rows, and `tests/contracts/test_intraday_history_only_grows.py` keeps it
there. Nothing here is the last copy of anything.

    python scripts/maintenance/remove_scrambled_intraday_bars.py
    python scripts/maintenance/remove_scrambled_intraday_bars.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"
BACKUP = "market_data_raw_prepurge_20260805"

#: Everything earlier than this in the 15m slice is a restored row: the live
#: table began here before the restore.
BOUNDARY = pd.Timestamp("2026-06-09", tz="UTC")

IDENTITY = ["datetime", "open", "high", "low", "close", "volume"]


def _verdicts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    restored = con.execute(
        """select datetime, ticker, open, high, low, close, volume
           from market_data_raw
           where interval = '15m' and datetime < timestamp '2026-06-09'"""
    ).fetchdf()
    if restored.empty:
        return pd.DataFrame()
    restored["datetime"] = pd.to_datetime(restored["datetime"], utc=True)

    daily = con.execute(
        """select ticker, min(low) as day_lo, max(high) as day_hi
           from market_data_raw
           where interval = '1d' and datetime >= timestamp '2026-03-01'
             and datetime < timestamp '2026-06-10'
           group by ticker"""
    ).fetchdf().set_index("ticker")

    duplicated = restored[restored.duplicated(IDENTITY, keep=False)]
    cross = duplicated[duplicated.groupby(IDENTITY)["ticker"].transform("nunique") > 1]

    ordered = restored.sort_values(["ticker", "datetime"])
    returns = ordered.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    jumps = ordered[returns.abs() > 0.50]

    midnight = restored[
        (restored["datetime"].dt.hour == 0) & (restored["datetime"].dt.minute == 0)
    ]

    table = restored.groupby("ticker").size().rename("rows").to_frame()
    for name, frame in (("shared_with_another_ticker", cross),
                        ("stamped_midnight", midnight),
                        ("moves_over_50pct", jumps)):
        table[name] = frame.groupby("ticker").size().reindex(table.index).fillna(0).astype(int)

    lows = restored.groupby("ticker")["close"].min()
    highs = restored.groupby("ticker")["close"].max()
    table = table.join(daily)
    table["outside_own_daily_range"] = (
        (lows < table["day_lo"] * 0.7) | (highs > table["day_hi"] * 1.3)
    ).fillna(True)

    table["keep"] = (
        (table["shared_with_another_ticker"] == 0)
        & (table["stamped_midnight"] == 0)
        & (table["moves_over_50pct"] == 0)
        & (~table["outside_own_daily_range"])
    )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="delete the rows; without it nothing is touched")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=not args.apply)

    tables = {row[0] for row in con.execute(
        "select table_name from information_schema.tables").fetchall()}
    if BACKUP not in tables:
        print(f"REFUSING: {BACKUP} is missing, so this deletion would not be "
              f"reversible. It is the only other copy of these rows.")
        return 1

    table = _verdicts(con)
    if table.empty:
        print("no restored 15m rows remain; nothing to do.")
        return 0

    print(table[["rows", "shared_with_another_ticker", "stamped_midnight",
                 "moves_over_50pct", "outside_own_daily_range", "keep"]]
          .sort_values(["keep", "rows"], ascending=[False, False]).to_string())

    drop = table[~table["keep"]]
    keep = table[table["keep"]]
    print(f"\nkeep:   {len(keep):>3} tickers, {int(keep['rows'].sum()):>7,} rows")
    print(f"delete: {len(drop):>3} tickers, {int(drop['rows'].sum()):>7,} rows")

    if not args.apply:
        print("\ndry run: nothing written. Re-run with --apply.")
        return 0

    before = con.execute(
        "select count(*) from market_data_raw where interval = '15m'"
    ).fetchone()[0]

    con.execute(
        """delete from market_data_raw
           where interval = '15m'
             and datetime < timestamp '2026-06-09'
             and ticker in (select unnest($tickers))""",
        {"tickers": list(drop.index)},
    )

    after = con.execute(
        "select count(*) from market_data_raw where interval = '15m'"
    ).fetchone()[0]
    span = con.execute(
        """select min(datetime), max(datetime), count(distinct ticker)
           from market_data_raw where interval = '15m'"""
    ).fetchone()
    con.close()

    print(f"\ndeleted {before - after:,} rows ({before:,} -> {after:,})")
    print(f"15m now spans {span[0]} -> {span[1]} over {span[2]} tickers")
    print(f"the deleted rows remain in {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
