"""Why does the price filter drop the whole 15m timeframe, and did the restore cause it?

Run 14 (2026-09-02, 13:45:42) dropped 15m before it ever reached enrichment:

    PriceFilter - ERROR - Timeframe '15m' DROPPED on
      cross_ticker_duplicate_ohlcv,extreme_return_contamination (206833 rows).
      cadence_match=0.9514, extreme_return_ratio=0.02054.
      Thresholds: cadence>=0.60, extreme<=0.010.

Two things make that worth measuring rather than arguing about.

FIRST, it is not new and it is not consistent. `price_filter.py` records that
the 2026-08-04 batch lost 15m the same way -- "no 15m features, no 15m targets,
and 0 of 506 champions on 15m". Then on 2026-08-05 a manual operation deleted
44,315 15m bars covering 2026-03-16 -> 2026-06-08 over 24 tickers. AFTER that
deletion, 15m passed: the 2026-08-30 ladder graded seven 15m champions
(REGISTER #173). On 2026-09-01 those bars were restored (REGISTER #218), and
15m is dropped again.

Deleted -> passes. Restored -> dropped. That ordering is the reason for this
script and it points one way, so state the uncomfortable reading plainly: the
2026-08-05 purge may have been someone removing contaminated rows on purpose,
and the restore may have put the contamination back. The restore checked three
things -- that backup and live agree where they overlap, that prices are
usable, that hashes match the collector's -- and none of them would notice
either defect the filter is complaining about. "No code performs that deletion"
was read as "the deletion was an accident". It is equally consistent with "a
person deleted bad rows by hand", and that alternative was never tested.

It has now been answered, by a measurement that was already in this repository
when the restore was written. `tests/unit/test_price_filter_drop_reporting.py`
records, of the 2026-08-05 run:

    4,668 rows of 15m carry prices belonging to another instrument (KO above
    200, INTC above 300), 16 of 24 tickers have a 15m range inconsistent with
    their own daily range, and 1d and 1h have none of it.

Sixteen of the same twenty-four tickers whose bars were purged and then
restored. So the purge was a deliberate cleanup of measured contamination, and
the restore put it back. This script is therefore no longer here to decide
WHETHER; it is here to say WHICH ROWS, so the fix can delete the contaminated
bars and keep the rest of the recovered history rather than repeating a blanket
purge.

SECOND, whichever way it falls, a timeframe is being decided by a whole-frame
verdict. `extreme_return_ratio` is one number over 110 tickers, so a handful of
corrupt series drops the cadence for every clean one. That is a design question
this script also feeds: the output says how many TICKERS carry the contamination.

WHAT IS MEASURED

The filter's own function is imported and called -- `_assess_temporal_identity`
-- rather than reimplemented here, so this reports what the pipeline decides
and not what a second copy of the arithmetic decides.

It is called three times on the stored 15m bars:

    all rows                     what the run saw
    rows from 2026-06-09 on      the live table as it stood BEFORE the restore
    the restored rows alone      2026-03-16 -> 2026-06-08, the 24 tickers

If the middle one passes both thresholds and the first does not, the restore is
the cause and REGISTER #218 has to be reopened against my own change. If the
middle one fails too, the contamination is older, the restore is innocent, and
the 2026-08-30 15m champions were graded on data this filter would reject.

CAVEAT, so the numbers are not over-read: the pipeline applies
`PricePreprocessor` and `DataCleaner.remove_outliers_zscore` before the filter
sees the frame. This reads `market_data_raw` directly, so absolute ratios here
need not equal the run's 0.02054. The COMPARISON between the three slices is
what this is for, and it is unaffected -- all three get the same treatment.

    python scripts/diagnostics/why_15m_is_dropped.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.processing.filters.price_filter import PriceFilter  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"

#: The live 15m table began here before the restore; everything earlier in the
#: 15m slice today is a restored row. Taken from the restore script's own
#: measurement, not guessed.
RESTORE_BOUNDARY = pd.Timestamp("2026-06-09", tz="UTC")

#: Thresholds are read from the config the pipeline reads, so this cannot drift
#: away from what actually decides the drop.
CONFIG = PROJECT_ROOT / "src" / "config" / "processing.yaml"


def _filter_thresholds() -> dict:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    found: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {
                    "max_cross_ticker_duplicate_ratio",
                    "min_cadence_match_ratio",
                    "max_extreme_return_ratio",
                }:
                    found[key] = value
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return found


def _load_15m(db: Path) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    frame = con.execute(
        """select datetime, ticker, interval, open, high, low, close, volume
           from market_data_raw where interval = '15m'"""
    ).fetchdf()
    con.close()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame


def _report(name: str, frame: pd.DataFrame, thresholds: dict, filt: PriceFilter) -> dict:
    if frame.empty:
        print(f"\n{name}: no rows")
        return {}
    quality = filt._assess_temporal_identity(frame)
    duplicate = quality["cross_ticker_duplicate_ratio"]
    cadence = quality["cadence_match_ratio"]
    extreme = quality["extreme_return_ratio"]

    max_dup = thresholds.get("max_cross_ticker_duplicate_ratio", filt.max_cross_ticker_duplicate_ratio)
    min_cad = thresholds.get("min_cadence_match_ratio", filt.min_cadence_match_ratio)
    max_ext = thresholds.get("max_extreme_return_ratio", filt.max_extreme_return_ratio)

    failures = []
    if duplicate > max_dup:
        failures.append("cross_ticker_duplicate_ohlcv")
    if cadence is not None and cadence < min_cad:
        failures.append("timeframe_cadence_mismatch")
    if extreme > max_ext:
        failures.append("extreme_return_contamination")

    print(f"\n{name}")
    print(f"  rows {len(frame):,}   tickers {frame['ticker'].nunique()}   "
          f"{frame['datetime'].min()} -> {frame['datetime'].max()}")
    print(f"  cross_ticker_duplicate_ratio {duplicate:.6f}  (limit <= {max_dup})")
    print(f"  cadence_match_ratio          {cadence}  (limit >= {min_cad})")
    print(f"  extreme_return_ratio         {extreme:.6f}  (limit <= {max_ext})")
    print(f"  VERDICT: {'DROPPED on ' + ','.join(failures) if failures else 'kept'}")
    return {"failures": failures, **quality}


def _blame_extremes(frame: pd.DataFrame, limit: int = 15) -> None:
    """Which tickers and which bars produce the >50% moves.

    The filter reports one ratio for the whole frame. If the moves sit in a few
    names, dropping 110 tickers' worth of cadence for them is a different
    decision from dropping a frame that is bad throughout -- so count both.
    """
    work = frame.sort_values(["ticker", "datetime"]).copy()
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["prev_close"] = work.groupby("ticker", sort=False)["close"].shift(1)
    work["ret"] = work.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    bad = work[work["ret"].abs() > 0.50].dropna(subset=["ret"])
    if bad.empty:
        print("\n  no bar-to-bar move above 50%")
        return

    per_ticker = bad.groupby("ticker").size().sort_values(ascending=False)
    print(f"\n  {len(bad):,} bars move more than 50% in one 15m bar, "
          f"over {len(per_ticker)} of {frame['ticker'].nunique()} tickers")
    print("  worst tickers by count:")
    for ticker, count in per_ticker.head(limit).items():
        share = count / (work["ticker"] == ticker).sum()
        print(f"    {ticker:<8} {count:>6,} bars  ({share:.1%} of its own rows)")

    print("  a few of the actual moves:")
    for _, row in bad.nlargest(min(limit, len(bad)), "ret").head(8).iterrows():
        print(f"    {row['ticker']:<8} {row['datetime']}  "
              f"{row['prev_close']:.4f} -> {row['close']:.4f}  ({row['ret']:+.1%})")


def _blame_duplicates(frame: pd.DataFrame, limit: int = 10) -> None:
    """Which tickers share an identical bar with another ticker.

    Identical open/high/low/close AND identical volume at the same instant for
    two different instruments is not a market event.
    """
    identity = ["datetime", "open", "high", "low", "close", "volume"]
    if not set(identity).issubset(frame.columns):
        print("\n  identity columns missing; duplicates not attributable")
        return
    duplicated = frame[frame.duplicated(identity, keep=False)]
    if duplicated.empty:
        print("\n  no repeated OHLCV rows at all")
        return
    grouped = duplicated.groupby(identity, dropna=False)["ticker"]
    cross = duplicated[grouped.transform("nunique") > 1]
    if cross.empty:
        print(f"\n  {len(duplicated):,} repeated OHLCV rows, but all within a "
              "single ticker (a flat bar repeating), none across tickers")
        return

    print(f"\n  {len(cross):,} rows share an identical bar with a DIFFERENT ticker")
    per_ticker = cross.groupby("ticker").size().sort_values(ascending=False)
    print("  tickers involved:")
    for ticker, count in per_ticker.head(limit).items():
        print(f"    {ticker:<8} {count:>6,} rows")
    example = cross.sort_values(identity).head(6)
    print("  example rows:")
    for _, row in example.iterrows():
        print(f"    {row['ticker']:<8} {row['datetime']}  o={row['open']} "
              f"h={row['high']} l={row['low']} c={row['close']} v={row['volume']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"no database at {args.db}")
        return 1

    thresholds = _filter_thresholds()
    filt = PriceFilter(thresholds)

    frame = _load_15m(args.db)
    restored = frame[frame["datetime"] < RESTORE_BOUNDARY]
    live = frame[frame["datetime"] >= RESTORE_BOUNDARY]

    print(f"thresholds read from {CONFIG.name}: {thresholds}")

    whole = _report("ALL 15m rows (what run 14 saw)", frame, thresholds, filt)
    _blame_extremes(frame)
    _blame_duplicates(frame)

    pre = _report(
        f"WITHOUT the restored rows (>= {RESTORE_BOUNDARY.date()}) "
        "-- the table as it stood on 2026-08-30",
        live, thresholds, filt,
    )

    _report(
        f"THE RESTORED ROWS ALONE (< {RESTORE_BOUNDARY.date()})",
        restored, thresholds, filt,
    )
    _blame_extremes(restored)

    print("\n" + "=" * 70)
    if whole.get("failures") and not pre.get("failures"):
        print("READING: the restore is the cause. The pre-restore table passes "
              "both checks and the restored one does not.\n"
              "REGISTER #218 is wrong as written and reopens against my own "
              "change; the 2026-08-05 purge was a deliberate cleanup.")
    elif whole.get("failures") and pre.get("failures"):
        print("READING: the contamination is older than the restore. 15m was "
              "already unfit before 2026-06-09, which also means the seven 15m "
              "champions graded on 2026-08-30 were graded on data this filter "
              "would reject (REGISTER #173).")
    elif not whole.get("failures"):
        print("READING: measured on raw stored bars nothing fails, so the "
              "contamination is produced BETWEEN the store and the filter -- "
              "PricePreprocessor or DataCleaner. Measure there next.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
