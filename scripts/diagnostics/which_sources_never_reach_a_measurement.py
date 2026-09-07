"""Which collected sources land entirely inside the seal, and so contribute nothing.

THE OWNER'S POINT, and it is the right one: a collector that "works" can still
be wrong, and a wrong stage poisons the answer on one ticker exactly as on a
thousand. This is one measurable instance of it.

`fear_greed_data` holds 312 rows, the newest 2026-09-02. The seal starts
2023-09-01, so every row it has ever collected sits inside the held-back
period. It contributes ZERO rows to every measurement this project has made --
and the feature catalogue does not say so. It prints `t = NaN` and the verdict
"market-wide: use as interaction", which reads as a statement about the DATA
when the truth is a statement about its ABSENCE. A reader cannot tell "this
series does not vary across names" from "this series does not exist here".

So the scan asks one question of every collected table: how many of its rows
are EXPLORABLE -- before the seal -- and how many years do they span. A source
whose explorable share is zero is invisible to every verdict the project has
issued, however healthy its collector looks.

    python scripts/diagnostics/which_sources_never_reach_a_measurement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.sealed_period import SEAL_START  # noqa: E402

DATABASE = PROJECT_ROOT / "data" / "trading_data.duckdb"

#: Column names a table might use for its own time axis, in the order we trust
#: them. `collected_at` is deliberately LAST and only as a fallback: it is when
#: WE fetched the row, not when the fact happened, and treating it as the
#: latter is the defect that made every FRED observation knowable in 2026
#: (#286).
TIME_COLUMNS = ("date", "datetime", "observation_date", "period_end",
                "trade_date", "filing_date", "filingDate", "reportDate",
                "published_date", "publishedAt", "decision_timestamp",
                "timestamp", "released_at", "filed", "collected_at")

#: Tables that are BACKUPS or scratch. They are outside the explorable period
#: on purpose and reporting them as findings would bury the four that matter --
#: the first version of this scan did exactly that, listing six backups among
#: twenty-one "sources reaching no measurement".
IGNORED = ("_backup", "_prepurge", "_orphan", "test_", "cache_metadata")


def main() -> int:
    if not DATABASE.exists():
        print(f"no database at {DATABASE}")
        return 1
    connection = duckdb.connect(str(DATABASE), read_only=True)
    tables = [row[0] for row in connection.execute(
        "select table_name from information_schema.tables "
        "where table_schema = 'main' order by table_name").fetchall()]
    seal = pd.Timestamp(SEAL_START).tz_localize(None)
    print(f"seal starts {seal.date()}; a row is EXPLORABLE if it is earlier\n")

    header = (f"{'table':<34}{'rows':>12}{'time column':>18}"
              f"{'explorable':>12}{'share':>8}{'years':>8}")
    print(header)
    print("-" * len(header))
    silent = []
    for table in tables:
        if any(mark in table for mark in IGNORED):
            continue
        try:
            count = connection.execute(
                f'select count(*) from "{table}"').fetchone()[0]
        except Exception:  # noqa: BLE001 - a scan reports, never raises
            continue
        if not count:
            print(f"{table:<34}{0:>12}{'--':>18}{'--':>12}{'--':>8}{'--':>8}")
            silent.append((table, 0, "empty"))
            continue
        columns = [row[0] for row in
                   connection.execute(f'describe "{table}"').fetchall()]
        stamp = next((name for name in TIME_COLUMNS if name in columns), None)
        if stamp is None:
            print(f"{table:<34}{count:>12,}{'(none found)':>18}"
                  f"{'?':>12}{'?':>8}{'?':>8}")
            silent.append((table, count, "no time column"))
            continue
        frame = connection.execute(
            f'select "{stamp}" as t from "{table}"').fetch_df()
        times = pd.to_datetime(frame["t"], errors="coerce", utc=True)
        times = times.dropna().dt.tz_localize(None)
        if times.empty:
            print(f"{table:<34}{count:>12,}{stamp:>18}{'unparsed':>12}"
                  f"{'--':>8}{'--':>8}")
            silent.append((table, count, f"{stamp} does not parse as a time"))
            continue
        explorable = int((times < seal).sum())
        share = explorable / len(times)
        span = ((times[times < seal].max() - times[times < seal].min()).days
                / 365.25) if explorable else 0.0
        print(f"{table:<34}{count:>12,}{stamp:>18}{explorable:>12,}"
              f"{share:>8.0%}{span:>8.1f}")
        if explorable == 0:
            silent.append((table, count, "every row is inside the seal"))

    connection.close()
    print()
    if silent:
        print("SOURCES THAT REACH NO MEASUREMENT, and why:\n")
        for table, count, reason in silent:
            print(f"    {table:<34}{count:>10,} rows   {reason}")
        print("\n    A healthy collector filling a table nobody can measure "
              "looks identical,\n    from the outside, to one that works. The "
              "catalogue prints a verdict for\n    these columns rather than "
              "saying it has nothing to judge.")
    else:
        print("every table with a time column reaches the explorable period.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
