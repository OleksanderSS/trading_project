"""Did the backfilled sources actually reach the feature panel?

Three sources were made to reach the explorable period on 2026-09-07/08: VIX
(0 -> 6,791 explorable rows, #298), sec_filings (0 -> 442,127, #298 and #299)
and the news tables' UTC conversion (#294). None of that means a FEATURE is
alive: the assembler decides that, and only a rebuild answers it.

This is the control for the rebuild. Run it against the preserved batch and
against the new one; the difference is the answer. Without the before, the
after is just a number.

    python scripts/diagnostics/did_the_backfill_reach_the_features.py
    python scripts/diagnostics/did_the_backfill_reach_the_features.py --batch data/colab/accumulated/pre_backfill_20260908
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.pipeline.sealed_period import SEAL_START  # noqa: E402

DEFAULT_BATCH = PROJECT_ROOT / "data/colab/accumulated/main_database"

#: The columns the backfills were supposed to bring to life, matched
#: case-insensitively. The point is the SHARE of a family that carries a real
#: value before the seal, not any single column.
#:
#: `vix (ours)` and `vix (FRED)` are SEPARATE on purpose, and the reason is the
#: first thing this script found. The panel's VIX columns are named
#: `FRED_VIXCLS_*` -- they arrive through FRED, not through our `vix_data`
#: table -- and on the pre-backfill batch `FRED_VIXCLS_1d` was ALREADY alive:
#: 623,165 of 623,398 explorable rows, 2,424 distinct values, across the whole
#: window. So the VIX level was never missing from the panel.
#:
#: What our own table contributes is the DERIVED state -- volatility_regime,
#: the percentiles, the z-score, the classification -- and those were absent
#: before the seal because the table itself was (#298). Lumping the two
#: together would have credited a backfill with data that FRED had supplied
#: all along, which is the shape of a manufactured result.
FAMILIES = {
    "vix (ours)": ("volatility_regime", "extreme_volatility", "vix_sma",
                   "vix_percentile", "vix_zscore", "vix_classification",
                   "vix_close", "vix_change", "vix_range", "vix_signal"),
    "vix (FRED)": ("vixcls",),
    "filings": ("filing", "sec_"),
    "news/sentiment": ("news_", "sentiment_", "hype_", "keyword_", "entity_"),
    "fear/greed": ("fear_greed",),
}

BLOCK = 150


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=str(DEFAULT_BATCH))
    parser.add_argument(
        "--suffix", default="_1d",
        help=("Only count columns ending in this, '' for all. Defaults to the "
              "daily frame because comparing two batches WITHOUT it is "
              "misleading: the rebuild of 2026-09-08 gained 456 `_15m` columns "
              "the previous batch did not have at all, and doubled `_60m` from "
              "461 to 920. Every 15m column is post-seal by construction "
              "(#297), so all 456 land in `absent` and the family totals stop "
              "being comparable. `_1d` went 916 -> 918, which is a comparison."))
    args = parser.parse_args()
    batch = Path(args.batch)
    features = batch / "features.parquet"
    if not features.exists():
        print(f"no features.parquet under {batch}")
        return 1

    frame = pd.read_parquet(features, columns=["datetime", "interval"])
    is_daily = frame["interval"].astype(str).eq("1d").to_numpy()
    stamps = pd.to_datetime(frame.loc[is_daily, "datetime"]).reset_index(drop=True)
    seal = pd.Timestamp(SEAL_START).tz_localize(None)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    unsealed = (stamps < seal).to_numpy()
    mask = is_daily.copy()
    mask[mask] = unsealed
    print(f"batch: {batch}")
    print(f"daily rows {is_daily.sum():,}, of them explorable {mask.sum():,} "
          f"(seal {seal.date()})\n")
    del frame

    names = [n for n in pq.ParquetFile(features).schema_arrow.names
             if n not in {"datetime", "ticker", "interval"}
             and not n.startswith("target_")
             and n.endswith(args.suffix)]
    print(f"counting columns ending in {args.suffix!r}: {len(names):,} of "
          f"{len(pq.ParquetFile(features).schema_arrow.names):,}\n")
    wanted = {family: [n for n in names
                       if any(mark in n.lower() for mark in marks)]
              for family, marks in FAMILIES.items()}

    print(f"{'family':<18}{'columns':>9}{'alive here':>12}{'constant':>10}"
          f"{'absent':>9}")
    print("-" * 58)
    for family, columns in wanted.items():
        alive = constant = absent = 0
        for start in range(0, len(columns), BLOCK):
            chunk = columns[start:start + BLOCK]
            block = pd.read_parquet(features, columns=chunk)
            block = block.loc[mask]
            for name in chunk:
                # NOT `pd.to_numeric(errors='coerce')`, which is what this did
                # until 2026-09-10 and which made it lie. `volatility_regime_1d`
                # is a dictionary of strings -- normal/low/extreme/high, fully
                # populated on all 623,398 explorable rows -- and coercing it to
                # a number turned every value into NaN, so the script reported it
                # ABSENT. A diagnostic that cannot see a categorical column
                # silently classifies it as missing data, which is the exact
                # confusion R63 exists to remove.
                column = block[name]
                present = column.notna()
                if not present.any():
                    absent += 1
                elif column[present].nunique(dropna=True) <= 1:
                    constant += 1
                else:
                    alive += 1
            del block
        print(f"{family:<18}{len(columns):>9}{alive:>12}{constant:>10}"
              f"{absent:>9}")

    print()
    print("'alive here' means the column carries more than one value BEFORE "
          "the seal --\nthe only place any measurement happens. 'constant' is "
          "a filled default and\n'absent' never reached the explorable period "
          "at all (CLAIMS R63).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
