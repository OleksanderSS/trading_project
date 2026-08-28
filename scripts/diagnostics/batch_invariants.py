"""Check a batch or a checkpoint against what must be true of it.

Every check here is a defect this project actually shipped, and every one of
them was found late -- after eight, eleven, or twelve hours of enrichment --
although each is visible in the finished data within seconds.

The first checkpoint lands about fifty minutes into a rebuild. Run this
against it and a bad run ends at minute fifty instead of hour twelve.

    python scripts/diagnostics/batch_invariants.py
    python scripts/diagnostics/batch_invariants.py data/checkpoints/enriched/enriched_1d.parquet
    python scripts/diagnostics/batch_invariants.py --interval 1d

Exit code is non-zero when any check fails, so it can gate a run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT = Path("data/colab/accumulated/main_database/features.parquet")
TRAIN_FRACTION = 0.70
BLOCK = 150


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    story: str


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    have = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = [c for c in columns if c in have]
    return pd.read_parquet(path, columns=wanted) if wanted else pd.DataFrame()


def check_market_wide_series_agree(path: Path, frame: pd.DataFrame) -> Result:
    """A macro series is one number for the whole economy on a date.

    On 2026-08-27 `FRED_INDPRO_1d` held two to four values per date on 98% of
    dates, and on one date the split was exactly the 22 tickers of the old
    preset against the 88 added later. The column was not measuring industrial
    production; it was labelling which cohort a ticker belonged to -- and it
    then led the leading-feature report, because a ranking measures precisely
    that variation.
    """
    names = [c for c in pq.ParquetFile(path).schema_arrow.names
             if c.startswith("FRED_")][:20]
    if not names or "ticker" not in frame.columns:
        return Result("macro agrees across tickers", True, "no macro columns", "")

    data = _read(path, names)
    data = data.loc[frame.index]
    data["_d"] = frame["_date"].to_numpy()
    offenders = []
    for column in names:
        subset = data[["_d", column]].dropna()
        if subset.empty:
            continue
        grouped = subset.groupby("_d")[column]
        sizes, uniques = grouped.size(), grouped.nunique()
        multi = sizes > 1
        if multi.any() and (uniques[multi] > 1).mean() > 0.01:
            offenders.append(f"{column} {(uniques[multi] > 1).mean() * 100:.0f}%")
    return Result(
        "macro agrees across tickers", not offenders,
        f"{len(offenders)} of {len(names)} differ between names on one date"
        + (f": {', '.join(offenders[:4])}" if offenders else ""),
        check_market_wide_series_agree.__doc__ or "",
    )


def check_nothing_is_mostly_its_median(path: Path, frame: pd.DataFrame) -> Result:
    """A fabricated fill leaves the column's own median everywhere.

    70.5% of every `FRED_INDPRO_1d` value and 70.6% of `FRED_VIXCLS_1d` was
    exactly the column median -- a constant computed over the whole frame, so
    it contained the future. Seven of every ten macro readings were not data.

    A column with many distinct values that nonetheless piles 25% of them on
    one point is the signature. A flag or a calendar field is not: those are
    legitimately constant and are skipped by the distinct-value floor.
    """
    names = [c for c in pq.ParquetFile(path).schema_arrow.names
             if c not in ("datetime", "ticker", "interval", "hash", "timestamp", "date")]
    offenders = []
    for start in range(0, len(names), BLOCK):
        block = _read(path, names[start:start + BLOCK])
        if block.empty:
            continue
        block = block.loc[frame.index]
        for column in block.columns:
            series = block[column]
            if series.dtype.kind not in "fiu":
                continue
            series = series.dropna()
            if len(series) < 5_000 or series.nunique() < 100:
                continue
            counts = series.value_counts()
            share = counts.iloc[0] / len(series)
            if share > 0.25 and abs(float(counts.index[0]) - float(series.median())) < 1e-9:
                offenders.append(f"{column} {share * 100:.0f}%")
        del block
    return Result(
        "no column is mostly its own median", not offenders,
        f"{len(offenders)} column(s)"
        + (f": {', '.join(offenders[:4])}" if offenders else ""),
        check_nothing_is_mostly_its_median.__doc__ or "",
    )


def check_rows_are_time_ordered(path: Path, frame: pd.DataFrame) -> Result:
    """`ffill`, `rolling` and `shift` walk rows, not dates.

    On 2026-08-28 `FRED_CPIAUCSL_1d` held 313.569 on every row from 1996 to
    2023 -- the 2024 level -- because the frame was ordered newest-first and a
    forward fill therefore ran thirty years into the past. There is no `bfill`
    anywhere in the codebase; the lookahead came from row order alone.
    """
    if "ticker" not in frame.columns:
        ordered = frame["_time"].is_monotonic_increasing
        return Result("rows are time-ordered", ordered,
                      "single series" + ("" if ordered else " OUT OF ORDER"),
                      check_rows_are_time_ordered.__doc__ or "")
    bad = [
        name for name, group in frame.groupby("ticker", sort=False)
        if not group["_time"].is_monotonic_increasing
    ]
    return Result(
        "rows are time-ordered", not bad,
        f"{len(bad)} ticker(s) out of order"
        + (f": {', '.join(map(str, bad[:5]))}" if bad else ""),
        check_rows_are_time_ordered.__doc__ or "",
    )


def check_context_state_agrees(path: Path, frame: pd.DataFrame) -> Result:
    """A market-wide regime must be the same for every name on a date.

    `state_champion_1d` held +1 on 659,509 rows against -1 on 39,194, and 16 of
    110 tickers carried +1 on every row of their history -- because the state
    was spread by row order rather than joined by date. Only the champion
    itself had a believable value.
    """
    names = [c for c in pq.ParquetFile(path).schema_arrow.names
             if "champion" in c.lower() or c.startswith("state_FRED_")][:12]
    if not names or "ticker" not in frame.columns:
        return Result("context state agrees across tickers", True, "none present", "")
    data = _read(path, names).loc[frame.index]
    data["_d"] = frame["_date"].to_numpy()
    offenders = []
    for column in names:
        subset = data[["_d", column]].dropna()
        if subset.empty:
            continue
        grouped = subset.groupby("_d")[column]
        sizes, uniques = grouped.size(), grouped.nunique()
        multi = sizes > 1
        if multi.any() and (uniques[multi] > 1).mean() > 0.01:
            offenders.append(column)
    return Result(
        "context state agrees across tickers", not offenders,
        f"{len(offenders)} differ between names on one date"
        + (f": {', '.join(offenders[:4])}" if offenders else ""),
        check_context_state_agrees.__doc__ or "",
    )


def check_indicators_match_recomputation(path: Path, frame: pd.DataFrame) -> Result:
    """A 20-day moving average is the mean of twenty closes. Recompute it.

    This is the cheapest check in the file and the one nobody ran for months.
    It answers whether the whole price block means what its names say. On
    2026-08-28 it matched on 99.4% of rows -- so the absence of signal there is
    real rather than an artefact of scrambled rows -- while exposing that the
    first 47 to 54 rows of every ticker do not match, which remains open.
    """
    names = pq.ParquetFile(path).schema_arrow.names
    sma = next((c for c in names if c.startswith("SMA_20")), None)
    if sma is None or "close" not in names or "ticker" not in frame.columns:
        return Result("indicators match a recomputation", True, "no SMA_20", "")

    data = _read(path, ["close", sma]).loc[frame.index]
    data["ticker"] = frame["ticker"].to_numpy()
    data["_t"] = frame["_time"].to_numpy()

    matched = total = 0
    for _, group in list(data.groupby("ticker", sort=False))[:8]:
        group = group.sort_values("_t")
        recomputed = group["close"].rolling(20, min_periods=1).mean()
        agree = np.isclose(recomputed, group[sma], rtol=1e-3, equal_nan=True)
        matched += int(agree.sum())
        total += len(agree)
    share = matched / total if total else 1.0
    return Result(
        "indicators match a recomputation", share > 0.98,
        f"{share * 100:.1f}% of {total:,} rows on 8 tickers",
        check_indicators_match_recomputation.__doc__ or "",
    )


def check_features_learnable(path: Path, frame: pd.DataFrame) -> Result:
    """A feature constant while a model learns cannot be learned.

    65 features were constant across the training window and only started
    varying after it: news exists for 2026 alone, and the macro columns were
    fed a two-year fetch while the accumulated table sat unused. A model gives
    no weight to something that never moved, and then it moves in the holdout.
    """
    split = frame["_time"].quantile(TRAIN_FRACTION)
    in_train = (frame["_time"] <= split).to_numpy()
    if not in_train.any() or in_train.all():
        return Result("features vary during training", True, "no split", "")

    names = [c for c in pq.ParquetFile(path).schema_arrow.names
             if c not in ("datetime", "ticker", "interval", "hash", "timestamp", "date")]
    dead = []
    for start in range(0, len(names), BLOCK):
        block = _read(path, names[start:start + BLOCK])
        if block.empty:
            continue
        block = block.loc[frame.index]
        for column in block.columns:
            if block[column].dtype.kind not in "fiu":
                continue
            values = block[column].to_numpy(dtype=float, na_value=np.nan)
            train = values[in_train]
            train = train[~np.isnan(train)]
            if len(np.unique(train)) > 1:
                continue
            rest = values[~in_train]
            rest = rest[~np.isnan(rest)]
            if len(np.unique(rest)) > 1:
                dead.append(column)
        del block
    return Result(
        "features vary during training", not dead,
        f"{len(dead)} constant in training, alive after"
        + (f": {', '.join(dead[:4])}" if dead else ""),
        check_features_learnable.__doc__ or "",
    )


CHECKS = (
    check_rows_are_time_ordered,
    check_market_wide_series_agree,
    check_context_state_agrees,
    check_nothing_is_mostly_its_median,
    check_indicators_match_recomputation,
    check_features_learnable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT))
    parser.add_argument("--interval", default=None,
                        help="restrict to one timeframe, e.g. 1d")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per check, no explanations")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}")
        return 2

    frame = _read(path, ["datetime", "ticker", "interval"])
    if frame.empty or "datetime" not in frame.columns:
        print(f"{path} has no datetime column; nothing to check.")
        return 2
    if args.interval and "interval" in frame.columns:
        frame = frame.loc[frame["interval"].astype(str) == args.interval]
    frame = frame.copy()
    frame["_time"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["_date"] = frame["_time"].dt.date

    rows = pq.ParquetFile(path).metadata.num_rows
    print(f"{path.name}: {rows:,} rows, checking {len(frame):,}"
          + (f" ({args.interval})" if args.interval else "") + "\n")

    results = [check(path, frame) for check in CHECKS]
    for result in results:
        mark = "OK  " if result.ok else "FAIL"
        print(f"  [{mark}] {result.name:38s} {result.detail}")
        if not result.ok and not args.quiet and result.story:
            first = " ".join(result.story.split("\n\n")[0].split())
            print(f"         why it matters: {first}")

    failed = [r for r in results if not r.ok]
    print(f"\n{len(failed)} failing check(s) of {len(results)}.")
    if failed:
        print("A rebuild started now would carry these into the batch.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
