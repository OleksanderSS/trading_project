"""Is this feature leading, is it usable, and would it survive being traded?

Run three times by hand on 2026-08-23 and it caught three different lies, so it
is a tool now rather than a scratch script. Each column below exists because
something passed the other checks and failed this one.

    varies        Does the feature differ BETWEEN names on the same date? A
                  market-wide series does not: `cftc_*` has cross-sectional
                  variation on 0.0% of dates, so in a ranking it contributes
                  exactly zero however good its correlation looks. Such a
                  series is not useless -- it enters as an interaction with a
                  per-name sensitivity, never as a column.

    ic_out        Rank correlation with the target on a HOLDOUT split by time.
                  Chosen on the train slice, measured after it. In-sample IC
                  measures memory, not skill.

    kept_sign     Does the direction survive the split? Under the null this is
                  a coin flip. It is the cheapest check that separates a
                  relationship from a coincidence.

    ic_within     The same correlation after each ticker's own mean is removed.
                  This is the one that matters most and is easiest to skip. A
                  cross-sectional book built without it held NVDA on 100% of
                  days against SPY on 100% of days and "returned" 34.9% a year
                  -- which is not a signal but one bet, discovered after the
                  fact. If ic_within collapses toward zero, the feature is not
                  predicting WHEN, it is labelling WHICH NAME.

    history       How many rows the feature actually has. Attention was
                  collected 30 days deep against a frame spanning decades: the
                  column existed, reported success, and could never be trained
                  on. Depth is checked before anything else is believed.

Nothing here says a feature is profitable. It says whether a feature is worth
the cost of testing properly.

    python scripts/diagnostics/leading_feature_report.py
    python scripts/diagnostics/leading_feature_report.py --target target_return_5d --top 40
    python scripts/diagnostics/leading_feature_report.py --features wiki_views filing_count_30d_1d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

FEATURES = Path("data/colab/accumulated/main_database/features.parquet")
TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")

DEFAULT_TARGET = "target_relative_return_5d"
TRAIN_FRACTION = 0.70
BLOCK = 250

#: Below this a correlation is being read off too few rows to mean anything.
MIN_ROWS = 500

#: A feature covering less of the frame than this cannot be trained on, whatever
#: its correlation says. Attention sat at 0.4% before 2026-08-23.
THIN_COVERAGE = 0.05


def _daily_targets(target: str) -> pd.DataFrame:
    frame = pd.read_parquet(TARGETS, columns=["datetime", "ticker", "interval", target])
    frame = frame[frame["interval"].astype(str).eq("1d")].reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    return frame


def _safe_ic(values: np.ndarray, outcome: np.ndarray, mask: np.ndarray) -> float:
    usable = mask & np.isfinite(values) & np.isfinite(outcome)
    if usable.sum() < MIN_ROWS:
        return float("nan")
    if np.nanstd(values[usable]) == 0 or np.nanstd(outcome[usable]) == 0:
        return float("nan")
    return float(spearmanr(values[usable], outcome[usable]).statistic)


def _demean(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return (series - series.groupby(pd.Series(keys)).transform("mean")).to_numpy()


def _examine(name: str, values: np.ndarray, book: dict) -> dict:
    outcome, is_train = book["outcome"], book["is_train"]
    finite = np.isfinite(values)
    coverage = finite.mean()

    per_date_spread = pd.Series(values).groupby(book["dates"]).std()
    varies = float((per_date_spread.fillna(0) > 1e-12).mean())

    ic_in = _safe_ic(values, outcome, is_train)
    ic_out = _safe_ic(values, outcome, ~is_train)
    within = _safe_ic(_demean(values, book["tickers"]),
                      book["outcome_demeaned"], ~is_train)

    return {
        "feature": name,
        "history": int(finite.sum()),
        "coverage": coverage,
        "varies": varies,
        "ic_in": ic_in,
        "ic_out": ic_out,
        "kept_sign": bool(np.isfinite(ic_in) and np.isfinite(ic_out)
                          and np.sign(ic_in) == np.sign(ic_out)),
        "ic_within": within,
    }


def _verdict(row: pd.Series) -> str:
    if row["coverage"] < THIN_COVERAGE:
        return "too thin to judge"
    if row["varies"] < 0.05:
        return "market-wide: use as interaction"
    if not np.isfinite(row["ic_out"]):
        return "not measurable"
    if not row["kept_sign"]:
        return "sign flipped out of sample"
    if abs(row["ic_within"]) < abs(row["ic_out"]) * 0.34:
        return "labels the name, not the moment"
    if abs(row["ic_out"]) < 0.01:
        return "survives, but tiny"
    return "survives, worth testing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--features", nargs="*", default=None,
                        help="specific columns; default is every usable one")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    if not FEATURES.exists() or not TARGETS.exists():
        print("No batch on disk; run --mode prepare first.")
        return 1

    targets = _daily_targets(args.target)
    if targets[args.target].notna().sum() < MIN_ROWS:
        print(f"{args.target} has almost no values on the daily frame.")
        return 1

    cut = targets["datetime"].quantile(TRAIN_FRACTION)
    outcome = targets[args.target].to_numpy(dtype=float)
    book = {
        "outcome": outcome,
        "outcome_demeaned": _demean(outcome, targets["ticker"].to_numpy()),
        "is_train": (targets["datetime"] <= cut).to_numpy(),
        "dates": targets["datetime"],
        "tickers": targets["ticker"].to_numpy(),
    }
    print(f"target {args.target} | daily rows {len(targets):,} | split {cut.date()} "
          f"| {targets['ticker'].nunique()} names\n")

    mask = pd.read_parquet(FEATURES, columns=["interval"])
    mask = mask["interval"].astype(str).eq("1d").to_numpy()

    wanted = args.features or [
        name for name in pq.ParquetFile(FEATURES).schema_arrow.names
        if name not in {"datetime", "ticker", "interval"}
        and not name.startswith("target_")
    ]

    rows = []
    for start in range(0, len(wanted), BLOCK):
        columns = wanted[start:start + BLOCK]
        try:
            block = pd.read_parquet(FEATURES, columns=columns)
        except (OSError, ValueError):
            continue
        block = block.loc[mask].reset_index(drop=True)
        for name in columns:
            if name not in block.columns:
                continue
            values = pd.to_numeric(block[name], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).sum() == 0:
                continue
            rows.append(_examine(name, values, book))
        del block

    if not rows:
        print("None of the requested features exist on the daily frame.")
        return 1

    report = pd.DataFrame(rows)
    report["verdict"] = report.apply(_verdict, axis=1)
    report = report.reindex(
        report["ic_out"].abs().sort_values(ascending=False, na_position="last").index
    )

    print(f"{'feature':38s} {'hist':>7s} {'cov':>6s} {'varies':>7s} "
          f"{'ic_out':>8s} {'sign':>5s} {'within':>8s}  verdict")
    print("-" * 108)
    for _, row in report.head(args.top).iterrows():
        ic_out = "     —  " if pd.isna(row["ic_out"]) else f"{row['ic_out']:+8.4f}"
        within = "     —  " if pd.isna(row["ic_within"]) else f"{row['ic_within']:+8.4f}"
        print(f"{row['feature'][:38]:38s} {row['history']:7,} {row['coverage']:5.0%} "
              f"{row['varies']:6.0%} {ic_out} {'yes' if row['kept_sign'] else 'no':>5s} "
              f"{within}  {row['verdict']}")

    print()
    print("=== how many features fall into each verdict ===")
    for verdict, count in report["verdict"].value_counts().items():
        print(f"  {verdict:34s} {count:5d}")
    print()
    survivors = report[report["verdict"] == "survives, worth testing"]
    print(f"{len(survivors)} of {len(report)} are worth the cost of testing properly.")
    print("None of this says a feature is profitable. It says which ones are")
    print("not already disqualified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
