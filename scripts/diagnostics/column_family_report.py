"""What each family of columns is worth, against what it costs.

The map has carried a line saying 1,494 `ctx_`/`state_` columns "give nothing"
since an earlier session. I could not find the measurement behind it, and the
largest consumer of stage 3 -- RedundancyDetector, 88 to 158 minutes --
computes variance-inflation factors across every one of them. Dropping two
thirds of the feature space on a remembered claim is exactly the move this
project keeps having to undo.

So: measure it. For each family this reports, per column,

    usable      not all-NaN, and not constant among the rows that exist
    coverage    share of rows carrying a value
    |IC|        absolute Spearman correlation with the target, on aligned rows

and then how the family's usable columns distribute against the price columns
they would be replacing. A family whose best column ranks below the median
price column is not carrying the stage.

Nothing here is a decision. It is the number the decision needs.

    python scripts/diagnostics/column_family_report.py
    python scripts/diagnostics/column_family_report.py --target target_return_5d
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURES = Path("data/colab/accumulated/main_database/features.parquet")
TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")

#: Prefixes in the order they are tested; the first match wins, so the
#: narrower ones come first.
FAMILIES = ("ctx_", "state_", "peer_", "news_", "macro_", "sent")

KEYS = ("ticker", "datetime", "interval")


def family_of(column: str) -> str:
    for prefix in FAMILIES:
        if column.startswith(prefix):
            return prefix.rstrip("_") or prefix
    return "price/other"


def spearman_ic(feature: pd.Series, target: pd.Series) -> float:
    """Rank correlation on the rows where both sides exist."""
    both = feature.notna() & target.notna()
    if both.sum() < 200:
        return np.nan
    left = feature[both]
    if left.nunique() < 2:
        return np.nan
    return abs(left.rank().corr(target[both].rank()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None,
                        help="target column; default: the first return target found")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--sample", type=int, default=60_000,
                        help="rows to sample; 0 reads everything")
    args = parser.parse_args()

    for path in (FEATURES, TARGETS):
        if not path.exists():
            print(f"missing {path}")
            return 1

    target_names = [c for c in pq.ParquetFile(TARGETS).schema_arrow.names
                    if c.startswith("target_")]
    target = args.target or next(
        (c for c in target_names if "return" in c and c.endswith(("_1d", "_5d"))),
        target_names[0] if target_names else None,
    )
    if target is None:
        print("no target columns in the batch")
        return 1
    print(f"target: {target}   interval: {args.interval}")

    tgt = pd.read_parquet(TARGETS, columns=[*KEYS, target])
    tgt = tgt[tgt["interval"] == args.interval]
    if tgt.empty:
        print(f"no target rows at interval {args.interval}")
        return 1

    feature_columns = [c for c in pq.ParquetFile(FEATURES).schema_arrow.names
                       if c not in KEYS]
    print(f"{len(feature_columns)} feature columns, {len(tgt)} target rows\n")

    frame = pd.read_parquet(FEATURES)
    frame = frame[frame["interval"] == args.interval] if "interval" in frame else frame
    joined = frame.merge(tgt, on=[k for k in KEYS if k in frame.columns], how="inner")
    if joined.empty:
        print("features and targets share no rows at this interval")
        return 1
    if args.sample and len(joined) > args.sample:
        joined = joined.sample(args.sample, random_state=0)
    print(f"{len(joined)} rows aligned\n")

    y = pd.to_numeric(joined[target], errors="coerce")
    rows = []
    for column in feature_columns:
        if column not in joined.columns:
            continue
        values = pd.to_numeric(joined[column], errors="coerce")
        present = int(values.notna().sum())
        rows.append({
            "column": column,
            "family": family_of(column),
            "coverage": present / len(joined),
            "usable": present > 0 and values.nunique(dropna=True) > 1,
            "ic": spearman_ic(values, y),
        })

    report = pd.DataFrame(rows)
    price_median = report.loc[report["family"] == "price/other", "ic"].median()

    print(f"{'family':12s} {'cols':>5s} {'usable':>7s} {'cov':>6s} "
          f"{'IC med':>7s} {'IC p90':>7s} {'IC max':>7s} {'>price med':>11s}")
    print("-" * 72)
    for name, group in report.groupby("family"):
        good = group[group["usable"]]
        ic = good["ic"].dropna()
        beats = int((ic > price_median).sum()) if price_median == price_median else 0
        print(f"{name:12s} {len(group):5d} {len(good):7d} "
              f"{group['coverage'].mean():6.1%} "
              f"{ic.median():7.4f} {ic.quantile(0.9):7.4f} {ic.max():7.4f} "
              f"{beats:11d}")

    print(f"\nmedian |IC| among price/other columns: {price_median:.4f}")
    ballast = report[report["family"].isin(("ctx", "state"))]
    carrying = ballast[(ballast["ic"] > price_median) & ballast["usable"]]
    print(f"ctx_ + state_: {len(ballast)} columns, {len(carrying)} above that median")
    if len(ballast):
        print(f"  best: {carrying.nlargest(min(5, len(carrying)), 'ic')[['column', 'ic']].to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
