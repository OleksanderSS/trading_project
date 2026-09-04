"""The one gap R24 left open: is the per-name edge itself larger on dearer names?

R24 measured the price lever at 1.36x against a gap of 1.51x and closed it --
but that arithmetic assumes the edge per name is the same whatever the share
costs. If the edge is systematically larger on dearer names, the lever is worth
more than 1.36x and the conclusion moves.

There is a reason to think it might go the OTHER way: dear shares are usually
the larger, more heavily arbitraged companies, where a candlestick shape has
had more eyes on it for longer.

ONE TEST, NOT FORTY-SIX. R23 exists because a threshold that ignores the number
of attempts is how this project fools itself, so this does not hunt for the
feature with the biggest gap. It asks a single aggregate question -- across all
46 survivors at once, is the mean |IC| higher on the dear half than the cheap
half -- and answers it with a paired test over the features. One number, one
threshold, no maximum taken anywhere.

The sealed period is untouched.

    python scripts/diagnostics/is_the_edge_bigger_on_dear_names.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: Imported, never restated. Eight diagnostics each kept their own copy of this
#: date until 2026-09-04. The policy in docs/SEALED_HOLDOUT.md says moving the
#: seal EARLIER is "always safe" -- with eight copies it would have been safe in
#: one file and silently ignored in the other seven, which is the duplication
#: family this codebase's defects come from.
SEALED = SEAL_START


def _daily_ic(values: pd.Series, outcome: pd.Series, dates: pd.Series) -> float:
    """Mean per-date rank correlation -- the same statistic 1.2 reports.

    Vectorised, because the obvious `groupby(...).apply(spearman)` is 6,800
    calls per feature per half and 625,000 in total: the first version of this
    script timed out at nine minutes without finishing one pass. Ranking inside
    each date and then correlating the ranks IS Spearman, and a correlation is
    three grouped sums.
    """
    frame = pd.DataFrame({"v": values, "y": outcome, "d": dates}).dropna()
    if len(frame) < 5_000:
        return float("nan")
    grouped = frame.groupby("d")
    rv = grouped["v"].rank()
    ry = grouped["y"].rank()
    rv = rv - grouped["v"].rank().groupby(frame["d"]).transform("mean")
    ry = ry - grouped["y"].rank().groupby(frame["d"]).transform("mean")
    work = pd.DataFrame({"d": frame["d"], "xy": rv * ry, "xx": rv * rv, "yy": ry * ry,
                         "n": 1})
    sums = work.groupby("d").sum()
    sums = sums[sums["n"] >= 10]
    denom = np.sqrt(sums["xx"] * sums["yy"])
    per_date = (sums["xy"] / denom).replace([np.inf, -np.inf], np.nan)
    return float(per_date.mean(skipna=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    roles = pd.read_csv(ROLES)
    names = roles[(roles["passes_fdr"]) & (roles["varies"] > 0.5)]["feature"].tolist()

    ident = ["ticker", "datetime", "interval"]
    wanted = list(dict.fromkeys(ident + ["close"] + names))
    frame = pd.read_parquet(BATCH / "features.parquet", columns=wanted)
    frame = frame[frame["interval"] == "1d"].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[frame["datetime"] < SEALED]

    targets = pd.read_parquet(BATCH / "targets.parquet",
                              columns=ident + ["target_return_1d"])
    targets = targets[targets["interval"] == "1d"]
    frame = frame.merge(targets[["ticker", "datetime", "target_return_1d"]],
                        on=["ticker", "datetime"], how="inner")
    frame = frame.dropna(subset=["close", "target_return_1d"])

    median_price = frame.groupby("ticker")["close"].median().sort_values()
    cheap = set(median_price.head(len(median_price) // 2).index)
    dear = set(median_price.tail(len(median_price) - len(median_price) // 2).index)
    is_dear = frame["ticker"].isin(dear)

    print(f"{len(dear)} dear names (median ${median_price[list(dear)].median():.1f}) "
          f"against {len(cheap)} cheap (${median_price[list(cheap)].median():.1f})")
    print(f"{len(names)} features, {len(frame):,} rows\n")

    rows = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        ic_dear = _daily_ic(values[is_dear], frame.loc[is_dear, "target_return_1d"],
                            frame.loc[is_dear, "datetime"])
        ic_cheap = _daily_ic(values[~is_dear], frame.loc[~is_dear, "target_return_1d"],
                             frame.loc[~is_dear, "datetime"])
        if np.isfinite(ic_dear) and np.isfinite(ic_cheap):
            rows.append({"feature": name, "ic_dear": ic_dear, "ic_cheap": ic_cheap,
                         "abs_dear": abs(ic_dear), "abs_cheap": abs(ic_cheap)})

    report = pd.DataFrame(rows)
    if report.empty:
        print("nothing measurable")
        return 1

    diff = report["abs_dear"] - report["abs_cheap"]
    stat, p = wilcoxon(report["abs_dear"], report["abs_cheap"])
    ratio = report["abs_dear"].mean() / report["abs_cheap"].mean()

    print(f"{'':<12}{'mean |IC|':>12}{'median |IC|':>14}")
    print(f"{'dear half':<12}{report['abs_dear'].mean():>12.5f}"
          f"{report['abs_dear'].median():>14.5f}")
    print(f"{'cheap half':<12}{report['abs_cheap'].mean():>12.5f}"
          f"{report['abs_cheap'].median():>14.5f}")
    print(f"\nratio dear/cheap                {ratio:.3f}x")
    print(f"features where dear is stronger  {int((diff > 0).sum())} of {len(report)}")
    print(f"paired Wilcoxon                  p = {p:.4f}")

    print("\n" + "=" * 64)
    # R24's lever is 1.36x against a gap of 1.51x, so the edge on dear names
    # would have to be about 1.11x stronger to close it -- that is the number
    # this test is against, not "is there any difference at all".
    needed = 1.51 / 1.36
    if p < 0.05 and ratio >= needed:
        print(f"The edge IS stronger on dear names by {ratio:.2f}x, which clears "
              f"the {needed:.2f}x\nthe lever still needs. R24's closure reopens.")
    elif p < 0.05 and ratio > 1.0:
        print(f"The edge is stronger on dear names, but by {ratio:.2f}x against "
              f"the {needed:.2f}x\nneeded. The lever moves and still does not "
              f"close the gap.")
    else:
        print(f"No systematic difference (ratio {ratio:.2f}x, p = {p:.3f}). "
              f"R24's assumption\nholds and the price lever stays closed at "
              f"1.36x against a gap of 1.51x.")
    print("=" * 64)
    out = PROJECT_ROOT / "diagnostic_reports" / "ic_by_price_half.csv"
    report.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
