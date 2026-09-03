"""Which conditions precede a move, and does that mean anything.

Reverse analysis -- taking the moves that happened and asking what the state
looked like beforehand -- answers P(condition | move). Acting on it needs
P(move | condition), and the two are not the same. A condition present before
80% of large moves is worthless if it is also present on 78% of all bars.

So this reports both, always against the base rate, and never without the
number of rows behind it:

    n         how many bars carry the condition. A rate without this is not
              a result: 60% is two standard errors on a hundred cases and
              eleven on three thousand.
    p         P(move | condition) on the training slice
    base      P(move) over everything -- the number to beat
    lift      p / base. 1.0 means the condition says nothing
    z         (p - base) / standard error of p
    p_out     the same rate on the held-out slice, computed after the fact

Conditions are the single-bar `context_fingerprint`, which takes about 2,000
values on the daily frame -- 348 rows each, 166 of them above a thousand -- not
`context_pattern_id`, whose sequence hash is unique per row.

**Screening many conditions is the danger, not the method.** The count of
conditions tested is printed, a Benjamini-Hochberg threshold is applied, and
nothing is called a finding unless the holdout agrees. A condition chosen
because it looked good is a fitted condition.

    python scripts/diagnostics/conditional_pattern_report.py
    python scripts/diagnostics/conditional_pattern_report.py --horizon 5 --quantile 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# The diagnostics run as scripts, not as part of the package, so the
# project root has to be on the path before src/ can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pipeline.sealed_period import SEAL_START, describe

DEFAULT = Path("data/colab/accumulated/main_database/features.parquet")
TRAIN_FRACTION = 0.70
MIN_ROWS = 200


def _pick_columns(path: Path, interval: str,
                  condition: str | None = None) -> tuple[str, str]:
    """The condition and close columns, whatever suffix the timeframe added.

    The default condition is `context_fingerprint`, and six of its eight
    drivers are transformations of price -- SMA_200, SMA_20, ATR_14,
    VOLATILITY_20, obv, RSI_14 -- with the two non-price ones being exactly
    the sentiment and macro composites measured as broken on 2026-08-28. So
    conditioning on it asks what price states imply, and what came back looked
    like a risk premium: the surviving conditions carried twice the spread.

    Any column can be named instead. The states worth trying are the ones from
    the leading class, and they have the coverage for it on the daily frame:
    `state_fund_price_to_book` is non-zero on 10.7% of rows -- about 75,000 --
    and the `state_cftc_*` family on 2.2 to 3.4%, around 9,500 rows per side.
    That is a measurable bucket, which is why this is worth pointing somewhere
    other than at price.
    """
    # Pick the column belonging to the interval being examined.
    #
    # The union lists 15m columns first, so `startswith("context_fingerprint")`
    # returned `context_fingerprint_15m` for every request. Sliced to 1d that
    # column is null by construction, the dropna emptied the frame, and the
    # report announced "0 rows" -- on 2026-08-29 it printed
    # "0 rows withheld; 0 remain" against a batch of 705,274 daily bars.
    #
    # Third file in this directory with the identical defect: the SMA_20 pick
    # and both agreement checks in batch_invariants.py had it too, and each
    # was fixed on its own without looking next door.
    names = pq.ParquetFile(path).schema_arrow.names
    suffix = f"_{interval}" if interval else ""

    def _prefer(candidates: list[str]) -> str | None:
        if suffix:
            exact = [c for c in candidates if c.endswith(suffix)]
            if exact:
                return exact[0]
        return candidates[0] if candidates else None

    if condition:
        chosen = _prefer([c for c in names
                          if c == condition or c.startswith(condition + "_")])
        if chosen is None:
            raise SystemExit(f"no column matching '{condition}' in {path.name}")
    else:
        chosen = _prefer([c for c in names
                          if c.startswith("context_fingerprint")])
    close = _prefer([c for c in names if c == "close" or c.startswith("close_")])
    return chosen, close


def load(path: Path, interval: str | None,
         condition: str | None = None) -> pd.DataFrame:
    fingerprint, close = _pick_columns(path, interval or "", condition)
    if fingerprint is None or close is None:
        raise SystemExit(
            f"{path} has no context_fingerprint or close column; nothing to do."
        )
    wanted = ["datetime", "ticker", "interval", fingerprint, close]
    have = set(pq.ParquetFile(path).schema_arrow.names)
    frame = pd.read_parquet(path, columns=[c for c in wanted if c in have])
    if interval and "interval" in frame.columns:
        frame = frame.loc[frame["interval"].astype(str) == interval]
    frame = frame.rename(columns={fingerprint: "condition", close: "close"})
    frame["_time"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame.dropna(subset=["_time", "condition", "close"])


def add_outcome(frame: pd.DataFrame, horizon: int, quantile: float) -> pd.DataFrame:
    """A move is a forward return above the quantile, measured within a ticker.

    Forward, not backward: the whole question is whether the condition precedes
    the move. And within a ticker, because a return computed across a boundary
    between two names is not a return.
    """
    frame = frame.sort_values(["ticker", "_time"], kind="mergesort")
    grouped = frame.groupby("ticker", sort=False)["close"]
    forward = grouped.shift(-horizon) / grouped.transform(lambda s: s) - 1.0
    frame["_forward"] = forward.to_numpy()
    frame = frame.dropna(subset=["_forward"])
    cut = frame["_forward"].quantile(quantile)
    frame["_move"] = (frame["_forward"] >= cut).astype(int)
    return frame


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> float:
    """The largest p-value that still passes, or 0 when none does."""
    if pvalues.size == 0:
        return 0.0
    ordered = np.sort(pvalues)
    ranks = np.arange(1, ordered.size + 1)
    passing = ordered <= alpha * ranks / ordered.size
    return float(ordered[passing].max()) if passing.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon", type=int, default=5,
                        help="bars ahead the move is measured over")
    parser.add_argument("--quantile", type=float, default=0.90,
                        help="a move is a forward return at or above this quantile")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--condition", default=None,
                        help="column to condition on; default context_fingerprint")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}")
        return 2

    frame = add_outcome(
        load(path, args.interval, args.condition), args.horizon, args.quantile
    )
    # The sealed stretch is not read here either.
    before = len(frame)
    frame = frame.loc[frame["_time"] < SEAL_START]
    print(describe())
    print(f"  {before - len(frame):,} rows withheld; {len(frame):,} remain.")
    print()

    split = frame["_time"].quantile(TRAIN_FRACTION)
    train = frame.loc[frame["_time"] <= split]
    holdout = frame.loc[frame["_time"] > split]

    base = float(train["_move"].mean())
    print(f"condition: {args.condition or 'context_fingerprint'}")
    print(f"{path.name} [{args.interval}]  horizon {args.horizon} bars, "
          f"move = top {(1 - args.quantile) * 100:.0f}%")
    print(f"train {len(train):,} rows to {split:%Y-%m-%d}, "
          f"holdout {len(holdout):,}")
    print(f"base rate P(move) = {base * 100:.2f}%\n")

    # Excess return over the base, not the probability of an extreme move.
    #
    # The first version of this report ranked conditions by P(forward return in
    # the top decile). Measured on 2026-08-28, that quantity correlates +0.889
    # with the dispersion of returns inside the condition: it was finding
    # volatility, not direction, and 205 of 368 conditions "survived" because
    # falling markets move more in both directions. A second attempt used the
    # mean return's t-statistic against zero, which is dominated by drift --
    # the base five-day return is +0.326%, so almost any condition looks
    # positive.
    #
    # What a condition has to beat is the base rate, so that is what is
    # measured: mean return inside the condition minus the mean everywhere,
    # over the standard error of that difference.
    base_return = float(train["_forward"].mean())
    rows = []
    for condition, group in train.groupby("condition", sort=False):
        n = len(group)
        if n < args.min_rows:
            continue
        mean = float(group["_forward"].mean())
        spread = float(group["_forward"].std())
        se = spread / np.sqrt(n) if spread > 0 else np.nan
        z = (mean - base_return) / se if se and not np.isnan(se) else 0.0
        after = holdout.loc[holdout["condition"] == condition, "_forward"]
        rows.append({
            "condition": str(condition)[:18],
            "n": n,
            "p": mean,
            "lift": spread,
            "z": z,
            "n_out": len(after),
            "p_out": float(after.mean()) if len(after) else np.nan,
        })

    if not rows:
        print(f"no condition reaches {args.min_rows} rows; nothing to report.")
        return 0

    report = pd.DataFrame(rows)
    from scipy.stats import norm
    report["pvalue"] = 2 * (1 - norm.cdf(report["z"].abs()))
    threshold = benjamini_hochberg(report["pvalue"].to_numpy())
    report = report.sort_values("z", key=lambda s: s.abs(), ascending=False)

    print(f"{'condition':20s} {'n':>7} {'mean':>8} {'spread':>7} {'z':>7} "
          f"{'n_out':>7} {'out':>8}  verdict")
    print("-" * 88)
    for _, row in report.head(args.top).iterrows():
        survives = (
            row["pvalue"] <= threshold
            and row["n_out"] >= args.min_rows // 4
            and np.sign(row["p_out"] - base_return) == np.sign(row["p"] - base_return)
        )
        verdict = "holds out of sample" if survives else (
            "fails the correction" if row["pvalue"] > threshold
            else "does not repeat"
        )
        print(f"{row['condition']:20s} {row['n']:7,} {row['p'] * 100:+7.3f}% "
              f"{row['lift'] * 100:6.2f}% {row['z']:7.2f} {row['n_out']:7,} "
              f"{row['p_out'] * 100:+7.3f}%  {verdict}")

    survivors = sum(
        1 for _, r in report.iterrows()
        if r["pvalue"] <= threshold and r["n_out"] >= args.min_rows // 4
        and np.sign(r["p_out"] - base_return) == np.sign(r["p"] - base_return)
    )
    print(f"\n{len(report):,} conditions screened at {args.min_rows}+ rows.")
    print(f"Benjamini-Hochberg threshold at 5%: p <= {threshold:.2e}")
    print(f"{survivors} survive the correction AND repeat out of sample.")
    print(
        "\nA condition that survives is worth testing properly. It is not a\n"
        "strategy: nothing here charges commission, and none of these\n"
        "conditions was chosen for a reason before being measured."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
