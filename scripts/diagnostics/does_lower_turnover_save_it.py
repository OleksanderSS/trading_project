"""R21's falsifier: does holding longer, or trading bigger, save the book?

R21 measured the one surviving feature as a daily cross-sectional book and it
lost 99.4% of capital: gross Sharpe 2.30, net -4.37, because round-trip
friction is 0.1094% a bar and a daily rebalance pays it 252 times a year --
27.6% against a gross return near 10.5%.

Two levers were named and neither measured:

    HOLDING PERIOD  friction is paid once per round trip, so holding N days
                    divides the annual cost by N -- IF the signal still says
                    anything N days out. Whether it does is the question; a
                    candlestick shape has no obvious reason to persist.

    ORDER SIZE      27.6% is mostly `min_fee_per_order: 0.35` against
                    `order_value: 10000`, which is 3.5bp a side before the
                    spread ever arrives. The config says "CHANGE THIS to
                    re-price" in as many words. At $100,000 the minimum stops
                    binding and only the per-share fee and the spread remain.

Both legs pay the friction here, which is the correction R21 exists to record:
the stored target already subtracts cost, so a short multiplied by -1 is
CREDITED it, and half a long/short book being credited is how a gross curve
gets read as a net one.

The sealed period is not touched.

    python scripts/diagnostics/does_lower_turnover_save_it.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
FEATURE = "CDL_UPPER_WICK_RATIO_1d"
from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: Imported, never restated. Eight diagnostics each kept their own copy of this
#: date until 2026-09-04. The policy in docs/SEALED_HOLDOUT.md says moving the
#: seal EARLIER is "always safe" -- with eight copies it would have been safe in
#: one file and silently ignored in the other seven, which is the duplication
#: family this codebase's defects come from.
SEALED = SEAL_START


def _panel(feature: str) -> pd.DataFrame:
    ident = ["ticker", "datetime", "interval"]
    frame = pd.read_parquet(BATCH / "features.parquet",
                            columns=ident + [feature, "close"])
    frame = frame[frame["interval"] == "1d"].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[frame["datetime"] < SEALED]
    frame = frame.dropna(subset=[feature, "close"])
    return frame.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def _sharpe(returns: np.ndarray, per_year: float) -> tuple[float, float]:
    """Annualised Sharpe of a per-period series, and the annualised return."""
    usable = returns[np.isfinite(returns)]
    if usable.size < 30 or usable.std() <= 0:
        return float("nan"), float("nan")
    return (float(usable.mean() / usable.std() * np.sqrt(per_year)),
            float(usable.mean() * per_year))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default=FEATURE)
    parser.add_argument("--holds", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--orders", type=float, nargs="+",
                        default=[10_000, 100_000])
    args = parser.parse_args()

    config = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8"))
    base_costs = config["targets"]["target_return_1d"]["params"]["transaction_costs"]

    frame = _panel(args.feature)
    print(f"{args.feature}: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].min().date()} -> {frame['datetime'].max().date()}\n")

    # Positions: cross-sectionally centred rank, long above the median.
    frame["position"] = np.sign(
        frame.groupby("datetime")[args.feature].rank(pct=True) - 0.5
    )

    # DOLLAR-NEUTRAL, OR THE ANSWER IS THE MARKET (CLAIMS R28).
    #
    # `sign(rank - 0.5)` on a column with heavy ties gives +1 to EVERYONE:
    # pandas ranks ties by their average, so a flag that is 98.9% one value
    # ranks just above 0.5 for every name and the "long/short book" is long
    # everything. Measured 2026-09-04: seven such columns scored net Sharpe
    # ~1.00 and the constant opponent -- own every name, same clock, same
    # friction -- scored 1.018.
    #
    # This script takes the feature as an ARGUMENT, so it is one `--feature`
    # away from reproducing that. Subtracting the per-date mean removes the
    # exposure and leaves a degenerate column with no position at all.
    frame["position"] = frame["position"] - frame.groupby("datetime")[
        "position"].transform("mean")
    exposure = float(frame["position"].mean())
    print(f"net exposure after centring: {exposure:+.4f} "
          "(0 = dollar-neutral; anything else is market "
          "beta in the curve)")
    print()

    header = (f"{'hold':>6}{'order $':>10}{'friction/rt':>13}{'cost/year':>11}"
              f"{'gross Sharpe':>14}{'NET Sharpe':>12}{'net ann.ret':>13}")
    print(header)
    print("-" * len(header))

    for hold in args.holds:
        # Forward N-day return, within each ticker. The position is taken on
        # the bar's own close and held N bars, so nothing here is knowable
        # before it is used.
        forward = (frame.groupby("ticker", sort=False)["close"]
                   .transform(lambda s: s.shift(-hold) / s - 1.0))
        gross = frame["position"].to_numpy() * forward.to_numpy()
        per_year = 252.0 / hold
        # One bar in N is an independent, non-overlapping holding period; using
        # every bar would count the same trade N times and shrink the standard
        # deviation by a factor nobody earned.
        #
        # And the Sharpe is the PORTFOLIO's, not a single position's. A book of
        # 110 roughly independent positions has about 1/sqrt(110) of the
        # volatility of one of them, so measuring per-row understates the
        # Sharpe by a factor of ten: the first version of this script reported
        # 0.252 where R21's portfolio curve said 2.30, and 0.252*sqrt(110) is
        # 2.64. The mismatch is what exposed the bug.
        work = frame[["datetime"]].copy()
        work["gross"] = gross
        by_date = work.groupby("datetime")["gross"].mean().sort_index()
        dates = by_date.index[::hold]
        g_sharpe, _ = _sharpe(by_date.loc[dates].to_numpy(), per_year)

        for order in args.orders:
            costs = dict(base_costs, order_value=order)
            friction = np.asarray(
                RegressionCalculator._round_trip_cost(frame["close"], costs),
                dtype=float,
            )
            net = gross - np.abs(frame["position"].to_numpy()) * friction
            work["net"] = net
            net_by_date = work.groupby("datetime")["net"].mean().sort_index()
            n_sharpe, n_return = _sharpe(
                net_by_date.loc[dates].to_numpy(), per_year)
            print(f"{hold:>6}{order:>10,.0f}{friction.mean():>13.4%}"
                  f"{friction.mean() * per_year:>11.1%}"
                  f"{g_sharpe:>14.3f}{n_sharpe:>12.3f}{n_return:>13.2%}")

    print("\nReading: gross is the same signal at every order size -- only the")
    print("net column moves. A row whose net Sharpe is positive is the first")
    print("real candidate this project has had, and the point at which the")
    print("sealed years become the right thing to spend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
