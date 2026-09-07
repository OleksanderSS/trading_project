"""Two ways to size a book, and which of them actually binds.

The owner asked whether the adequate drawdown and return can be computed
backwards from the Sharpe rather than chosen. Partly yes, and the part that can
is worth separating sharply from the part that cannot.

WHAT IS OBJECTIVE. For a book with Sharpe S run at volatility sigma, the
long-run growth rate of capital is

    g(sigma) = S*sigma - sigma^2 / 2

which has a maximum, and it is not at "as much as possible". It peaks at

    sigma* = S          giving  g* = S^2 / 2
    g(sigma) = 0 at sigma = 2S

That is the Kelly result, and it contains a hard fact that has nothing to do
with taste: PAST TWICE THE SHARPE IN VOLATILITY, MORE EXPOSURE LOSES MONEY WITH
CERTAINTY, however good the strategy. "As much as possible" is not merely
imprecise, it is wrong past a computable point.

WHAT IS NOT OBJECTIVE. Kelly maximises growth for someone indifferent to the
path. Nobody is: at full Kelly the chance of halving your capital at some point
is about one in two. So practitioners run a FRACTION -- half Kelly keeps 75% of
the growth for half the volatility, which is the whole argument for it.

AND THE POINT OF THIS SCRIPT. With fat tails and a long horizon, a survivable
drawdown budget turns out to bind FAR below even quarter Kelly, at every Sharpe
this project can plausibly reach. If that is so, the entire optimal-sizing
literature is inapplicable here and the only number that matters is the
drawdown budget. That is a claim worth testing rather than repeating, so both
are computed side by side.

Growth is computed analytically because for log returns the formula is exact.
Drawdown is bootstrapped from the panel's own daily returns in 21-day blocks,
because normal draws would understate it (measured excess kurtosis: 9.5).

    python scripts/diagnostics/what_exposure_is_optimal_and_what_survives.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "net_test", PROJECT_ROOT / "scripts/diagnostics/net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NET)

BLOCK = 21
YEARS = 27
PATHS = 2000
TOLERANCE = 0.05
BUDGET = 0.30

SHARPES = [
    ("SPY, the real market (R48)", 0.553),
    ("equal-weight survivors, measured", 0.928),
    ("what 20%/yr at a 30% budget needs (R57)", 1.700),
    ("Sharpe 2.0 (R48's 20% at 10% vol)", 2.000),
]


def _shape() -> np.ndarray:
    frame, _ = NET._panel([])
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    returns = (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s: s / s.shift(1) - 1.0).to_numpy())
    daily = np.nan_to_num(NET._mean_by_date(returns, codes, len(uniques)))
    return (daily - daily.mean()) / daily.std()


def _worst(shape, index, sharpe: float, vol: float) -> np.ndarray:
    """95th-percentile worst drawdown, in LOG space.

    Simple returns break at Kelly-scale volatility -- a 93% annual vol produces
    daily draws that take 1+r below zero, and the product then flips sign and
    reports nonsense. Compounding the log returns instead is both correct and
    the space the growth formula is written in.
    """
    logs = shape[index] * (vol / np.sqrt(252)) + (sharpe * vol - vol ** 2 / 2) / 252
    equity = np.exp(np.cumsum(logs, axis=1))
    peak = np.maximum.accumulate(equity, axis=1)
    return (equity / peak - 1.0).min(axis=1)


def _vol_for_budget(shape, index, sharpe: float, budget: float) -> float:
    low, high = 0.005, 3.0
    for _ in range(22):
        middle = (low + high) / 2
        if abs(np.percentile(_worst(shape, index, sharpe, middle),
                             100 * TOLERANCE)) > budget:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def main() -> int:
    shape = _shape()
    generator = np.random.default_rng(23)
    length = YEARS * 252
    starts = generator.integers(0, len(shape) - BLOCK,
                                size=(PATHS, length // BLOCK + 1))
    index = (starts[:, :, None] + np.arange(BLOCK)).reshape(PATHS, -1)[:, :length]

    print("GROWTH IS A CURVE WITH A PEAK, NOT A SLOPE\n")
    print(f"    {'book':<42}{'Sharpe':>7}{'best vol':>10}{'best growth':>13}"
          f"{'zero at':>10}")
    print("    " + "-" * 82)
    for name, sharpe in SHARPES:
        print(f"    {name:<42}{sharpe:>7.2f}{sharpe:>9.0%}"
              f"{sharpe ** 2 / 2:>12.1%}{2 * sharpe:>10.0%}")
    print("\n    Past twice the Sharpe in volatility the growth rate is NEGATIVE, "
          "whatever the\n    strategy. That is the part of 'as much as possible' "
          "that is simply wrong.\n")

    print("=" * 88)
    print("\nAND WHAT A SURVIVABLE DRAWDOWN BUDGET ALLOWS INSTEAD\n")
    print(f"    Budget: the drawdown {TOLERANCE:.0%} of healthy {YEARS}-year "
          f"histories reach, set at {BUDGET:.0%}.\n")
    header = (f"    {'book':<42}{'full Kelly':>11}{'1/4 Kelly':>11}"
              f"{'budget':>9}{'budget vs 1/4':>15}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name, sharpe in SHARPES:
        allowed = _vol_for_budget(shape, index, sharpe, BUDGET)
        quarter = 0.25 * sharpe
        print(f"    {name:<42}{sharpe:>10.0%}{quarter:>11.0%}"
              f"{allowed:>9.0%}{allowed / quarter:>14.2f}x")

    print("\n    Every row: the budget allows a FRACTION of even quarter Kelly. "
          "The growth-optimal\n    sizing literature does not bind here at any "
          "Sharpe this project can reach -- the\n    drawdown budget does, and "
          "by a wide margin. Optimal sizing is not the question.\n")

    print("=" * 88)
    print("\nSO THE ADEQUATE NUMBERS, COMPUTED BACKWARDS FROM THE SHARPE\n")
    print(f"    {'book':<42}{'vol':>7}{'return':>9}{'worst DD, 5%':>14}"
          f"{'median DD':>11}")
    print("    " + "-" * 82)
    for name, sharpe in SHARPES:
        allowed = _vol_for_budget(shape, index, sharpe, BUDGET)
        worst = _worst(shape, index, sharpe, allowed)
        growth = sharpe * allowed - allowed ** 2 / 2
        print(f"    {name:<42}{allowed:>7.1%}{growth:>9.1%}"
              f"{np.percentile(worst, 5):>13.1%}{np.median(worst):>11.1%}")

    print("\n    The return column is not a target. It is what the Sharpe and the "
          "budget leave.\n    Wanting a bigger number in it means wanting a "
          "bigger number in the Sharpe column,\n    and there is no other way in "
          "-- exposure is already capped by the budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
