"""Return, drawdown and Sharpe are not three choices. They are two plus a consequence.

The owner asked what to anchor a return target on, having said 20% was a
number off the top of his head, and that "as much as possible" is not a valid
answer. It is not, and there is a valid one.

    return = Sharpe x volatility
    drawdown is a function of volatility and horizon

Volatility is the knob -- exposure, leverage, investedness are all the same
knob wearing three names. Sharpe is the one thing NOT chosen: it is what the
data gives. So a drawdown budget and a Sharpe together DETERMINE the return
available. Naming a return target independently of a drawdown budget is naming
a Sharpe without knowing it.

This computes, for each Sharpe the project has actually measured and a few it
has not, what annual return a given drawdown budget buys. The drawdown
distribution is bootstrapped from the panel's own daily returns in 21-day
blocks, so the tails and the clustering are the real ones rather than a normal
approximation that would flatter every row (measured excess kurtosis: 9.5).

READ THE Sharpe COLUMN FIRST. Everything else is arithmetic once it is fixed,
and this project's whole difficulty is that column.

    python scripts/diagnostics/what_return_does_a_drawdown_budget_buy.py
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

#: The share of histories allowed to touch the budget. A limit reached by one
#: path in twenty is a limit that means something when it is reached; one
#: reached by three quarters of them is a stop button (R56).
TOLERANCE = 0.05

#: Sharpe ratios this project has MEASURED, and two it has not.
BOOKS = [
    ("SPY, the real market (R48)", 0.553, True),
    ("equal-weight survivors, measured", 0.928, True),
    ("what 20%/yr needs at 10% vol (R48)", 2.000, False),
    ("halfway there", 1.500, False),
]

BUDGETS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def _shape() -> np.ndarray:
    frame, _ = NET._panel([])
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    returns = (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s: s / s.shift(1) - 1.0).to_numpy())
    daily = np.nan_to_num(NET._mean_by_date(returns, codes, len(uniques)))
    return (daily - daily.mean()) / daily.std()


def _worst_drawdowns(shape, index, sharpe: float, vol: float) -> np.ndarray:
    path = shape[index] * (vol / np.sqrt(252)) + sharpe * vol / 252
    equity = np.cumprod(1.0 + path, axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    return (equity / peak - 1.0).min(axis=1)


def _vol_for_budget(shape, index, sharpe: float, budget: float) -> float:
    """The volatility whose 95th-percentile drawdown equals the budget.

    Bisection rather than algebra: drawdown does not scale exactly with vol on
    fat-tailed, clustered returns, and assuming it does is how a plausible
    formula quietly under-states the risk.
    """
    low, high = 0.005, 1.0
    for _ in range(24):
        middle = (low + high) / 2
        reached = abs(np.percentile(
            _worst_drawdowns(shape, index, sharpe, middle), 100 * TOLERANCE))
        if reached > budget:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def main() -> int:
    shape = _shape()
    generator = np.random.default_rng(17)
    length = YEARS * 252
    starts = generator.integers(0, len(shape) - BLOCK,
                                size=(PATHS, length // BLOCK + 1))
    index = (starts[:, :, None] + np.arange(BLOCK)).reshape(PATHS, -1)[:, :length]

    print(f"Bootstrapped from the panel's own returns, {YEARS} years, "
          f"{PATHS:,} paths, {BLOCK}-day blocks.")
    print(f"A budget is the drawdown that {TOLERANCE:.0%} of healthy histories "
          f"reach -- one path in {int(1 / TOLERANCE)}.\n")
    print("ANNUAL RETURN A DRAWDOWN BUDGET BUYS, at each Sharpe\n")
    header = f"{'book':<38}{'Sharpe':>7}" + "".join(
        f"{format(b, '.0%'):>9}" for b in BUDGETS)
    print(header)
    print("-" * len(header))
    for name, sharpe, measured in BOOKS:
        cells = []
        for budget in BUDGETS:
            vol = _vol_for_budget(shape, index, sharpe, budget)
            cells.append(f"{sharpe * vol:>8.1%}")
        mark = "" if measured else "  <- not measured, aspiration"
        print(f"{name:<38}{sharpe:>7.2f}" + "".join(cells) + mark)

    print("\nWHAT THE ROWS MEAN. Leverage, investedness and position size all move "
          "one knob:\n volatility. Moving it slides a row left and right and never "
          "changes the row it is\n on. Only a better Sharpe changes rows, and no "
          "amount of exposure substitutes for it.\n")

    # The question actually asked, in the other direction.
    print("AND THE SAME ARITHMETIC READ BACKWARDS: what Sharpe a wish requires\n")
    print(f"{'wish':<30}{'budget':>9}{'Sharpe needed':>15}"
          f"{'vs measured 0.93':>19}")
    print("-" * 73)
    # Interpolated from a coarse Sharpe grid rather than scanned: the scan was
    # 40,000 bootstraps and would have run for hours to answer what the grid
    # answers in seconds. Return rises monotonically with Sharpe at a fixed
    # budget, so interpolation between grid points is sound -- stated rather
    # than assumed.
    grid = np.array([0.4, 0.7, 1.0, 1.4, 1.8, 2.4, 3.0, 4.0])
    for budget in (0.20, 0.30):
        earned = np.array([sharpe * _vol_for_budget(shape, index, sharpe, budget)
                           for sharpe in grid])
        for wish in (0.10, 0.20, 0.30):
            if wish > earned.max():
                verdict, factor = f"above {grid[-1]:.1f}", ""
            else:
                needed = float(np.interp(wish, earned, grid))
                verdict, factor = f"{needed:.2f}", f"{needed / 0.928:.1f}x"
            print(f"{format(wish, '.0%') + ' a year':<30}{budget:>8.0%}"
                  f"{verdict:>15}{factor:>19}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
