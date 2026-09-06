"""Would the declared risk limits liquidate a healthy book, and do they agree with each other?

A risk limit exists to say "something has broken". A limit a WORKING book
reaches routinely cannot say that -- it only says "stop trading", and it says
it at the worst possible moment, which is the bottom of an ordinary drawdown.

So the question is not whether 15% is a nice round number. It is: how often
does a healthy book of the quality we are aiming for reach 15%? That is
measurable three ways, and this script does all three.

    1. ON THE REAL PANEL. The equal-weight "buy everything" book is the thing
       any strategy here has to beat, and it is the only thing in this project
       that actually makes money (R28: it pays +1.018 net Sharpe at a 60-day
       hold). If the limit would have liquidated IT, the limit is wrong.

    2. BY BOOTSTRAP, with the real tail shape. Normal draws understate
       drawdowns badly -- the measured excess kurtosis of this panel's daily
       returns is about 9.5 -- so the paths are built by block-bootstrapping
       the actual return shape and rescaling it to a target Sharpe and
       volatility. A 21-day block keeps the clustering that makes drawdowns.

    3. AGAINST EACH OTHER AND AGAINST THE GOAL. Limits can be individually
       plausible and jointly impossible. `max_total_risk_pct` caps invested
       capital; the owner's goal is a return on TOTAL capital; those two and
       `max_leverage` are three numbers with one arithmetic relation between
       them, and nothing had ever checked it.

WHAT THIS SCRIPT DOES NOT DO. It does not choose the numbers. It computes the
level a healthy book reaches with a stated probability, and prints where the
declared numbers sit against it. How often the owner is willing to be stopped
out is a risk appetite and is his; a limit that fires on three quarters of
healthy histories is not an appetite, it is an error, and that part is not a
matter of taste.

    python scripts/diagnostics/do_the_risk_limits_survive_their_own_book.py
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

from src.config.unified_config_manager import UnifiedConfigManager  # noqa: E402

#: Block length for the bootstrap. Drawdowns are made of CLUSTERED bad days, so
#: resampling single days would destroy the very thing being measured. A month
#: of trading keeps the clustering without pinning the path to one episode.
BLOCK = 21

#: Twenty-seven years, the explorable span of the daily frame.
YEARS = 27
PATHS = 6000

#: The owner's stated goal (R48): 20% a year is Sharpe ~2.0 at 10% volatility.
TARGET = ("the goal: Sharpe 2.0 at 10% vol", 2.0, 0.10)


def _equal_weight_daily() -> np.ndarray:
    frame, _ = NET._panel([])
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    returns = (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s: s / s.shift(1) - 1.0).to_numpy())
    return np.nan_to_num(NET._mean_by_date(returns, codes, len(uniques)))


def _drawdown_path(returns: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + returns, axis=-1)
    peak = np.maximum.accumulate(equity, axis=-1)
    return equity / peak - 1.0


def main() -> int:
    config = UnifiedConfigManager()
    risk = config.get("strategy.risk_management", {})
    declared_dd = float(risk.get("max_drawdown_pct", 0.15))
    declared_daily = float(risk.get("max_daily_loss_pct", 0.03))
    declared_total = float(risk.get("max_total_risk_pct", 0.30))
    declared_pos = float(risk.get("max_position_size_pct", 0.10))
    declared_lev = float(risk.get("max_leverage", 2.0))
    goal = 0.20

    daily = _equal_weight_daily()
    drawdown = _drawdown_path(daily)
    annual = daily.mean() * 252
    vol = daily.std() * np.sqrt(252)

    print("1. THE BOOK THE LIMITS WOULD HAVE GOVERNED -- equal weight, real panel\n")
    print(f"    return {annual:.2%}/yr, volatility {vol:.2%}, "
          f"Sharpe {annual / vol:.3f}")
    print(f"    WORST DRAWDOWN            {drawdown.min():.1%}")
    print(f"    declared max_drawdown_pct {declared_dd:.1%}"
          f"   -> {'BREACHED' if drawdown.min() < -declared_dd else 'survived'}")
    if drawdown.min() < -declared_dd:
        first = int(np.argmax(drawdown < -declared_dd))
        print(f"    first breach on day {first:,} of {len(drawdown):,}; the book "
              f"spends {(drawdown < -declared_dd).mean():.1%} of all days below it")
    hits = int((daily < -declared_daily).sum())
    print(f"    declared max_daily_loss_pct {declared_daily:.1%} would have fired "
          f"{hits} times in {len(daily) / 252:.0f} years")

    print("\n2. HOW DEEP A HEALTHY BOOK GOES, with this panel's real tails\n")
    shape = (daily - daily.mean()) / daily.std()
    print(f"    excess kurtosis of the real series: {pd.Series(shape).kurt():.1f} "
          f"(0 for normal), so normal draws would understate this\n")
    generator = np.random.default_rng(7)
    length = YEARS * 252
    starts = generator.integers(0, len(shape) - BLOCK, size=(PATHS, length // BLOCK + 1))
    index = (starts[:, :, None] + np.arange(BLOCK)).reshape(PATHS, -1)[:, :length]
    drawn = shape[index]

    print(f"    {'book':<30}{'median':>9}{'95th':>9}{'99th':>9}"
          f"{'breach ' + format(declared_dd, '.0%'):>14}")
    print("    " + "-" * 71)
    rows = [TARGET, ("Sharpe 1.5 at 10% vol", 1.5, 0.10),
            ("the equal-weight book", annual / vol, vol)]
    floor_for_target = None
    for name, sharpe, book_vol in rows:
        path = drawn * (book_vol / np.sqrt(252)) + sharpe * book_vol / 252
        worst = _drawdown_path(path).min(axis=1)
        print(f"    {name:<30}{np.median(worst):>8.1%}{np.percentile(worst, 5):>9.1%}"
              f"{np.percentile(worst, 1):>9.1%}{(worst < -declared_dd).mean():>13.0%}")
        if name == TARGET[0]:
            floor_for_target = (np.percentile(worst, 5), np.percentile(worst, 1))

    if floor_for_target is not None:
        five, one = (abs(x) for x in floor_for_target)
        print(f"\n    A limit is a 'something broke' signal only if a HEALTHY book "
              f"rarely reaches it.\n    For the goal that means at least "
              f"{five:.0%} (reached by 1 path in 20) and {one:.0%} (1 in 100).\n"
              f"    Where between them is risk appetite and belongs to the owner. "
              f"{declared_dd:.0%} is not\n    an appetite: it fires on "
              f"{(_drawdown_path(drawn * (TARGET[2] / np.sqrt(252)) + TARGET[1] * TARGET[2] / 252).min(axis=1) < -declared_dd).mean():.0%} "
              f"of healthy histories.")

    print("\n3. DO THE DECLARED NUMBERS AGREE WITH EACH OTHER AND WITH THE GOAL\n")
    print(f"    max_total_risk_pct {declared_total:.0%} caps invested capital, "
          f"while max_leverage {declared_lev:.1f} permits {declared_lev:.0%}.")
    print(f"    The leverage limit allows {declared_lev / declared_total:.1f}x what "
          f"the total limit allows, so it can never bind: it is decoration.\n")
    print(f"    A {goal:.0%} return on TOTAL capital with only {declared_total:.0%} "
          f"invested needs {goal / declared_total:.1%} on the invested part.")
    print(f"    The equal-weight book makes {annual:.2%} and SPY makes 10.80% "
          f"(R48), so that is about {goal / declared_total / annual:.1f}x the market.")
    print(f"    The goal and the total-risk cap cannot both be right.\n")
    print(f"    max_position_size_pct {declared_pos:.0%} against the "
          f"{declared_total:.0%} cap means at least "
          f"{int(np.ceil(declared_total / declared_pos))} positions. R33 measured "
          f"110 names\n    as 3.16 independent bets, so that is roughly all the "
          f"independence on offer -- the one\n    pair of these numbers that "
          f"agrees with a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
