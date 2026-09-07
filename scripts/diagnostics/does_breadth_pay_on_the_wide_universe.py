"""The same effect, eleven times the breadth: does it start paying?

THE QUESTION, AND WHY IT IS THIS ONE. R52 found real information on our 110
names -- short-term reversal, z up to 10.7, surviving a one-bar delay and a
friction-invariance check -- and it did not pay: the cost drag exceeded the
information at every horizon. R59 then measured that our universe carries 57
effective independent bets for a dollar-neutral book while a 2,684-name one
carries about 650. By IR = IC x sqrt(breadth) that is 3.3x the Sharpe at
constant IC.

So this is not a search. It is ONE pre-registered replication: take the effect
we already found, put it on the wider cross-section, and see whether breadth
turns an unpayable edge into a payable one. Six attempts, not a thousand.

DECLARED BEFORE THE RUN.

    signal    minus the name's own return over LOOKBACK days, ranked
              cross-sectionally. This is short-term reversal, which is what
              R52's `market_return_1d` was in disguise: that column is a
              leave-one-out market mean, so ranking by it is ranking by minus
              the name's own move. Stated plainly here instead of hidden in a
              leave-one-out.
    lookback  1 and 5 days. Two, because R52 found the effect at both and
              picking one after the fact would be choosing on the answer.
    holds     1, 5, 20, 40, 60, 120 -- the six the instrument always uses.
    delay     one bar between the signal being knowable and the position
              earning, always. R32 lost a headline to the zero-latency
              assumption and R52 re-ran with the delay; there is no reason to
              offer the flattering version.
    attempts  2 lookbacks x 6 holds = 12, and the bar is computed from that.
    book      dollar-neutral, both legs pay the per-share friction from
              targets.yaml, Sharpe averaged over every phase -- the same book
              in every respect as the one that produced R22, R28 and R52.
    seal      2013-08-16, this universe's own (R59). Nothing at or after it is
              touched here.

WHAT THE COST MODEL DOES AND DOES NOT INCLUDE. The per-share model charges more,
as a percentage, on cheap names -- which is most of this universe -- so the
friction here is honestly worse than on mega-caps. What it omits is MARKET
IMPACT, and these names carry $1M of median daily volume against our 110's
billions. At the account sizes this project is built for the omission is safe
and the script prints the arithmetic rather than asserting it: a $100k book
spread across hundreds of names holds a few hundred dollars in each, which is
basis points of a $1M day. It stops being safe long before institutional size,
and the printed number says where.

    python scripts/diagnostics/does_breadth_pay_on_the_wide_universe.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import norm  # noqa: E402

from src.pipeline.sealed_period import apply_seal, seal_start_for  # noqa: E402
from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

_spec = importlib.util.spec_from_file_location(
    "net_test", PROJECT_ROOT / "scripts/diagnostics/net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NET)

PANEL = PROJECT_ROOT / "data" / "wide_universe" / "kaggle_us_daily.parquet"
LOOKBACKS = (1, 5)
HOLDS = (1, 5, 20, 40, 60, 120)
ROTATIONS = 20
CAPITALS = (10_000, 100_000, 1_000_000, 10_000_000)


def main() -> int:
    panel = pd.read_parquet(PANEL)
    # THE RULE, NOT A DATE. The first version of this script carried
    # `SEAL = pd.Timestamp("2013-08-16")` and compared by hand, and the seal
    # guard caught it (test_the_seal_has_one_definition). The NUMBER was right;
    # the MECHANISM was a second copy of a rule that already exists, which is
    # the family that gave the seal nine definitions on 2026-09-04. Asked
    # properly, `seal_start_for` returns exactly the same date: this frame ends
    # in 2017, before the declared 2023-09-01, so the rule falls back to the
    # frame's own 80th percentile instead of withholding nothing.
    seal = seal_start_for(panel["datetime"])
    panel, withheld = apply_seal(panel)
    panel = panel.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    dates = panel["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    groups = len(uniques)
    print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique():,} names, "
          f"{panel['datetime'].min().date()} to {panel['datetime'].max().date()}")
    print(f"seal:  {seal.date()} by the project's own per-frame rule; "
          f"{withheld:,} rows held back\n")

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    friction = np.asarray(
        RegressionCalculator._round_trip_cost(panel["close"], costs), dtype=float)
    print(f"friction: {np.nanmean(friction) * 1e4:.2f} bp round trip on average "
          f"(our 110 names: 10.94 bp -- the per-share model charges more here "
          f"because\n          the names are cheaper, which is honest and not "
          f"an artefact)\n")

    # Market impact, priced rather than asserted.
    volume = panel.groupby("ticker")["volume"].median() * panel.groupby(
        "ticker")["close"].median()
    typical = float(volume.median())
    names = panel["ticker"].nunique()
    print(f"MARKET IMPACT, which the cost model above omits. Median name trades "
          f"${typical / 1e6:.1f}M a day.")
    print(f"    {'capital':>12}{'per name':>12}{'share of a day':>17}")
    for capital in CAPITALS:
        per_name = capital / names
        print(f"    {capital:>12,}{per_name:>12,.0f}{per_name / typical:>16.3%}")
    print("    Impact grows with the square root of participation, so a share "
          "measured in basis\n    points is not the binding constraint and a "
          "share in percent is (R24).\n")

    by_name = panel.groupby("ticker", sort=False)["close"]
    forwards = {hold: by_name.transform(
        lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy() for hold in HOLDS}

    attempts = len(LOOKBACKS) * len(HOLDS)
    z_bar = float(norm.ppf(1.0 - 0.025 / attempts))
    print(f"attempts declared before the run: {attempts}. "
          f"Bonferroni bar z >= {z_bar:.2f}\n")

    # The constant opponent, printed before any book.
    own = np.ones(len(panel))
    print("BUY EVERYTHING (the opponent), net of the same friction")
    opponent = {}
    for hold in HOLDS:
        opponent[hold] = NET._sharpe_all_phases(
            NET._mean_by_date(own * forwards[hold] - np.abs(own) * friction,
                              codes, groups), hold)[0]
        print(f"    hold {hold:>3}: {opponent[hold]:+.3f}")
    print("    SURVIVOR SNAPSHOT AT 2017: these names were all alive then, so "
          "this opponent is\n    inflated exactly as ours is (R47). It is a "
          "relative benchmark, not the market.\n")

    lags = [int(round((i + 1) / (ROTATIONS + 1) * groups)) for i in range(ROTATIONS)]
    rotations = {lag: NET._rotation_index(panel, lag) for lag in lags}

    header = (f"{'signal':<22}{'hold':>6}{'NET':>9}{'null':>9}{'sd':>7}"
              f"{'z':>8}{'vs opponent':>14}")
    print(header)
    print("-" * len(header))
    rows = []
    for lookback in LOOKBACKS:
        past = by_name.transform(
            lambda s, k=lookback: s / s.shift(k) - 1.0).to_numpy()
        column = -np.nan_to_num(past, nan=0.0)
        position = NET._position(column, dates)
        # One bar of delay, always.
        position = position[NET._rotation_index(panel, 1)]
        position = position - NET._mean_by_date(position, codes, groups)[codes]
        turned = {}
        for lag in lags:
            moved = position[rotations[lag]]
            turned[lag] = moved - NET._mean_by_date(moved, codes, groups)[codes]
        for hold in HOLDS:
            def score(book):
                return NET._sharpe_all_phases(
                    NET._mean_by_date(
                        book * forwards[hold] - np.abs(book) * friction,
                        codes, groups), hold)[0]

            real = score(position)
            draws = [score(turned[lag]) for lag in lags]
            mean, spread = float(np.nanmean(draws)), float(np.nanstd(draws, ddof=1))
            z = (real - mean) / spread if spread > 0 else float("nan")
            rows.append((lookback, hold, real, mean, spread, z,
                         real - opponent[hold]))
            print(f"{'reversal ' + str(lookback) + 'd':<22}{hold:>6}{real:>9.3f}"
                  f"{mean:>9.3f}{spread:>7.3f}{z:>8.2f}"
                  f"{real - opponent[hold]:>14.3f}", flush=True)

    print("\n" + "=" * len(header))
    knows = [r for r in rows if np.isfinite(r[5]) and r[5] >= z_bar]
    earns = [r for r in knows if r[2] > 0]
    beats = [r for r in earns if r[6] > 0]
    print(f"know something about WHEN (z >= {z_bar:.2f})      {len(knows)} of {len(rows)}")
    print(f"of those, net positive                     {len(earns)}")
    print(f"of those, BEAT buying everything           {len(beats)}")
    if beats:
        print("\nA book that knows, earns, and beats the opponent. That has not "
              "happened before.\nThe next step is NOT another variant: it is one "
              "pre-registered confirmation on\nthis universe's sealed period, "
              "which is what the seal is for.")
    elif earns:
        print("\nEarns but does not beat owning the same names. A measured "
              "number, not a candidate\n(R28, R53).")
    else:
        print("\nBreadth did not turn this edge into money. Since R59 measured "
              "the breadth gain at\nabout eleven times, the shortfall is not "
              "width -- it is the edge itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
