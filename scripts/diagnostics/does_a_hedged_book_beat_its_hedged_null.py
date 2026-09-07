"""A hedged Sharpe needs its own null, or it is just a new number with an old bar.

`fund_earnings_yield_1d` scores -0.234 raw and +0.374 once its market position
is removed. Before that is a finding it has to survive the two checks every
other number in this project has had to survive:

    the null   rotate the positions in time, hedge the ROTATED book the same
               way, and see whether the real one stands above them. Hedging
               the rotated books too is the point -- hedging only the real one
               would compare a corrected book with uncorrected nulls, which
               is how a correction manufactures an edge.

    the bar    1,404 attempts were made to find this, so the bar is
               Bonferroni on 1,404, not on one.

The in-sample beta is fitted separately for every rotation, so the null gets
exactly the same hindsight the real book gets.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path("D:/trading_project")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.targets.calculators.regression_calculator import RegressionCalculator

spec = importlib.util.spec_from_file_location(
    "net", ROOT / "scripts/diagnostics/net_test_every_survivor.py")
NET = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NET)

TOP = ["fund_earnings_yield_1d", "fund_debt_to_equity_1d",
       "fund_price_to_book_1d", "VOLATILITY_50_1d"]
ROTATIONS = 40
ATTEMPTS = 1404

frame, order = NET._panel([])
dates = frame["datetime"].to_numpy()
codes, uniques = pd.factorize(dates, sort=True)
groups = len(uniques)
costs = yaml.safe_load((ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
                       )["targets"]["target_return_1d"]["params"]["transaction_costs"]
friction = np.asarray(RegressionCalculator._round_trip_cost(frame["close"], costs),
                      dtype=float)
by = frame.groupby("ticker", sort=False)["close"]

lags = [int(round((i + 1) / (ROTATIONS + 1) * groups)) for i in range(ROTATIONS)]
rotations = {lag: NET._rotation_index(frame, lag) for lag in lags}
z_bar = float(norm.ppf(1.0 - 0.025 / ATTEMPTS))
print(f"bar: Bonferroni on {ATTEMPTS} attempts -> z >= {z_bar:.2f}\n")

available = pd.read_parquet(NET.BATCH / "features.parquet").columns
present = [c for c in TOP if c in available]
loaded = pd.read_parquet(NET.BATCH / "features.parquet", columns=present)

print(f"{'feature':<26}{'hold':>5}{'raw':>8}{'HEDGED':>9}"
      f"{'null':>8}{'sd':>7}{'z':>7}")
print("-" * 70)
for name in present:
    values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
    position = NET._position(values, dates)
    turned = {}
    for lag in lags:
        moved = position[rotations[lag]]
        turned[lag] = moved - NET._mean_by_date(moved, codes, groups)[codes]
    for hold in (60, 120):
        forward = by.transform(lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy()
        own = np.ones(len(frame))
        market = NET._mean_by_date(own * forward - np.abs(own) * friction,
                                   codes, groups)

        def hedged_sharpe(book_position):
            book = NET._mean_by_date(
                book_position * forward - np.abs(book_position) * friction,
                codes, groups)
            usable = np.isfinite(book) & np.isfinite(market)
            if usable.sum() < 60 or market[usable].std() <= 0:
                return float("nan")
            beta = float(np.cov(book[usable], market[usable])[0, 1]
                         / np.var(market[usable]))
            return NET._sharpe_all_phases(book - beta * market, hold)[0], book

        real_hedged, real_book = hedged_sharpe(position)
        raw = NET._sharpe_all_phases(real_book, hold)[0]
        draws = [hedged_sharpe(turned[lag])[0] for lag in lags]
        mean = float(np.nanmean(draws))
        spread = float(np.nanstd(draws, ddof=1))
        z = (real_hedged - mean) / spread if spread > 0 else float("nan")
        print(f"{name:<26}{hold:>5}{raw:>8.3f}{real_hedged:>9.3f}"
              f"{mean:>8.3f}{spread:>7.3f}{z:>7.2f}")

print(f"\nA hedged number above its own hedged null by more than {z_bar:.2f} "
      f"would be the first\ncandidate this project has had. Anything less is a "
      f"number, not a finding.")
