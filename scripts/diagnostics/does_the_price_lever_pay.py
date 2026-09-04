"""Does restricting to dearer names buy more than it costs?

The one cost lever that measurement supports: the IBKR per-share commission as
a fraction of the position is `0.0035 / price`, so a $20 share pays 1.75bp a
side and a $230 share pays 0.15bp. Restricting the book to the dearer half of
the universe therefore cuts friction.

It also cuts breadth. Halving the names multiplies the portfolio's volatility
by about sqrt(2) for the same per-name edge, so gross Sharpe falls by the same
factor. The question is which falls faster, and it is arithmetic on data
already in hand -- no new search, no new multiplicity.

Running a second 46 x 5 sweep on a price-restricted universe would ADD 230
attempts to the 230 already made, and R23 exists because a threshold that
ignores the attempt count is how this project keeps fooling itself. So the
arithmetic goes first: if the trade cannot work, there is nothing to search.
"""
import sys
from pathlib import Path

sys.path.insert(0, "D:/trading_project")

import numpy as np
import pandas as pd
import yaml

from src.targets.calculators.regression_calculator import RegressionCalculator

BATCH = Path("D:/trading_project/data/colab/accumulated/main_database")
SEALED = pd.Timestamp("2023-09-01", tz="UTC")

costs = yaml.safe_load(
    Path("D:/trading_project/src/config/targets.yaml").read_text(encoding="utf-8")
)["targets"]["target_return_1d"]["params"]["transaction_costs"]

frame = pd.read_parquet(BATCH / "features.parquet",
                        columns=["ticker", "datetime", "interval", "close"])
frame = frame[frame["interval"] == "1d"].copy()
frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
frame = frame[frame["datetime"] < SEALED].dropna(subset=["close"])
frame["friction"] = np.asarray(
    RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)

median_price = frame.groupby("ticker")["close"].median().sort_values()
print(f"{len(median_price)} names, median price "
      f"{median_price.min():.1f} to {median_price.max():.1f}\n")

full_n = len(median_price)
full_friction = frame["friction"].mean()
print(f"{'kept':>22}{'names':>7}{'median $':>10}{'friction/rt':>13}"
      f"{'cost cut':>10}{'breadth cost':>14}{'net gain':>10}")
print("-" * 86)
for label, share in (("all", 1.0), ("dearest 3/4", 0.75), ("dearest half", 0.5),
                     ("dearest third", 1/3), ("dearest quarter", 0.25),
                     ("dearest tenth", 0.10)):
    keep = median_price.tail(max(2, int(round(full_n * share)))).index
    block = frame[frame["ticker"].isin(keep)]
    friction = block["friction"].mean()
    n = len(keep)
    cost_cut = full_friction / friction            # how much cheaper, as a factor
    breadth_cost = np.sqrt(full_n / n)             # gross Sharpe divided by this
    print(f"{label:>22}{n:>7}{block['close'].median():>10.1f}"
          f"{friction:>13.4%}{cost_cut:>10.2f}x{breadth_cost:>13.2f}x"
          f"{cost_cut / breadth_cost:>10.2f}x")

print("\nReading: 'cost cut' is how many times cheaper the friction becomes,")
print("'breadth cost' how many times the gross Sharpe shrinks. The last column")
print("is the ratio -- above 1.0 the restriction wins, below it loses.")
print("\nThis is necessary, not sufficient: the gross edge must also survive on")
print("the smaller universe, which this does not test.")
