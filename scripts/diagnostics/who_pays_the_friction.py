"""Who pays the friction in a long/short book?

`target_return_1d` is built with `adjust_for_costs: true`, so the stored value
is already `raw_return - round_trip_cost`. That is right for a LONG.

A short position multiplies it: `-1 * (raw - cost) = -raw + cost`. The short
is CREDITED the friction instead of paying it. In a book that is half short,
the costs roughly cancel and the curve is effectively gross -- which is
exactly the flattering direction, and the reason to check before believing a
Sharpe of 2.19.

Both legs pay. The correction is to add the cost back and then subtract it
against the ABSOLUTE position.
"""
import sys
from pathlib import Path

from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.pipeline.stages.evaluation.holdout_equity import build_holdout_equity
from src.pipeline.stages.evaluation.metrics_calculator import (
    EvaluationMetricsCalculator,
)
from src.targets.calculators.regression_calculator import RegressionCalculator

BATCH = Path("D:/trading_project/data/colab/accumulated/main_database")
FEATURE = "CDL_UPPER_WICK_RATIO_1d"
TARGET = "target_return_1d"
from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: Imported, never restated. Eight diagnostics each kept their own copy of this
#: date until 2026-09-04. The policy in docs/SEALED_HOLDOUT.md says moving the
#: seal EARLIER is "always safe" -- with eight copies it would have been safe in
#: one file and silently ignored in the other seven, which is the duplication
#: family this codebase's defects come from.
SEALED = SEAL_START

config = yaml.safe_load(
    Path("D:/trading_project/src/config/targets.yaml").read_text(encoding="utf-8")
)
costs = config["targets"][TARGET]["params"]["transaction_costs"]
print(f"cost profile: {costs}\n")

ident = ["ticker", "datetime", "interval"]
feat = pd.read_parquet(BATCH / "features.parquet", columns=ident + [FEATURE, "close"])
feat = feat[feat["interval"] == "1d"]
tgt = pd.read_parquet(BATCH / "targets.parquet", columns=ident + [TARGET])
tgt = tgt[tgt["interval"] == "1d"]
frame = feat.merge(tgt[["ticker", "datetime", TARGET]], on=["ticker", "datetime"])
frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
frame = frame[frame["datetime"] < SEALED].dropna(subset=[FEATURE, TARGET, "close"])

cost = RegressionCalculator._round_trip_cost(frame["close"], costs)
cost = pd.Series(np.asarray(cost, dtype=float), index=frame.index)
print(f"round-trip friction per bar: mean {cost.mean():.4%}, "
      f"median {cost.median():.4%}")
print(f"annualised if rebalanced daily: {cost.mean() * 252:.1%}\n")

ranks = frame.groupby("datetime")[FEATURE].rank(pct=True)
position = np.sign((ranks - 0.5).to_numpy())
# Dollar-neutral, or the answer is the market (CLAIMS R28). A column with
# heavy ties ranks every name just above 0.5, so `sign` sends them all the
# same way and the "book" is buy-and-hold. CDL_UPPER_WICK_RATIO_1d has
# 241,839 distinct values and a net exposure of +0.011, so this line changes
# nothing for the feature this script is pinned to -- it exists so that the
# next feature someone points it at cannot silently become the market.
position = position - pd.Series(position).groupby(
    frame["datetime"].to_numpy()).transform("mean").to_numpy()
print(f"net exposure after centring: {position.mean():+.4f}")
print()
raw = frame[TARGET].to_numpy() + cost.to_numpy()          # undo the long-only subtraction
honest = position * raw - np.abs(position) * cost.to_numpy()   # both legs pay

for label, series in (
    ("as stored (short is CREDITED the cost)", position * frame[TARGET].to_numpy()),
    ("gross, no costs at all", position * raw),
    ("both legs pay the friction", honest),
):
    book = pd.DataFrame({
        "target": TARGET, "context": "BOOK", "ticker": frame["ticker"].to_numpy(),
        "datetime": frame["datetime"].to_numpy(),
        "prediction": np.ones(len(frame)), "actual": series,
    })
    curve = build_holdout_equity(book)
    if curve.get("status") != "built":
        print(f"{label}: {curve}")
        continue
    m = EvaluationMetricsCalculator(None)._calculate_basic_metrics(
        curve["portfolio_history"])
    print(f"{label:<42} Sharpe {m['sharpe_ratio']:+7.3f}   "
          f"total return {m['total_return']:+8.3f}   "
          f"maxDD {m['max_drawdown']:+.3f}")
