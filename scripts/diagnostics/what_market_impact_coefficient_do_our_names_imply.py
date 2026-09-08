"""Is the market-impact coefficient 0.1, 0.0001, or neither?

REGISTER #290 found that the project measures with one cost model and would
trade with another, and that the dominant term of the trading one --
`market_impact_coefficient` -- was declared nowhere while a value a THOUSAND
times smaller sat in the config read only by dead archive code. It left the
choice open on purpose: which number is right has an empirical answer, and
inventing one instead of measuring is the defect the whole register is about.

This is that measurement, and it starts by reading the formula rather than the
number. `AdvancedBacktestEngine.calculate_execution_costs` computes

    impact = trade_value * C * sqrt(participation),   participation = Q / ADV

so impact AS A FRACTION of trade value is `C * sqrt(p)`. The square-root law
that shape comes from is

    impact_fraction ~= Y * sigma_daily * sqrt(p),     Y of order 1

-- and OUR formula has no sigma in it at all. Whatever C is, it is standing in
for a daily volatility, and that is a quantity this panel can measure. Note
the same engine DOES scale its slippage by volatility, so the omission is in
one term and not a stated convention.

So the question "0.1 or 0.0001" becomes "what is sigma on our names", which is
answerable, plus "what Y does each choice imply", which is then arithmetic.

    python scripts/diagnostics/what_market_impact_coefficient_do_our_names_imply.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: The preserved pre-rebuild batch is used when it exists, so this measurement
#: does not move under a rebuild running beside it.
BATCHES = (
    PROJECT_ROOT / "data/colab/accumulated/pre_backfill_20260908",
    PROJECT_ROOT / "data/colab/accumulated/main_database",
)

#: The two numbers in play. 0.1 is the live default, declared in strategy.yaml
#: on 2026-09-06 because it was already acting; 0.0001 is
#: `backtesting.market_impact.sqrt_coefficient`, read only by
#: src/archive/backtesting/engine.py.
DECLARED = {"live default (strategy.yaml)": 0.1,
            "config, read only by dead archive code": 0.0001}

#: Order sizes to price, in dollars. The register quotes costs at these.
ORDERS = (500.0, 10_000.0, 25_000.0)


def main() -> int:
    batch = next((b for b in BATCHES if (b / "features.parquet").exists()), None)
    if batch is None:
        print("no batch on disk")
        return 1

    frame = pd.read_parquet(
        batch / "features.parquet",
        columns=["datetime", "ticker", "interval", "close", "volume"])
    frame = frame[frame["interval"].astype(str) == "1d"]
    stamps = pd.to_datetime(frame["datetime"])
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_localize(None)
    seal = pd.Timestamp(SEAL_START).tz_localize(None)
    frame = frame.loc[(stamps < seal).to_numpy()].copy()
    frame["datetime"] = stamps[(stamps < seal).to_numpy()].to_numpy()
    frame = frame.sort_values(["ticker", "datetime"])
    print(f"batch {batch.name}: {len(frame):,} explorable daily rows, "
          f"{frame['ticker'].nunique()} names\n")

    frame["ret"] = frame.groupby("ticker")["close"].pct_change(fill_method=None)
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    usable = frame[np.isfinite(frame["ret"])
                   & (frame["dollar_volume"] > 0)].copy()

    sigma = usable.groupby("ticker")["ret"].std()
    adv = usable.groupby("ticker")["dollar_volume"].median()

    print("=== what the square-root law needs, measured on this panel ===")
    print(f"  daily return sd per name   median {sigma.median():.4f}   "
          f"25th {sigma.quantile(.25):.4f}   75th {sigma.quantile(.75):.4f}")
    print(f"  median dollar volume       median ${adv.median():,.0f}   "
          f"25th ${adv.quantile(.25):,.0f}   75th ${adv.quantile(.75):,.0f}")
    print()

    print("=== what each declared coefficient implies for Y ===")
    print("    impact_fraction = C * sqrt(p) is being asked to equal")
    print("    Y * sigma * sqrt(p), so Y = C / sigma. Y is O(1) in the")
    print("    literature this shape comes from.\n")
    for label, coefficient in DECLARED.items():
        implied = coefficient / sigma.median()
        print(f"  C = {coefficient:<8} ({label})")
        print(f"      implies Y = {implied:.4g}  -- "
              f"{'far too expensive' if implied > 3 else 'far too cheap' if implied < 0.2 else 'plausible'}")
    print(f"\n  C that makes Y = 1 on this panel: {sigma.median():.4f}")
    print(f"  C that makes Y = 0.5:              {sigma.median() * 0.5:.4f}")
    print()

    print("=== what an order actually costs at each C, on the median name ===")
    typical_adv = float(adv.median())
    header = f"{'order':>10}{'participation':>15}"
    for label in DECLARED:
        header += f"{DECLARED[label]:>14}"
    header += f"{sigma.median():>14.4f}"
    print(header)
    print(f"{'':>10}{'':>15}{'0.1':>14}{'0.0001':>14}{'measured':>14}")
    print("-" * len(header))
    for order in ORDERS:
        participation = order / typical_adv
        row = f"${order:>9,.0f}{participation:>15.2e}"
        for coefficient in list(DECLARED.values()) + [float(sigma.median())]:
            row += f"{coefficient * np.sqrt(participation):>13.4%} "
        print(row)

    print()
    print("Read the participation column first. At $25,000 against a median\n"
          "name it is a ten-thousandth of a day's volume, and the square-root\n"
          "law is not the binding cost at that size -- commission and spread\n"
          "are (R22, R24). The coefficient matters for what it would do at\n"
          "SIZE, which is a question about a capital base this project does\n"
          "not have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
