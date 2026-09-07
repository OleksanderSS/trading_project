"""Dollar-neutral is not market-neutral, and nobody here had measured the difference.

R60 built a book on 2,621 names, called it dollar-neutral as this project always
does, and found its correlation with buy-everything to be **+0.225**. Subtracting
the per-date mean removes the LEVEL of the cross-section; it does not remove
BETA. Reversal buys what just fell, and in a selloff what just fell is the
high-beta names, so the book ends up long beta exactly when beta is cheap.

That applies to every book this project has ever scored -- R22, R23, R28, R51,
R52 -- and in none of them was this number computed. If a column's net Sharpe is
market exposure wearing a neutral label, then part of what was read as "no edge"
was actually "an edge cancelled by an unintended market position", and part of
what was read as an edge was the market.

WHAT IS MEASURED, for every column at every horizon.

    beta        of the book's daily P&L on the equal-weight book's, which is
                this project's constant opponent (R28).
    correlation the same relationship without the scale, because a small beta
                on a volatile book is not small.
    hedged      the Sharpe of the residual after subtracting beta x the
                opponent. This is what the column earns with its market
                position taken away.

THE HEDGE IS GENEROUS AND THAT IS STATED. Beta is fitted IN SAMPLE, on the same
rows it is then removed from, so the hedge uses hindsight to pick its own ratio.
A real hedge would be estimated on earlier data and would remove less. So the
hedged column flatters every book, and any column that looks WORSE after this
hedge looks worse than it really is by a margin -- which is the safe direction
for the conclusion "the raw number was market exposure".

The sealed period is untouched: the panel comes from the instrument's own
loader, which applies the per-frame rule.

    python scripts/diagnostics/how_much_of_the_book_is_just_the_market.py
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

from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

_spec = importlib.util.spec_from_file_location(
    "net_test", PROJECT_ROOT / "scripts/diagnostics/net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NET)

HOLDS = (1, 5, 20, 40, 60, 120)

#: Columns the project's standing claims rest on, named so they are reported
#: whatever the sweep finds. Picking them after seeing the answer would be the
#: usual sin.
HEADLINES = ("VOLATILITY_50_1d", "gk_volatility_1d", "market_return_1d",
             "peer_breadth_1d", "state_SHARPE_RATIO_1d",
             "LEVEL_BREAKOUT_DOWN_20_1d", "state_SORTINO_RATIO_1d")


def _sharpe(series: np.ndarray, hold: int) -> float:
    return NET._sharpe_all_phases(series, hold)[0]


def main() -> int:
    frame, order = NET._panel([])
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    groups = len(uniques)
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}")
    print(f"seal:  nothing at or after {NET.SEALED.date()} is touched\n")

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    friction = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)

    by_name = frame.groupby("ticker", sort=False)["close"]
    forwards = {hold: by_name.transform(
        lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy() for hold in HOLDS}

    own = np.ones(len(frame))
    opponent = {hold: NET._mean_by_date(
        own * forwards[hold] - np.abs(own) * friction, codes, groups)
        for hold in HOLDS}
    print("the opponent this beta is measured against -- buy everything, same "
          "friction, same clock:")
    for hold in HOLDS:
        print(f"    hold {hold:>3}: {_sharpe(opponent[hold], hold):+.3f}")
    print()

    roles = pd.read_csv(NET.ROLES)
    names = [n for n in roles[roles["varies"] > NET.MIN_VARIES]["feature"].tolist()
             if n != "close"]
    print(f"{len(names)} columns, {len(HOLDS)} horizons\n")

    rows = []
    for start in range(0, len(names), NET.CHUNK):
        block = names[start:start + NET.CHUNK]
        loaded = pd.read_parquet(NET.BATCH / "features.parquet",
                                 columns=list(dict.fromkeys(block)))
        for name in block:
            values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
            if pd.Series(values).notna().sum() < 10_000:
                continue
            position = NET._position(values, dates)
            for hold in HOLDS:
                book = NET._mean_by_date(
                    position * forwards[hold] - np.abs(position) * friction,
                    codes, groups)
                market = opponent[hold]
                usable = np.isfinite(book) & np.isfinite(market)
                if usable.sum() < 60 or market[usable].std() <= 0:
                    continue
                beta = float(np.cov(book[usable], market[usable])[0, 1]
                             / np.var(market[usable]))
                correlation = float(np.corrcoef(book[usable], market[usable])[0, 1])
                hedged = book - beta * market
                rows.append({
                    "feature": name, "hold": hold,
                    "net": _sharpe(book, hold),
                    "beta": beta, "correlation": correlation,
                    "hedged": _sharpe(hedged, hold),
                })
        del loaded

    report = pd.DataFrame(rows)
    report["cost_of_hedging"] = report["net"] - report["hedged"]
    out = PROJECT_ROOT / "diagnostic_reports" / "book_beta.csv"
    report.to_csv(out, index=False)

    print("HOW NEUTRAL IS A DOLLAR-NEUTRAL BOOK, across every column and horizon\n")
    print(f"    {'quantity':<26}{'median':>10}{'90th pct':>11}{'max':>10}")
    print("    " + "-" * 57)
    for label, column in (("|beta|", report["beta"].abs()),
                          ("|correlation|", report["correlation"].abs())):
        print(f"    {label:<26}{column.median():>10.3f}"
              f"{column.quantile(0.9):>11.3f}{column.max():>10.3f}")
    share = float((report["correlation"].abs() > 0.2).mean())
    print(f"\n    books whose |correlation| with the market exceeds 0.2: "
          f"{share:.0%}")

    print("\n\nWHAT HEDGING DOES TO THE COLUMNS THE STANDING CLAIMS REST ON\n")
    print(f"    {'feature':<28}{'hold':>5}{'net':>8}{'beta':>8}"
          f"{'corr':>7}{'HEDGED':>9}{'change':>9}")
    print("    " + "-" * 74)
    for name in HEADLINES:
        rows_for = report[report["feature"] == name]
        if rows_for.empty:
            print(f"    {name:<28}   not among the measured columns")
            continue
        best = rows_for.loc[rows_for["net"].idxmax()]
        print(f"    {name:<28}{int(best['hold']):>5}{best['net']:>8.3f}"
              f"{best['beta']:>8.3f}{best['correlation']:>7.3f}"
              f"{best['hedged']:>9.3f}{best['hedged'] - best['net']:>9.3f}")

    print("\n\nAND THE QUESTION THIS RUN EXISTS FOR\n")
    best_raw = report.loc[report["net"].idxmax()]
    best_hedged = report.loc[report["hedged"].idxmax()]
    print(f"    best NET anywhere      {best_raw['net']:+.3f}  "
          f"({best_raw['feature']}, hold {int(best_raw['hold'])}) "
          f"-> hedged {best_raw['hedged']:+.3f}")
    print(f"    best HEDGED anywhere   {best_hedged['hedged']:+.3f}  "
          f"({best_hedged['feature']}, hold {int(best_hedged['hold'])}) "
          f"-> raw net {best_hedged['net']:+.3f}")
    helped = int((report["cost_of_hedging"] < 0).sum())
    print(f"\n    columns that improve when the market is removed: {helped} of "
          f"{len(report)}")
    print(f"    columns that get worse:                          "
          f"{len(report) - helped}")
    print(f"\n    A column that gets WORSE was being paid by its market position. "
          f"One that gets\n    BETTER was being taxed by one -- and its edge was "
          f"partly hidden by exposure\n    nobody intended.")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
