"""What the 27.6%/yr friction is actually made of, and what removing it buys.

WHY THIS EXISTS. Every "does not pay" verdict in this project rests on one cost
model in `targets.yaml` -- per-share commission $0.0035, minimum $0.35, order
$10,000, spread 1 bp, slippage 1 bp -- and nobody had ever taken it apart. A
per-share commission is a fixed cost per SHARE, so as a PERCENTAGE it depends
on the price of the name. That makes the bill a property of the broker and of
which names are held, not of the market. R52 found information worth +0.23
Sharpe at a 20-day hold against a cost drag of -0.39, so the split decides
whether that gap is a market fact or a purchasing decision.

FOUR THINGS ARE MEASURED, and the last three exist to KILL the answer the
first one suggests.

    1. the split      commission versus spread and slippage, overall and by
                      price bucket.
    2. the opponent   buy every name, same cost model, same clock. R28 exists
                      because seven columns once scored ~1.00 net and every
                      one of them WAS this opponent.
    3. the margin     how much worse execution can get before the answer flips.
                      A commission-free broker is paid through order flow, so
                      the 1 bp spread this model keeps is the optimistic half
                      of the trade and has to be stress-tested.
    4. invariance     the rotation null shifts a position in time while the
                      friction stays attached to the row. If the bill moves,
                      a rotation z is measuring which names are cheap rather
                      than when to hold them.

    python scripts/diagnostics/what_the_costs_are_made_of.py
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
    "control",
    PROJECT_ROOT / "scripts/diagnostics/what_edge_would_the_net_test_have_seen.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

#: The horizon R52 put the candidate on. Fixed here rather than swept: this
#: file prices ONE result, and sweeping the horizon would make it a search.
HOLD = 5

#: The column R52 left standing, and the three it left beside it.
CANDIDATE = "market_return_1d"
BESIDE = ["peer_breadth_1d", "peer_return_1d", "state_LAG_5_1d"]


def main() -> int:
    frame, order = C.NET._panel([])
    dates = frame["datetime"].to_numpy()
    sorted_codes, uniques = pd.factorize(dates, sort=True)
    n_groups = len(uniques)

    base = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    free = dict(base, per_share_fee=0.0, min_fee_per_order=0.0)

    def friction_for(model):
        return np.asarray(
            RegressionCalculator._round_trip_cost(frame["close"], model),
            dtype=float)

    full, spread_only = friction_for(base), friction_for(free)
    commission = full - spread_only

    print(f"panel {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"median close ${frame['close'].median():.2f}, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}")
    print(f"seal:  nothing at or after {C.NET.SEALED.date()} is touched\n")

    # ------------------------------------------------------------- 1. split
    print("1. WHAT THE ROUND TRIP IS MADE OF\n")
    print(f"    {'component':<28}{'bp':>8}{'share':>9}{'annualised':>13}")
    print("    " + "-" * 58)
    for label, series in (("full round trip", full),
                          ("  commission (per share)", commission),
                          ("  spread + slippage", spread_only)):
        bp = float(np.nanmean(series)) * 1e4
        print(f"    {label:<28}{bp:>8.2f}{np.nanmean(series) / np.nanmean(full):>8.0%}"
              f"{bp / 1e4 * 252:>12.1%}")

    print(f"\n    a per-share fee is a PERCENTAGE that depends on the price:\n")
    print(f"    {'close':<14}{'rows':>11}{'bp':>8}{'commission share':>19}")
    print("    " + "-" * 52)
    edges = [0, 20, 50, 100, 200, 1e9]
    labels = ["under $20", "$20-50", "$50-100", "$100-200", "over $200"]
    bucket = pd.cut(frame["close"], edges, labels=labels)
    for label in labels:
        mask = (bucket == label).to_numpy()
        if not mask.sum():
            continue
        print(f"    {label:<14}{int(mask.sum()):>11,}"
              f"{float(np.nanmean(full[mask])) * 1e4:>8.2f}"
              f"{float(np.nanmean(commission[mask]) / np.nanmean(full[mask])):>18.0%}")

    forward = (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s: s.shift(-HOLD) / s - 1.0).to_numpy())

    # ---------------------------------------------------------- 2. opponent
    print(f"\n2. THE CONSTANT OPPONENT at hold {HOLD}, printed before any book\n")
    for label, fr in (("full cost", full), ("commission-free", spread_only)):
        own = np.ones(len(frame))
        print(f"    buy everything, {label:<18}"
              f"Sharpe {C._sharpe_given_position(own, dates, forward, fr, HOLD, sorted_codes, n_groups):+.3f}")
    print("    SPY over this window          Sharpe +0.553  (R48)")

    loaded = pd.read_parquet(C.NET.BATCH / "features.parquet",
                             columns=[CANDIDATE] + BESIDE)
    values = pd.to_numeric(loaded[CANDIDATE], errors="coerce").to_numpy()[order]
    position = C._position(values, dates)
    # One-bar delay: the column is same-bar leave-one-out, so trading on the
    # same close assumes zero latency. R32 lost a headline to exactly this.
    position = position[C._rotation_index(frame, 1)]
    position = position - C._mean_by_date(position, sorted_codes, n_groups)[sorted_codes]

    # ------------------------------------------------------------ 3. margin
    print(f"\n3. HOW MUCH WORSE EXECUTION CAN GET, commission-free, {CANDIDATE}\n")
    print(f"    {'spread each way':<20}{'slippage':>11}{'net Sharpe':>14}")
    print("    " + "-" * 45)
    for extra in (0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012):
        model = dict(free, spread_pct=extra, slippage_pct=extra)
        print(f"    {extra * 1e4:>9.0f} bp{'':<9}{extra * 1e4:>8.0f} bp"
              f"{C._sharpe_given_position(position, dates, forward, friction_for(model), HOLD, sorted_codes, n_groups):>14.3f}")

    # -------------------------------------------------------- 4. invariance
    print("\n4. DOES ROTATION CHANGE THE BILL -- if it does, a z is not timing\n")
    lags = [int(round((i + 1) / 9 * n_groups)) for i in range(8)]
    rotations = {lag: C._rotation_index(frame, lag) for lag in lags}
    print(f"    {'feature':<24}{'aligned':>11}{'rotated':>11}{'change':>9}")
    print("    " + "-" * 55)
    for name in [CANDIDATE] + BESIDE:
        vals = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
        pos = C._position(vals, dates)
        aligned = float(np.nanmean(C._mean_by_date(
            np.abs(pos) * full, sorted_codes, n_groups)))
        bills = []
        for lag in lags:
            moved = pos[rotations[lag]]
            moved = moved - C._mean_by_date(moved, sorted_codes, n_groups)[sorted_codes]
            bills.append(float(np.nanmean(C._mean_by_date(
                np.abs(moved) * full, sorted_codes, n_groups))))
        rotated = float(np.mean(bills))
        print(f"    {name:<24}{aligned:>11.6f}{rotated:>11.6f}"
              f"{rotated / aligned - 1.0:>8.1%}")
    print("\n    A change of about a percent means both books pay the same and "
          "the rotation\n    comparison is about TIMING. A large one means a "
          "short-hold z is a cost\n    artefact and has to be thrown away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
