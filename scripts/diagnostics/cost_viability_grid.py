"""What each assumption buys, instead of one number that hides them.

The decision about position size and broker turns on quantities nobody knows
yet: how large the gross edge is, what a broker charges, how much capital
there is. A single answer would be a guess wearing a decimal point. A grid is
honest: it says what follows FROM each assumption, and lets the reader see
which assumptions matter.

Nothing here claims the strategy works. Every figure is a cost, and costs are
knowable in advance -- which is exactly why they can be decided on before the
edge is.

What is measured, not assumed:

    median absolute 1-day move   0.948%    153,852 daily bars, 22 tickers
    median absolute 5-day move   2.204%
    the cost profiles            src/config/targets.yaml

What is varied, because it is not known:

    capital, gross edge per trade, broker fee structure

    python scripts/diagnostics/cost_viability_grid.py
    python scripts/diagnostics/cost_viability_grid.py --positions 10
"""

from __future__ import annotations

import argparse
import io
import sys

import yaml

#: A 5-day horizon held to expiry is roughly 50 round trips a year.
ROUND_TRIPS_PER_YEAR = 50

#: Measured on the batch, 153,852 daily bars across 22 tickers.
MEDIAN_MOVE_5D = 0.02204

CAPITALS = (5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000)
EDGES = (0.0007, 0.0010, 0.0014, 0.0020, 0.0030)


def _profiles() -> dict:
    config = yaml.safe_load(io.open("src/config/targets.yaml", encoding="utf-8"))

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None

    return find(config, "cost_profiles") or {}


def round_trip(notional: float, profile: dict, price: float = 150.0) -> float:
    """Commission both ways plus spread and slippage both ways, as a fraction."""
    friction = 2 * (float(profile.get("spread_pct", 0.0))
                    + float(profile.get("slippage_pct", 0.0)))
    if profile.get("model") == "per_share":
        fee = float(profile["per_share_fee"]) * (notional / price)
        fee = max(fee, float(profile.get("min_fee_per_order", 0.0)))
        fee = min(fee, float(profile.get("max_fee_pct_of_order", 1.0)) * notional)
        return 2 * fee / notional + friction
    return 2 * float(profile.get("commission_pct", 0.0)) + friction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=22,
                        help="names held at once; capital splits equally")
    parser.add_argument("--price", type=float, default=150.0)
    args = parser.parse_args()

    profiles = _profiles()
    if not profiles:
        print("no cost_profiles in src/config/targets.yaml")
        return 1

    print(f"{args.positions} positions, capital split equally, "
          f"share price ${args.price:.0f}\n")

    for name, profile in profiles.items():
        print(f"=== {name} ===")
        print(f"{'capital':>10s} {'per pos.':>9s} {'round trip':>11s} "
              + "".join(f"{e:>9.2%}" for e in EDGES))
        print(f"{'':>10s} {'':>9s} {'':>11s} "
              + "".join(f"{'net/yr':>9s}" for _ in EDGES))
        print("-" * (32 + 9 * len(EDGES)))
        for capital in CAPITALS:
            notional = capital / args.positions
            cost = round_trip(notional, profile, args.price)
            cells = []
            for edge in EDGES:
                net_per_trade = edge - cost
                annual = net_per_trade * ROUND_TRIPS_PER_YEAR
                cells.append(f"{annual:>+8.1%} " if net_per_trade > 0
                             else f"{'—':>9s}")
            print(f"{capital:10,} {notional:9,.0f} {cost:11.3%} " + "".join(cells))
        print()

    # How much of the barrier is the venue, and how much is the market.
    friction_only = {"model": "flat", "commission_pct": 0.0,
                     "spread_pct": 0.0001, "slippage_pct": 0.0001}
    print("=== reference: no commission at all (not a broker, a floor) ===")
    print(f"{'capital':>10s} {'per pos.':>9s} {'round trip':>11s} "
          + "".join(f"{e:>9.2%}" for e in EDGES))
    print("-" * (32 + 9 * len(EDGES)))
    for capital in (5_000, 25_000, 100_000):
        notional = capital / args.positions
        cost = round_trip(notional, friction_only, args.price)
        cells = []
        for edge in EDGES:
            net = (edge - cost) * ROUND_TRIPS_PER_YEAR
            cells.append(f"{net:>+8.1%} " if edge > cost else f"{'—':>9s}")
        print(f"{capital:10,} {notional:9,.0f} {cost:11.3%} " + "".join(cells))
    print()
    print("  That row is spread and slippage alone -- what the MARKET charges,")
    print("  with the broker set to zero. Everything above it that is worse is")
    print("  the venue, and is a choice. Note it does not vary with capital:")
    print("  small accounts are punished by the per-ORDER minimum, not by the")
    print("  market.")
    print()

    print("Reading it:")
    print("  A dash means the cost exceeds the edge: the position loses money")
    print("  before any question of skill. Columns are ASSUMED gross edge per")
    print("  trade; none of them is measured, and the largest one this project")
    print("  has ever measured is 0.14%.")
    print()
    print(f"  Net per year assumes {ROUND_TRIPS_PER_YEAR} round trips, which is a")
    print("  5-day horizon held to expiry. Trading more often multiplies the")
    print("  cost and not the edge.")
    print()
    print(f"  Context: the median absolute 5-day move is {MEDIAN_MOVE_5D:.2%}, so an")
    print("  edge of 0.14% is about a sixteenth of a typical move. That is the")
    print("  scale of thing being fought over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
