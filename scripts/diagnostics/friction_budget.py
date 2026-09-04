"""How cheap must execution be for a book like this to clear the bar?

The question was put the wrong way round. "What does the broker charge" needs
a broker; "what would the charge have to be" needs only what is already
measured, and it is the number that decides whether another instrument class is
worth collecting at all.

WHAT IS KNOWN, all measured on this project's own data:

    gross Sharpe of the daily book                 2.241   (R22)
    friction, round trip                          0.1094%  (R22)
    annual cost at daily rebalancing                27.6%
    net Sharpe                                     -4.350
    honest threshold for 230 attempts               0.714   (R23)

Those four fix the book's annual volatility without needing to measure it
again: net = (gross_return - cost) / vol and gross_return = gross_Sharpe * vol,
so vol = cost / (gross_Sharpe - net_Sharpe). From the row above that is 4.19%,
and the implied gross annual return is 9.4% -- which is the arithmetic behind
"27.6% of friction against about 10% of gross".

From there the friction BUDGET is a subtraction rather than a search: the cost
a book can pay and still clear a target net Sharpe.

The ranges printed for instrument classes are ESTIMATES, marked as such, and
exist to be replaced by real quotes. What does not need replacing is the
budget itself: it comes from measurement and it is the line those quotes have
to beat.

    python scripts/diagnostics/friction_budget.py
    python scripts/diagnostics/friction_budget.py --instruments 40
"""
from __future__ import annotations

import argparse

# Measured, R22.
GROSS_SHARPE = 2.241
FRICTION_ROUND_TRIP = 0.001094
REBALANCES_PER_YEAR = 252
NET_SHARPE = -4.350
UNIVERSE = 110

# Measured, R23: the honest threshold once the attempts are counted.
BONFERRONI = 0.714

#: Rough round-trip friction by instrument class, in basis points of notional,
#: as an ORDER OF MAGNITUDE to be replaced by real quotes. Equities here is not
#: an estimate -- it is this project's own measurement.
CLASSES = {
    "US equities, $29 median (MEASURED)": (10.9, 10.9),
    "US equities, dearest quarter (MEASURED)": (5.6, 5.6),
    "index futures, liquid": (1.0, 3.0),
    "commodity futures, liquid": (2.0, 6.0),
    "FX majors, spot": (0.5, 2.0),
    "crypto majors, major venue": (5.0, 20.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", type=int, default=UNIVERSE,
                        help="size of the candidate universe")
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[0.0, BONFERRONI, 1.0])
    parser.add_argument("--holds", action="store_true",
                        help="sweep the measured gross Sharpe by holding period")
    args = parser.parse_args()

    annual_cost = FRICTION_ROUND_TRIP * REBALANCES_PER_YEAR
    vol = annual_cost / (GROSS_SHARPE - NET_SHARPE)
    print(f"implied annual volatility of the book   {vol:.2%}")
    print(f"implied gross annual return             {GROSS_SHARPE * vol:.2%}")
    print(f"friction actually paid                  {annual_cost:.1%} a year "
          f"({FRICTION_ROUND_TRIP:.4%} a round trip)\n")

    # A smaller universe carries less breadth: for the same per-name edge the
    # portfolio's Sharpe falls as sqrt(N_small / N_full). That is the price of
    # leaving equities, and it has to be paid out of the friction saved.
    scale = (args.instruments / UNIVERSE) ** 0.5
    gross_there = GROSS_SHARPE * scale
    print(f"on {args.instruments} instruments the same per-name edge gives "
          f"gross Sharpe {gross_there:.3f}")
    print(f"(breadth costs a factor of {1/scale:.2f} against 110 names)\n")

    print(f"{'target net Sharpe':>19}{'affordable cost/year':>22}"
          f"{'round trip, bp':>17}{'vs the 10.9bp paid now':>24}")
    print("-" * 82)
    budgets = {}
    for target in args.targets:
        affordable = (gross_there - target) * vol
        per_trip_bp = affordable / REBALANCES_PER_YEAR * 10_000
        budgets[target] = per_trip_bp
        verdict = ("impossible: the gross edge is smaller than the target"
                   if per_trip_bp <= 0 else f"{10.9 / per_trip_bp:.1f}x cheaper needed")
        print(f"{target:>19.3f}{affordable:>21.1%}{per_trip_bp:>17.2f}{verdict:>24}")

    if args.holds:
        # Gross Sharpe by holding period, measured in R22 on the one feature
        # that survived 1.2. The budget is only a budget while the gross
        # exceeds the target: below that, no execution price helps, because
        # there is nothing left to pay with.
        measured = {1: 2.241, 5: 0.575, 20: 0.205, 40: 0.203, 120: 0.139}
        print(f"\n{'hold':>6}{'gross Sharpe':>14}{'rebalances/yr':>15}"
              f"{'budget for net ' + f'{BONFERRONI:.3f}':>24}{'clears 10.9bp?':>16}")
        print("-" * 75)
        for hold, gross in measured.items():
            per_year = REBALANCES_PER_YEAR / hold
            affordable = (gross - BONFERRONI) * vol
            bp = affordable / per_year * 10_000 if affordable > 0 else 0.0
            shown = f"{bp:.1f} bp" if bp > 0 else "impossible at any price"
            print(f"{hold:>6}{gross:>14.3f}{per_year:>15.1f}{shown:>24}"
                  f"{('YES' if bp >= 10.9 else 'no'):>16}")
        print("\nBelow a gross Sharpe of the target, free execution would not "
              "clear the bar: there is nothing left to pay costs with. "
              "That is the state at every hold beyond one day for this "
              "signal.")

    budget = budgets.get(BONFERRONI)
    if budget and budget > 0:
        print(f"\nSo the line any instrument class has to beat, for a book of "
              f"this shape\nrebalanced daily on {args.instruments} instruments, "
              f"is about {budget:.1f} bp round trip.\n")
        print(f"{'class':<40}{'round trip, bp':>18}{'clears the line?':>18}")
        print("-" * 76)
        for name, (low, high) in CLASSES.items():
            mark = ("YES" if high <= budget else
                    "maybe" if low <= budget else "no")
            print(f"{name:<40}{f'{low:.1f} - {high:.1f}':>18}{mark:>18}")

    print("\nThe class ranges are ESTIMATES and exist to be replaced by quotes.")
    print("The budget is not an estimate: it comes from this project's own")
    print("measurements, and it is what the quotes have to beat.")
    print("\nTwo things this does NOT say. It assumes a future signal with the")
    print("same gross Sharpe, and R25 gives reason to doubt this one is even")
    print("real -- the edge sits in the cheapest, noisiest names. And breadth")
    print("scaling assumes the per-name edge carries across, which is exactly")
    print("what a new instrument class has not been shown to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
