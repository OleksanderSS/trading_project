"""Plant an edge of KNOWN size in the REAL panel and ask what the net test returns.

WHY THIS EXISTS. CRITIQUE recommendation B asked for the instrument to be
calibrated on an effect that certainly exists, and named post-earnings drift:

    "If our apparatus cannot see PEAD, the apparatus is broken, not the market."

The apparatus was pointed at PEAD on 2026-09-04 and saw nothing -- gross Sharpe
oscillating around zero, -0.112 / -0.067 / +0.091 / -0.119 (R30), and the SUE
line closed at eight attempts with a best of +0.250 against a noise maximum of
0.440 (R32). By B's own rule that reads "broken apparatus". R30 escaped it with
an ARGUMENT -- PEAD decayed in liquid large caps after the 2000s -- and an
argument is exactly what this project does not accept in place of a number.

So B cannot be closed by another real-data variant. It needs the control that
does not depend on the effect being there: put in an edge whose size we CHOSE,
and see what comes out.

WHAT IS SYNTHETIC AND WHAT IS NOT. Only the feature column. The panel, the
dates, the closes, the forward returns, the friction, the cross-sectional
ranking, the dollar-neutralisation, the phase averaging and the thresholds are
the ones that produced R22, R23 and R28. R9 and R10 planted edges too, but on
SYNTHETIC panels and through stage 7 and the learning path -- neither touched
the instrument that delivered every "nothing survives" verdict of this project.

    plant(rho) = rho * z(forward return) + sqrt(1 - rho^2) * z(noise)

standardised WITHIN each date, so the cross-sectional correlation between the
planted column and the thing it predicts is rho by construction. This column
looks ahead on purpose: it is the definition of a known effect, not a
candidate, and it can never be traded. That is the whole point of a control.

THE GRID IS DECLARED HERE, BEFORE THE RUN.

    rho    0.00 0.01 0.02 0.03 0.05 0.10 0.20
    holds  the six the real run used
    seeds  SEEDS_NULL at rho=0, SEEDS_EDGE elsewhere

WHAT COMES OUT. The smallest planted rho whose net Sharpe clears the bar the
real run had to clear. Below that size the instrument is blind, and every
"zero survivors" in this project is silent about effects under it.

AND WHAT DOES NOT, THOUGH THE FIRST VERSION OF THIS FILE CLAIMED IT WOULD. The
rho=0 rows were meant to measure the standard error of an annualised Sharpe
directly, since SHARPE_SE = 0.193 sits under every threshold the project has
printed and was asserted from sqrt(1/T) rather than measured. They cannot: a
planted column at rho=0 is redrawn on every row, so its book is a fresh random
portfolio each date, and one run already averages thousands of independent
draws. Measured that way the spread came out at 0.018-0.162 -- three to ten
times too tight -- and reading SHARPE_SE off it would have said every threshold
in the project was set several times too high. That is `--mode shuffle`, which
builds the null two honest ways instead, and the rotation is the one to
believe.

    python scripts/diagnostics/what_edge_would_the_net_test_have_seen.py
    python scripts/diagnostics/what_edge_would_the_net_test_have_seen.py --mode shuffle
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import norm  # noqa: E402

from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

#: The instrument itself, imported rather than copied. Every constant that
#: decides an answer -- the seal date, SHARPE_SE, the threshold algebra, the
#: phase averaging -- has to be the SAME OBJECT as the one the real run used,
#: or this calibrates a lookalike. The module guards its own main() behind
#: __main__, so importing it runs nothing.
_spec = importlib.util.spec_from_file_location(
    "net_test", PROJECT_ROOT / "scripts" / "diagnostics" / "net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NET)

#: Sizes of the planted edge, as a cross-sectional information coefficient.
#: Fixed before the run. 0.03-0.06 is where R8's algebra puts the daily frame's
#: minimum detectable IC, so the grid has to straddle it rather than start
#: above it.
RHOS = (0.00, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20)

#: More draws at rho=0 because that row is estimating a SPREAD, not a mean.
SEEDS_NULL = 24
SEEDS_EDGE = 6

#: The attempt count of the run being calibrated: 235 features x 6 holds
#: (R28). The bar a planted edge must clear is the bar the real features had
#: to clear, not a fresh single-test bar -- using the latter here would flatter
#: the instrument by exactly the multiplicity correction the project spent a
#: week installing.
REAL_ATTEMPTS = 1410


def _standardise(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Zero mean, unit deviation WITHIN each date.

    Cross-sectional, because the book is cross-sectional: a column standardised
    over the whole panel would carry the market's own time variation into the
    plant and the planted rho would no longer be the number it claims to be.
    """
    series = pd.Series(values)
    grouped = series.groupby(dates)
    centred = series - grouped.transform("mean")
    spread = grouped.transform("std")
    out = centred / spread.replace(0.0, np.nan)
    return np.nan_to_num(out.to_numpy())


def _position(column: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """The identical book the real run builds, on a supplied column."""
    position = np.sign(
        pd.Series(column).groupby(dates).rank(pct=True).to_numpy() - 0.5)
    position = np.nan_to_num(position)
    return position - (pd.Series(position).groupby(dates)
                       .transform("mean").to_numpy())


#: Rotation and the fast date-mean live in the INSTRUMENT now, not here. They
#: were written in this file on 2026-09-06 and moved into
#: `net_test_every_survivor.py` the same day, because a null that decides how a
#: verdict reads belongs to the thing that issues the verdict -- a diagnostic
#: holding its own copy is how two definitions of the seal happened (R45).
_mean_by_date = NET._mean_by_date
_rotation_index = NET._rotation_index


def _sharpe_given_position(position: np.ndarray, dates: np.ndarray,
                           forward: np.ndarray, friction: np.ndarray,
                           hold: int, codes: np.ndarray | None = None,
                           groups: int = 0) -> float:
    net = position * forward - np.abs(position) * friction
    if codes is None:
        by_date = (pd.DataFrame({"d": dates, "net": net})
                   .groupby("d")["net"].mean().sort_index().to_numpy())
    else:
        by_date = _mean_by_date(net, codes, groups)
    mean, _ = NET._sharpe_all_phases(by_date, hold)
    return mean


def _book_sharpe(column: np.ndarray, dates: np.ndarray, forward: np.ndarray,
                 friction: np.ndarray, hold: int) -> float:
    return _sharpe_given_position(
        _position(column, dates), dates, forward, friction, hold)


def _shuffle_within_date(forward: np.ndarray, date_codes: np.ndarray,
                         in_date_order: np.ndarray,
                         generator: np.random.Generator) -> np.ndarray:
    """Re-deal the forward returns among the names present on each date.

    BETTER THAN THE PLANTED rho=0, AND STILL NOT RIGHT. It keeps the real
    position -- the real feature's persistence and its tilt toward whatever
    kinds of name it likes -- along with the friction attached to each name and
    the cross-sectional distribution of that date's returns, and destroys only
    the association between them.

    What it also destroys is the AUTOCORRELATION of the book's daily P&L, by
    re-dealing independently on every date. The standard error of a Sharpe on
    an autocorrelated series is much larger than on white noise, so this null
    comes out too tight and flatters anything measured against it: on
    MAX_DRAWDOWN_1d at a 20-day hold it gives 0.048 where the rotation gives
    0.142. Kept because the gap between the two IS the measurement of what
    re-dealing throws away.
    """
    receiver = np.lexsort((generator.random(forward.size), date_codes))
    out = np.empty_like(forward)
    out[in_date_order] = forward[receiver]
    return out


def _shuffled_null(args, frame, order, dates, friction, forwards) -> int:
    """What the book scores on REAL columns that have been told nothing.

    SHARPE_SE = 0.193 sits under every threshold this project has printed. It
    comes from sqrt(1/T) -- algebra about an i.i.d. series -- and has never met
    the panel it judges. Two nulls are built here rather than one, because
    they bracket the answer: the re-deal is too tight by construction and the
    rotation keeps the persistence that widens it. If they disagree, the
    rotation is the one to believe, and the gap between them is the size of
    the thing the re-deal throws away.
    """
    roles = pd.read_csv(NET.ROLES)
    varying = roles[roles["varies"] > 0.5]["feature"].tolist()
    varying = [n for n in varying if n != "close"]
    if args.only:
        missing = [n for n in args.only if n not in varying]
        if missing:
            print(f"refusing: {', '.join(missing)} is not a column with "
                  f"cross-sectional variation, so the real run never measured "
                  f"it and there is no verdict to check.")
            return 1
        chosen = list(args.only)
    elif args.all:
        chosen = list(varying)
    else:
        step = max(1, len(varying) // args.features)
        chosen = varying[::step][:args.features]
    print(f"null on {len(chosen)} real columns, {args.shuffles} rotations each, "
          f"holds {args.holds}")

    # DECLARED BEFORE THE RUN, from the attempts this run will actually make.
    # A z-score is a Sharpe divided by its own measured spread, so the bar is
    # the same multiplicity correction in sigma units -- and it is stated here
    # rather than after the table, because a bar chosen once the answers are
    # visible is not a bar.
    attempts = len(chosen) * len(args.holds)
    z_bonferroni = float(norm.ppf(1.0 - 0.025 / max(attempts, 1)))
    log_n = math.log(max(attempts, 2))
    root = math.sqrt(2.0 * log_n)
    z_noise = max(root - (math.log(log_n) + math.log(4.0 * math.pi)) / (2.0 * root),
                  float(norm.ppf(0.95)))
    print(f"attempts {attempts}  ->  expected maximum of noise z={z_noise:.2f}, "
          f"Bonferroni family-wise 5% z={z_bonferroni:.2f}\n")

    date_codes = pd.factorize(dates)[0]
    in_date_order = np.lexsort((np.arange(dates.size), date_codes))
    # Sorted codes, so a bincount lands in DATE order without a further sort.
    sorted_codes, uniques = pd.factorize(dates, sort=True)
    n_groups = len(uniques)

    # Lags spread evenly over the record rather than drawn at random: a lag
    # near zero leaves the book almost aligned and would quietly pull the null
    # toward the real answer.
    n_dates = int(pd.Series(dates).nunique())
    lags = [int(round((i + 1) / (args.shuffles + 1) * n_dates))
            for i in range(args.shuffles)]
    rotations = {lag: _rotation_index(frame, lag) for lag in lags}
    date_mean = lambda v: _mean_by_date(v, sorted_codes, n_groups)[sorted_codes]

    header = (f"{'feature':<32}"
              + "".join(f"{'z' + str(h):>8}" for h in args.holds)
              + f"{'best z':>8}{'REAL':>8}{'null':>8}{'sd':>7}")
    print(header)
    print("-" * len(header))

    rotate_sd: dict[int, list[float]] = {h: [] for h in args.holds}
    redeal_sd: dict[int, list[float]] = {h: [] for h in args.holds}
    verdicts = []
    for start in range(0, len(chosen), NET.CHUNK):
        block = chosen[start:start + NET.CHUNK]
        loaded = pd.read_parquet(NET.BATCH / "features.parquet",
                                 columns=list(dict.fromkeys(block)))
        for name in block:
            values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
            if pd.Series(values).notna().sum() < 10_000:
                print(f"{name:<32}   skipped: fewer than 10,000 usable rows")
                continue
            position = _position(values, dates)
            if args.skip_bars:
                # Yesterday's book, today's returns. `_rotation_index(frame, 1)`
                # maps each row to the previous row of the SAME name, which is
                # what a one-bar delay is; the roll wraps 110 rows out of
                # 623,398, which cannot carry a result.
                for _ in range(args.skip_bars):
                    position = position[_rotation_index(frame, 1)]
                position = position - date_mean(position)
            turned = {}
            for lag in lags:
                moved = position[rotations[lag]]
                # Re-neutralise: after the shift the names present on a date
                # are not the ones the original weights balanced.
                turned[lag] = moved - date_mean(moved)
            per_hold = {}
            for hold in args.holds:
                real = _sharpe_given_position(
                    position, dates, forwards[hold], friction, hold,
                    sorted_codes, n_groups)
                if not args.no_redeal:
                    deals = [_sharpe_given_position(
                                position, dates,
                                _shuffle_within_date(
                                    forwards[hold], date_codes, in_date_order,
                                    np.random.default_rng(9_000 + i)),
                                friction, hold, sorted_codes, n_groups)
                             for i in range(args.shuffles)]
                    redeal_sd[hold].append(float(np.nanstd(deals, ddof=1)))
                turns = [_sharpe_given_position(
                             turned[lag], dates, forwards[hold], friction,
                             hold, sorted_codes, n_groups)
                         for lag in lags]
                r_mean = float(np.nanmean(turns))
                r_sd = float(np.nanstd(turns, ddof=1))
                rotate_sd[hold].append(r_sd)
                z = (real - r_mean) / r_sd if r_sd > 0 else float("nan")
                per_hold[hold] = (real, r_mean, r_sd, z)
                verdicts.append((name, hold, real, r_mean, r_sd, z))
            best = max(per_hold,
                       key=lambda h: (per_hold[h][3]
                                      if np.isfinite(per_hold[h][3]) else -99))
            real, r_mean, r_sd, z = per_hold[best]
            print(f"{name:<32}"
                  + "".join(f"{per_hold[h][3]:>8.2f}" for h in args.holds)
                  + f"{z:>8.2f}{real:>8.3f}{r_mean:>8.3f}{r_sd:>7.3f}",
                  flush=True)
            # With a handful of columns the compact row hides the thing that
            # decides whether a finding is money: REAL at EVERY hold, not just
            # at the best-z one. A sweep cannot print this without becoming
            # unreadable; a shortlist must.
            if len(chosen) <= 10:
                for h in args.holds:
                    h_real, h_mean, h_sd, h_z = per_hold[h]
                    verdict = ("INFORMATION AND MONEY" if h_z >= z_bonferroni
                               and h_real > 0 else
                               "information, no money" if h_z >= z_bonferroni
                               else "")
                    print(f"    hold {h:>3}   REAL {h_real:>+7.3f}   "
                          f"null {h_mean:>+7.3f}   sd {h_sd:>5.3f}   "
                          f"z {h_z:>+6.2f}   {verdict}")
        del loaded

    print("\n" + "=" * len(header))
    print("The spread SHARPE_SE claims to describe, measured two ways:\n")
    for hold in args.holds:
        if not rotate_sd[hold]:
            continue
        deal = (f"re-deal {np.median(redeal_sd[hold]):.3f}   "
                if redeal_sd[hold] else "")
        print(f"    hold {hold:>3}: {deal}"
              f"ROTATION {np.median(rotate_sd[hold]):.3f}   "
              f"({np.median(rotate_sd[hold]) / NET.SHARPE_SE:.2f}x the "
              f"asserted {NET.SHARPE_SE})")
    every_rotation = [s for hold in args.holds for s in rotate_sd[hold]]
    if every_rotation:
        overall = float(np.median(every_rotation))
        print(f"\n    rotation median {overall:.3f}, asserted {NET.SHARPE_SE:.3f}"
              f"  -> thresholds are "
              f"{'too LOW' if overall > NET.SHARPE_SE else 'too HIGH'} by "
              f"{max(overall, NET.SHARPE_SE) / min(overall, NET.SHARPE_SE):.1f}x")
        print("\n    A threshold set from an ASSERTED spread is a free "
              "parameter wearing a formula.\n    The rotation column is what "
              "this panel actually does when the column knows nothing.")
    if not verdicts:
        return 0

    ranked = sorted((row for row in verdicts if np.isfinite(row[5])),
                    key=lambda row: -row[5])
    print("\n" + "=" * len(header))
    print(f"THE QUESTION THIS RUN WAS FOR: does any column know something "
          f"about WHEN,\nrather than simply holding a tilt? A column that does "
          f"scores far above its own\nrotated null. Bar declared before the "
          f"run: z={z_bonferroni:.2f} (Bonferroni on {attempts} attempts).\n")
    print(f"{'feature':<32}{'hold':>6}{'REAL':>9}{'null':>9}{'sd':>8}{'z':>8}")
    print("-" * 72)
    for name, hold, real, r_mean, r_sd, z in ranked[:12]:
        print(f"{name:<32}{hold:>6}{real:>9.3f}{r_mean:>9.3f}"
              f"{r_sd:>8.3f}{z:>8.2f}")

    clearing = [row for row in ranked if row[5] >= z_bonferroni]
    above_noise = [row for row in ranked if row[5] >= z_noise]
    print(f"\n    clear Bonferroni z>={z_bonferroni:.2f}:      "
          f"{len(clearing)} of {len(ranked)}")
    print(f"    above the noise maximum z>={z_noise:.2f}:  "
          f"{len(above_noise)} of {len(ranked)}")
    if not above_noise:
        print("\n    NOTHING in this batch knows about timing. Every score it "
              "has is a tilt that\n    survives having its dates shuffled -- "
              "which is what a factor is, and what a\n    prediction is not.")
    elif not clearing:
        print("\n    Above noise, below Bonferroni. Worth stating as a measured "
              "number, not as a\n    candidate -- and NOT worth another variant "
              "here: that spends attempts on it.")
    else:
        print("\n    A column beats its own rotated null past the declared bar. "
              "This is the first\n    time that has happened. The next step is "
              "NOT another variant: it is one\n    pre-registered confirmation "
              "on the sealed period, which is what the seal is for.")
    # A book can beat its own null on information and still lose money, which
    # is a different question and has to be printed as one.
    if clearing:
        earners = [row for row in clearing if row[2] > 0]
        print(f"\n    of those, NET POSITIVE after costs: {len(earners)}. "
              f"Information and money are\n    separate tests and a column has "
              f"to pass both.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", type=int, nargs="+",
                        default=[1, 5, 20, 40, 60, 120])
    parser.add_argument("--rhos", type=float, nargs="+", default=list(RHOS))
    parser.add_argument("--mode", choices=["plant", "shuffle"], default="plant",
                        help="plant: how big an edge must be to be seen. "
                             "shuffle: what the book scores on REAL feature "
                             "columns whose returns have been re-dealt, which "
                             "is the null SHARPE_SE claims to describe.")
    parser.add_argument("--features", type=int, default=6,
                        help="how many real columns to build the shuffled "
                             "null on, in shuffle mode.")
    parser.add_argument("--shuffles", type=int, default=24)
    parser.add_argument("--no-commission", action="store_true",
                        help="price the book under a commission-free broker: "
                             "per-share fee and minimum set to zero, spread "
                             "and slippage untouched. This is NOT a parameter "
                             "being tuned -- it is a second BROKER, and the "
                             "measured split says which of the two the answer "
                             "depends on: commission is 63%% of the 10.94 bp "
                             "round trip and 80%% of it under $20 a share. "
                             "The honest caveat travels with it: a "
                             "zero-commission broker is paid through order "
                             "flow, so the 1 bp spread and 1 bp slippage this "
                             "keeps are the OPTIMISTIC half of the trade.")
    parser.add_argument("--skip-bars", type=int, default=0,
                        help="bars between the column being knowable and the "
                             "position earning. 0 assumes you compute all 110 "
                             "names' cross-section AT the close and trade at "
                             "that same close -- zero latency. `peer_return` "
                             "and `market_return` are same-bar leave-one-out, "
                             "so that assumption is doing real work and has to "
                             "be priced: R32 lost a +0.642 headline to exactly "
                             "this check. 1 is the conservative reading.")
    parser.add_argument("--all", action="store_true",
                        help="every column with cross-sectional variation -- "
                             "the same 235 the real run measured.")
    parser.add_argument("--no-redeal", action="store_true",
                        help="skip the within-date re-deal. It is known to be "
                             "too tight (R50) and costs as much as the "
                             "rotation, so a full sweep does not pay for it.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="name the columns to build the null on, instead "
                             "of sampling the varying list. The point of the "
                             "null is that it belongs to ONE column -- its own "
                             "persistence, its own turnover -- so a verdict "
                             "about a named feature has to be measured on that "
                             "feature and not on a stand-in.")
    args = parser.parse_args()

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    if args.no_commission:
        costs = dict(costs, per_share_fee=0.0, min_fee_per_order=0.0)
        print("COSTS: commission-free broker -- spread and slippage only. "
              "Every number below\n       is under a DIFFERENT trading "
              "arrangement than the rest of the project's,\n       and may "
              "not be compared with one measured at full cost.\n")

    frame, order = NET._panel([])
    dates = frame["datetime"].to_numpy()
    friction = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].nunique():,} dates, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}")
    print(f"seal:  nothing at or after {NET.SEALED.date()} is touched\n")

    forwards = {
        hold: (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy())
        for hold in args.holds
    }
    if args.mode == "shuffle":
        return _shuffled_null(args, frame, order, dates, friction, forwards)

    z_forward = {hold: _standardise(forwards[hold], dates) for hold in args.holds}

    bonferroni, noise_max = NET._thresholds(REAL_ATTEMPTS)
    print(f"the bar the real run had to clear ({REAL_ATTEMPTS} attempts):")
    print(f"    expected maximum of noise                {noise_max:.3f}")
    print(f"    Bonferroni family-wise 5%                {bonferroni:.3f}")
    print(f"    best net Sharpe the real run found       +0.586  "
          f"(VOLATILITY_50_1d at hold 60, R22/R28)\n")

    header = (f"{'planted IC':>11}{'seeds':>7}"
              + "".join(f"{'h' + str(h):>9}" for h in args.holds)
              + f"{'best':>9}{'sd(best)':>10}")
    print(header)
    print("-" * len(header))

    table: dict[float, dict[int, list[float]]] = {}
    for rho in args.rhos:
        seeds = SEEDS_NULL if rho == 0.0 else SEEDS_EDGE
        per_hold: dict[int, list[float]] = {h: [] for h in args.holds}
        for seed in range(seeds):
            generator = np.random.default_rng(1_000 + seed)
            noise = _standardise(generator.standard_normal(len(frame)), dates)
            for hold in args.holds:
                column = rho * z_forward[hold] + np.sqrt(1.0 - rho ** 2) * noise
                per_hold[hold].append(
                    _book_sharpe(column, dates, forwards[hold], friction, hold))
        table[rho] = per_hold
        means = {h: float(np.nanmean(per_hold[h])) for h in args.holds}
        best = max(means, key=lambda h: means[h])
        print(f"{rho:>11.2f}{seeds:>7}"
              + "".join(f"{means[h]:>9.3f}" for h in args.holds)
              + f"{means[best]:>9.3f}{float(np.nanstd(per_hold[best])):>10.3f}",
              flush=True)

    print("\n" + "=" * len(header))

    # ANSWER 2 FIRST, because it decides how to read answer 1: if the measured
    # spread is not 0.193, every threshold in the project is the wrong height
    # and the floor below is being compared with a mis-set bar.
    null = table.get(0.0)
    if null:
        for hold in args.holds:
            draws = np.asarray(null[hold], dtype=float)
            print(f"rho=0, hold {hold:>3}: mean {np.nanmean(draws):+.3f}, "
                  f"sd {np.nanstd(draws, ddof=1):.3f}  "
                  f"(SHARPE_SE asserted {NET.SHARPE_SE:.3f})")
        print()

    # ANSWER 1: the smallest planted edge that clears the real bar anywhere.
    cleared = [rho for rho in sorted(table)
               if max(float(np.nanmean(table[rho][h])) for h in args.holds)
               >= bonferroni]
    if not cleared:
        print(f"NO planted edge up to IC {max(table):.2f} clears "
              f"{bonferroni:.3f}. The instrument could not have returned a "
              f"verdict on\nany effect this universe is capable of carrying, "
              f"and every 'nothing survives' in\nthis project is a statement "
              f"about the instrument.")
    else:
        floor = min(cleared)
        print(f"DETECTION FLOOR: planted IC {floor:.2f} is the smallest that "
              f"clears {bonferroni:.3f}.")
        print(f"Below IC {floor:.2f} this instrument returns 'nothing "
              f"survives' whether or not something is\nthere. The strongest "
              f"effect ever found in this project is IC +0.0388 "
              f"(fund_debt_to_equity, R8).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
