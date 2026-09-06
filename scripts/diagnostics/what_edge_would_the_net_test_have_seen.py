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


def _sharpe_given_position(position: np.ndarray, dates: np.ndarray,
                           forward: np.ndarray, friction: np.ndarray,
                           hold: int) -> float:
    net = position * forward - np.abs(position) * friction
    by_date = pd.DataFrame({"d": dates, "net": net}).groupby("d")["net"].mean()
    mean, _ = NET._sharpe_all_phases(by_date.sort_index().to_numpy(), hold)
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


def _rotation_index(frame: pd.DataFrame, lag: int) -> np.ndarray:
    """Row indices that shift each name's position series `lag` bars later.

    WHY ROTATION AND NOT THE RE-DEAL ABOVE. Re-dealing within a date makes the
    book's daily P&L independent across dates. A real feature holds a similar
    position for months, so its P&L is autocorrelated, and the standard error
    of a Sharpe on an autocorrelated series is much larger than on white noise.
    The re-deal therefore destroys the very thing that widens the null and
    returns a spread that is too tight -- flattering any candidate measured
    against it.

    Rotation keeps the position series ENTIRE: the same persistence, the same
    tilt toward whatever kinds of name the column likes, the same friction. It
    breaks only the alignment in time with the returns. That is the one thing a
    genuine edge needs and a spurious one does not.

    `frame` is sorted by ticker then datetime, so each name is a contiguous
    block in date order and the roll is a roll within that block.
    """
    rows = np.arange(len(frame))
    starts = frame.groupby("ticker", sort=False).size().cumsum().to_numpy()
    starts = np.concatenate([[0], starts])
    out = np.empty_like(rows)
    for begin, end in zip(starts[:-1], starts[1:]):
        block = rows[begin:end]
        out[begin:end] = np.roll(block, lag % max(len(block), 1))
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
    else:
        step = max(1, len(varying) // args.features)
        chosen = varying[::step][:args.features]
    print(f"null on {len(chosen)} real columns, {args.shuffles} draws each: "
          f"{', '.join(chosen)}\n")

    date_codes = pd.factorize(dates)[0]
    in_date_order = np.lexsort((np.arange(dates.size), date_codes))
    loaded = pd.read_parquet(NET.BATCH / "features.parquet", columns=chosen)

    # Lags spread evenly over the record rather than drawn at random: a lag
    # near zero leaves the book almost aligned and would quietly pull the null
    # toward the real answer.
    n_dates = int(pd.Series(dates).nunique())
    lags = [int(round((i + 1) / (args.shuffles + 1) * n_dates))
            for i in range(args.shuffles)]
    rotations = {lag: _rotation_index(frame, lag) for lag in lags}
    date_mean = lambda v: (pd.Series(v).groupby(dates).transform("mean")
                           .to_numpy())

    header = (f"{'feature':<28}{'hold':>5}{'REAL':>8}"
              f"{'redeal sd':>11}{'ROTATE mean':>13}{'ROTATE sd':>11}"
              f"{'z(rotate)':>11}")
    print(header)
    print("-" * len(header))

    rotate_sd: dict[int, list[float]] = {h: [] for h in args.holds}
    redeal_sd: dict[int, list[float]] = {h: [] for h in args.holds}
    verdicts = []
    for name in chosen:
        values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
        if pd.Series(values).notna().sum() < 10_000:
            print(f"{name:<28}   skipped: fewer than 10,000 usable rows")
            continue
        position = _position(values, dates)
        turned = {}
        for lag in lags:
            moved = position[rotations[lag]]
            # Re-neutralise: after the shift the names present on a date are
            # not the ones the original weights balanced.
            turned[lag] = moved - date_mean(moved)
        for hold in args.holds:
            real = _sharpe_given_position(
                position, dates, forwards[hold], friction, hold)
            deals = [_sharpe_given_position(
                        position, dates,
                        _shuffle_within_date(forwards[hold], date_codes,
                                             in_date_order,
                                             np.random.default_rng(9_000 + i)),
                        friction, hold)
                     for i in range(args.shuffles)]
            turns = [_sharpe_given_position(turned[lag], dates,
                                            forwards[hold], friction, hold)
                     for lag in lags]
            d_sd = float(np.nanstd(deals, ddof=1))
            r_mean = float(np.nanmean(turns))
            r_sd = float(np.nanstd(turns, ddof=1))
            redeal_sd[hold].append(d_sd)
            rotate_sd[hold].append(r_sd)
            z = (real - r_mean) / r_sd if r_sd > 0 else float("nan")
            verdicts.append((name, hold, real, r_mean, r_sd, z))
            print(f"{name:<28}{hold:>5}{real:>8.3f}{d_sd:>11.3f}"
                  f"{r_mean:>13.3f}{r_sd:>11.3f}{z:>11.2f}", flush=True)

    print("\n" + "=" * len(header))
    print("The spread SHARPE_SE claims to describe, measured two ways:\n")
    for hold in args.holds:
        if not rotate_sd[hold]:
            continue
        print(f"    hold {hold:>3}: re-deal {np.median(redeal_sd[hold]):.3f}   "
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
    if verdicts:
        best = max(verdicts, key=lambda row: row[5] if np.isfinite(row[5]) else -9)
        print(f"\n    strongest of these columns against its own rotated null: "
              f"{best[0]} at hold {best[1]}, z={best[5]:+.2f}")
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
