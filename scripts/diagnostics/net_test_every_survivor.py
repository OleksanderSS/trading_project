"""The net test, applied to every feature -- not just the ones that pass 1.2.

R22 ran the holding sweep on ONE feature, the single one whose verdict was
"survives, worth testing", and found it untradeable at every turnover. That
closes the question for that feature and not for the report.

Forty-six features passed Benjamini-Hochberg at 5% FDR. Their verdicts were
"survives but tiny", "faded", "labels the name" -- all reasons to doubt, none
of them "cannot be traded", because until R22 nobody had asked about cost. A
feature with a weaker IC but a slower decay would beat the winner at a longer
hold, and the ranking in 1.2 cannot see that: it ranks by strength at one bar,
not by what survives friction.

So the question this answers is not "is there a better feature" but "does the
conclusion depend on which feature we picked". If all are negative net at every
horizon, the finding is about the COST STRUCTURE and the universe, and looking
for more features in the same batch is looking in the wrong place.

WHY THE UNIVERSE WIDENED ON 2026-09-04. The paragraph above named the defect
and then ran the 46 anyway, which reproduced it: those 46 were selected by
Benjamini-Hochberg against `target_return_1d`, a ONE-DAY-AHEAD target. A
feature that says nothing about tomorrow and something about the next quarter
fails that screen first and is never seen here. Since the arithmetic that
motivates long holds is precisely that friction stops binding when you trade
rarely, screening candidates on a one-bar target is selecting against the only
thing that could work:

    hold   cost/yr   gross Sharpe needed to clear 0.714 net
       1     27.6%                   7.31
      20      1.4%                   1.04
     120      0.2%                   0.77

At 120 days the required gross Sharpe is almost the threshold itself -- the
cost problem has gone, in the equities already held. So `--universe varying`
runs all 235 features with real cross-sectional variation, and is now the
default. `--universe fdr` restores the old 46 for comparison with R23.

TWO OTHER CORRECTIONS MADE AT THE SAME TIME.

    The thresholds are computed from the attempts actually made rather than
    stated as constants for 230. Hardcoding a multiplicity correction next to
    a `--holds` flag that changes the multiplicity is the same shape as
    `family_size` being declared in one file and verified in none.

    The Sharpe is averaged over every phase of the non-overlapping sampling
    instead of being read off offset zero. Taking `[::hold]` from the first
    date is one arbitrary alignment out of `hold`, and at a 120-day hold that
    is 57 observations chosen by an accident of where the data starts. The
    spread across alignments is reported: a result that depends on the phase
    is not a result.

THE SECOND VERDICT, ADDED 2026-09-06, AND IT CHANGES HOW THE FIRST READS.

Until then this script asked one question -- did the book beat a bar -- and that
question cannot tell an EDGE from a TILT. A column whose ranking barely moves
from year to year holds nearly the same names whatever the date, and collects
whatever those names pay. Its net Sharpe is real money and says nothing about
prediction.

Measured that day on this script's own headline: `VOLATILITY_50_1d` at hold 60,
+0.586, the best result the project had held for seven days. With its positions
shifted to random dates it scored +0.559. The contribution of knowing WHEN was
+0.027 -- 0.41 of its own standard deviations. It was the low-volatility tilt
(CLAIMS R51).

So every column is now also scored against its OWN null, built by rotating its
positions in time: same book, same persistence, same taste in names, same
friction, only the alignment broken. `z` is how far the real book stands above
that null. A result needs BOTH -- money above the bar and z above the bar --
and the summary now counts the three cases separately, including the one that
was invisible before: net positive but not above its own null.

Two limits of that null, stated rather than left to be discovered. It is built
at the best-NET horizon only unless `--rotate-all-holds` is passed, so a column
whose money is at one horizon and whose information is at another goes unseen
-- which is the reversal family of R52. And twelve rotations is a screen, not a
quotable z: a shortlist gets re-measured by
`what_edge_would_the_net_test_have_seen.py --mode shuffle`.

Both legs pay the friction, the Sharpe is the portfolio's, and holding periods
are sampled without overlap. The sealed period is untouched.

    python scripts/diagnostics/net_test_every_survivor.py
    python scripts/diagnostics/net_test_every_survivor.py --universe fdr
    python scripts/diagnostics/net_test_every_survivor.py --holds 1 20 120
    python scripts/diagnostics/net_test_every_survivor.py --rotate-all-holds
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import norm  # noqa: E402

from src.data.universe_membership import OPPONENT_CAVEAT  # noqa: E402
from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
from src.pipeline.sealed_period import SEAL_START  # noqa: E402

#: Imported, never restated. Eight diagnostics each kept their own copy of this
#: date until 2026-09-04. The policy in docs/SEALED_HOLDOUT.md says moving the
#: seal EARLIER is "always safe" -- with eight copies it would have been safe in
#: one file and silently ignored in the other seven, which is the duplication
#: family this codebase's defects come from.
SEALED = SEAL_START

#: The standard error of an annualised Sharpe over T years is about
#: sqrt(1/T); over the 27 explorable years that is 0.193.
#:
#: TWO of those is NOT enough here, and the first version of this script said
#: it was. A threshold chosen without counting the attempts is exactly the
#: defect the promotion gate spent a week having removed (CLAIMS R11, R17),
#: reproduced in the script that was meant to judge its output.
SHARPE_SE = 0.193

#: How many feature columns to hold in memory at once. 235 columns over 1.09M
#: rows is about 2GB as float64; there is no reason to pay it.
CHUNK = 40

#: A column enters the sweep if it varies across names on more than this share
#: of dates. It was written as a bare `> 0.5` in two places until 2026-09-06,
#: which made it invisible -- and it had been silently deciding an outcome: the
#: `wiki_*` family, the ONLY non-price attention family with any cross-sectional
#: variation at all (sentiment, news, keyword and cftc are 0.000 on every
#: column), sits at 0.295 and so was never measured here. Its verdict came from
#: the roles catalogue and never from this instrument, and the reason was a
#: threshold rather than a measurement. Lowering it is a real decision -- a
#: column that varies on a third of dates is a column that holds NO position on
#: the other two thirds -- so it is a flag with a default, not a constant.
MIN_VARIES = 0.5


def _thresholds(attempts: int) -> tuple[float, float]:
    """Family-wise 5% by Bonferroni, and the expected maximum of pure noise.

    Both in Sharpe units. Both assume the attempts are independent, which they
    are NOT -- five holds of one feature are nearly the same test, and the
    features are correlated with each other. So these are conservative: a real
    effect can be rejected by them. That is the direction to err in, and it is
    stated rather than quietly corrected for, because any correction would be
    a free parameter chosen after seeing the answer.
    """
    bonferroni = float(norm.ppf(1.0 - 0.025 / attempts)) * SHARPE_SE
    if attempts <= 1:
        # One pre-registered test: no correction, and the "expected maximum of
        # one draw" is just the one-sided 5% point. Guarded rather than left to
        # crash on log(1), because a family of one is not a corner case here --
        # it is the whole point of the sealed period, where a single hypothesis
        # formed in the open data is tested once.
        return float(norm.ppf(0.975)) * SHARPE_SE, float(norm.ppf(0.95)) * SHARPE_SE
    log_n = math.log(attempts)
    root = math.sqrt(2.0 * log_n)
    noise_max = (root - (math.log(log_n) + math.log(4.0 * math.pi)) / (2.0 * root))
    # The Gumbel approximation to the expected maximum is asymptotic and goes
    # WRONG below a few dozen draws: at 6 attempts it returns 1.07 sigma, which
    # is less than the 1.645 a SINGLE draw exceeds 5% of the time. A maximum
    # over more draws cannot be smaller than that, so the single-draw point is
    # the floor. Caught by running _thresholds(1) and _thresholds(6) after
    # guarding the log(1) crash -- the guard is what made this defect visible.
    floor = float(norm.ppf(0.95))
    return bonferroni, max(noise_max, floor) * SHARPE_SE


def _sharpe(series: np.ndarray, per_year: float) -> float:
    usable = series[np.isfinite(series)]
    if usable.size < 30 or usable.std() <= 0:
        return float("nan")
    return float(usable.mean() / usable.std() * np.sqrt(per_year))


def _sharpe_all_phases(by_date: np.ndarray, hold: int) -> tuple[float, float]:
    """Mean and spread of the Sharpe over every non-overlapping alignment.

    `by_date[::hold]` is one of `hold` equally valid samplings. Reporting the
    first one makes the answer depend on which date the panel happens to begin.
    """
    per_year = 252.0 / hold
    values = [_sharpe(by_date[phase::hold], per_year) for phase in range(hold)]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


#: How many rotations build each column's own null. Twelve is a screen: the
#: spread of twelve draws carries about 20% relative error, which is fine for
#: a z compared against a bar near four and NOT fine for quoting a z to two
#: decimals. A shortlist gets re-measured with more.
ROTATIONS = 12


def _mean_by_date(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    """Mean per date, NaN-skipping, in date order.

    What `groupby(dates).mean().sort_index()` returns, by bincount. Speed only,
    and it is not a nicety: the rotated null multiplies the number of these
    calls by thirteen, and the pandas path turned a half-hour run into a
    four-hour one.
    """
    finite = np.isfinite(values)
    sums = np.bincount(codes, weights=np.where(finite, values, 0.0),
                       minlength=groups)
    counts = np.bincount(codes, weights=finite.astype(float), minlength=groups)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)


def _position(column: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """The book: cross-sectional rank, signed, then made dollar-neutral.

    THIS IS THE DEFINITION OF THE BOOK and it lives here, in the instrument,
    because until 2026-09-07 it lived in TWO places -- inline in the loop below
    and again in `what_edge_would_the_net_test_have_seen.py`. Two copies of the
    thing every verdict is computed on is the duplication family that gave the
    seal two definitions (R45); a third script needing it is what made the
    second copy visible.

    DOLLAR-NEUTRAL, OR THE ANSWER IS THE MARKET. `sign(rank - 0.5)` on a column
    with heavy ties gives +1 to EVERYONE: pandas ranks ties by their average, so
    a binary flag that is 98.9% one value ranks near 0.5+ for every name and the
    "long/short book" is long everything. Measured 2026-09-04, seven features
    cleared Bonferroni at ~1.00 net while the constant opponent -- buy every
    name, rebalance on the same clock, pay the same friction -- scored 1.018 at
    a 60-day hold, and every one of the seven WAS that opponent to three
    decimals.

    Subtracting the per-date mean removes exactly that exposure and leaves a
    degenerate column holding no position at all, which is the honest answer for
    a column that says nothing about which name. Without it this script has no
    opponent ladder and compares against zero -- the defect the promotion gate
    spent a week having removed (CLAIMS R11, R17).
    """
    signed = np.sign(
        pd.Series(column).groupby(dates).rank(pct=True).to_numpy() - 0.5)
    signed = np.nan_to_num(signed)
    return signed - (pd.Series(signed).groupby(dates)
                     .transform("mean").to_numpy())


def _beta_on(book: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    """Beta and correlation of a book's P&L against the constant opponent.

    DOLLAR-NEUTRAL IS NOT MARKET-NEUTRAL, and this line is here because the
    difference went unmeasured for a fortnight. Subtracting the per-date mean
    removes the LEVEL of the cross-section; it leaves the beta. Measured
    2026-09-07 over 1,404 books: a third carry |correlation| above 0.2, and the
    project's headline result -- VOLATILITY_50_1d at +0.586 -- was 0.746
    correlated with simply owning the names and hedges to -0.260 (CLAIMS R61).

    Printed beside the net Sharpe rather than computed on request, for the same
    reason the rotated null is: a check that has to be remembered is a check
    that will one day be skipped.
    """
    usable = np.isfinite(book) & np.isfinite(market)
    if usable.sum() < 60 or market[usable].std() <= 0 or book[usable].std() <= 0:
        return float("nan"), float("nan")
    # ddof MATCHED ON BOTH SIDES. `np.cov` defaults to ddof=1 and `np.var` to
    # ddof=0, so the naive ratio is biased by N/(N-1) -- 1.0005 on the 2,000
    # points the contract uses and 1.00015 on the panel's 6,800, immaterial to
    # every number already published and wrong all the same. Caught by the
    # contract asserting that a book which IS the opponent has beta exactly 1.
    beta = float(np.cov(book[usable], market[usable])[0, 1]
                 / np.var(market[usable], ddof=1))
    return beta, float(np.corrcoef(book[usable], market[usable])[0, 1])


def _rotation_index(frame: pd.DataFrame, lag: int) -> np.ndarray:
    """Row indices that shift each name's position series `lag` bars later.

    THE NULL THIS SCRIPT LACKED UNTIL 2026-09-06. Until then a net Sharpe was
    compared with a constant bar, which cannot tell an edge from a TILT: a
    column whose ranking barely changes from year to year holds nearly the same
    book whatever the date, and earns whatever that book earns. Measured that
    day, this script's own best result of seven days' standing --
    VOLATILITY_50_1d at hold 60, +0.586 -- scored +0.559 with its positions
    moved to random dates. The contribution of knowing WHEN was +0.027, z=0.41.
    It was the low-volatility tilt, and the bar had no way to say so (CLAIMS
    R51).

    Rotation keeps the position ENTIRE -- its persistence, its taste in names,
    its friction -- and breaks only the alignment in time, which is the one
    thing a real edge needs and a tilt does not.

    `frame` is sorted by ticker then datetime, so each name is a contiguous
    block in date order and the shift is a roll within that block.
    """
    rows = np.arange(len(frame))
    starts = np.concatenate([[0], frame.groupby("ticker", sort=False)
                             .size().cumsum().to_numpy()])
    out = np.empty_like(rows)
    for begin, end in zip(starts[:-1], starts[1:]):
        out[begin:end] = np.roll(rows[begin:end], lag % max(end - begin, 1))
    return out


def _panel(names: list[str]):
    """Identifiers, close, and the row order every feature column must follow."""
    ident = pd.read_parquet(
        BATCH / "features.parquet",
        columns=["ticker", "datetime", "interval", "close"])
    ident["datetime"] = pd.to_datetime(ident["datetime"], utc=True)
    keep = ((ident["interval"] == "1d")
            & (ident["datetime"] < SEALED)
            & ident["close"].notna())
    frame = ident[keep].sort_values(["ticker", "datetime"])
    order = frame.index.to_numpy()
    return frame.reset_index(drop=True), order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", type=int, nargs="+",
                        default=[1, 5, 20, 40, 60, 120])
    parser.add_argument("--rotations", type=int, default=ROTATIONS,
                        help="draws of each column's OWN null, built by "
                             "shifting its positions in time. 0 restores the "
                             "pre-2026-09-06 behaviour, where a net Sharpe was "
                             "compared with a constant bar and a static tilt "
                             "was indistinguishable from an edge -- do not use "
                             "it to read a result, only to reproduce an old one.")
    parser.add_argument("--min-varies", type=float, default=MIN_VARIES,
                        help="share of dates on which a column must vary "
                             "across names to be measured at all. Lowering it "
                             "admits columns that hold a position only some of "
                             "the time -- honest, but it changes the attempt "
                             "count and therefore every threshold, so the run "
                             "prints both.")
    parser.add_argument("--rotate-all-holds", action="store_true",
                        help="build the null at EVERY horizon instead of only "
                             "at the best-net one. Off by default because it "
                             "costs six times the rotation work, and on when "
                             "the question is 'does this column know anything "
                             "anywhere' rather than 'is this headline an edge "
                             "or a tilt'. It matters: the reversal family found "
                             "on 2026-09-06 (R52) has its best NET at 120 days "
                             "and its information at 1 to 5, so the default "
                             "pass cannot see it.")
    parser.add_argument("--universe", choices=["varying", "fdr"],
                        default="varying",
                        help="varying: every feature with cross-sectional "
                             "variation. fdr: only the 46 that passed the "
                             "one-day screen, as R23 measured them.")
    args = parser.parse_args()

    roles = pd.read_csv(ROLES)
    if args.universe == "fdr":
        chosen = roles[(roles["passes_fdr"]) & (roles["varies"] > args.min_varies)]
        why = "passed FDR against target_return_1d and vary across names"
    else:
        chosen = roles[roles["varies"] > args.min_varies]
        why = ("have real cross-sectional variation -- selected on NOTHING "
               "about a target")
    names = chosen["feature"].tolist()
    print(f"{len(names)} features {why}")
    if args.min_varies != MIN_VARIES:
        print(f"  varies threshold LOWERED {MIN_VARIES} -> {args.min_varies}: "
              f"{len(names)} columns instead of "
              f"{int((roles['varies'] > MIN_VARIES).sum())}, so every threshold "
              f"below is computed on a larger family than the standard run's.")
    print(f"holds: {args.holds}\n")

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]

    frame, order = _panel(names)
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].nunique():,} dates, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}\n")

    # The scale mark, not a filter (Р47). A cross-section of our names is a
    # cross-section of survivors, and without this line a book measured here
    # reads as a book measured on the market. Printed at BOTH ends of the
    # panel because the share changes by a factor of three across it.
    try:
        from src.data.universe_membership import load as _load_members
        from src.data.universe_membership import scale_note as _scale_note
        _members = _load_members()
        for _edge in (frame["datetime"].min(), frame["datetime"].max()):
            _held = frame.loc[frame["datetime"] == _edge, "ticker"].nunique()
            print("  " + _scale_note(_edge, _held, _members))
    except FileNotFoundError:
        print("  universe scale: UNKNOWN -- no membership store on disk; "
              "run scripts/data/fetch_universe_membership.py")
    print()

    friction = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)
    dates = frame["datetime"].to_numpy()
    forwards = {
        hold: (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy())
        for hold in args.holds
    }

    # The constant opponent, printed BEFORE the features so no result can be
    # read without it. It is the naive book this whole exercise has to beat:
    # own everything, rebalance on the same clock, pay the same friction.
    sorted_codes, uniques = pd.factorize(dates, sort=True)
    n_groups = len(uniques)
    constant = np.ones(len(frame))
    const_sharpe, const_series = {}, {}
    for hold in args.holds:
        # KEPT, not just scored. Until 2026-09-07 this series was computed and
        # thrown away, and with it went the only thing that can say whether a
        # "dollar-neutral" book is actually market-neutral. It is not: a third
        # of these books carry |correlation| above 0.2 with this very series,
        # and the project's best result was 0.746 of it (CLAIMS R61).
        const_series[hold] = _mean_by_date(
            constant * forwards[hold] - np.abs(constant) * friction,
            sorted_codes, n_groups)
        const_sharpe[hold], _ = _sharpe_all_phases(const_series[hold], hold)

    # Each column's own null, built once per rotation and reused for every
    # feature: the shift is a property of the panel, not of the column.
    lags = [int(round((i + 1) / (args.rotations + 1) * n_groups))
            for i in range(args.rotations)]
    rotations = {lag: _rotation_index(frame, lag) for lag in lags}
    if args.rotations:
        print(f"each column is also scored against its OWN null: "
              f"{args.rotations} rotations of its\npositions in time, which "
              f"keeps the book and breaks only the timing. `z` is how\nfar the "
              f"real book stands above that null, in its standard deviations.\n")

    header = (f"{'feature':<34}" + "".join(f"{'h' + str(h):>9}" for h in args.holds)
              + f"{'best net':>10}{'at hold':>9}{'phase sd':>10}{'beta':>8}"
              + (f"{'null':>9}{'z':>8}" if args.rotations else ""))
    print(f"{'BUY EVERYTHING (the opponent)':<34}"
          + "".join(f"{const_sharpe[h]:>9.3f}" for h in args.holds))
    # SURVIVORSHIP: these are today's names carried back, so this number is
    # an upper bound and NOT what the market gave. Measured 2026-09-04: the
    # 1996-2003 slice returns 20.55% a year at Sharpe 1.144, through the
    # dot-com crash, and only 61 of the 110 names existed in 1996. Valid as
    # a RELATIVE opponent -- both books trade the same names -- and
    # misleading as a market benchmark (CLAIMS R34).
    print(f"{' ' * 30}{OPPONENT_CAVEAT}")
    print()
    print(header)
    print("-" * len(header))

    rows = []
    # `close` is itself a feature here, and asking parquet for a column twice
    # returns a DataFrame under that name rather than a Series.
    todo = [n for n in dict.fromkeys(names) if n != "close"]
    if "close" in names:
        todo.append("close")
    for start in range(0, len(todo), CHUNK):
        block = todo[start:start + CHUNK]
        loaded = pd.read_parquet(BATCH / "features.parquet",
                                 columns=list(dict.fromkeys(block)))
        for name in block:
            values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
            values = pd.Series(values)
            if values.notna().sum() < 10_000:
                continue
            position = _position(values.to_numpy(), dates)
            nets, spreads, betas, correlations = {}, {}, {}, {}
            for hold in args.holds:
                net = position * forwards[hold] - np.abs(position) * friction
                series = _mean_by_date(net, sorted_codes, n_groups)
                nets[hold], spreads[hold] = _sharpe_all_phases(series, hold)
                betas[hold], correlations[hold] = _beta_on(
                    series, const_series[hold])
            best = max(nets, key=lambda h: (nets[h] if np.isfinite(nets[h]) else -9))

            # The same book, told nothing about when. By default this is
            # scored at the BEST-NET hold only, because the null is here to say
            # what KIND of result the headline is, and six times the rotation
            # work is a different question with a different price. The cost of
            # that default is named rather than hidden: a column whose money is
            # at one horizon and whose information is at another goes unseen,
            # which is exactly the reversal family of R52. `--rotate-all-holds`
            # asks the other question.
            null_mean = null_sd = z = float("nan")
            z_hold = best
            if args.rotations:
                where = args.holds if args.rotate_all_holds else [best]
                moved_cache = {}
                for lag in lags:
                    moved = position[rotations[lag]]
                    # Re-neutralise: after the shift the names on a date are
                    # not the ones the original weights balanced.
                    moved_cache[lag] = moved - _mean_by_date(
                        moved, sorted_codes, n_groups)[sorted_codes]
                for hold in where:
                    turns = [
                        _sharpe_all_phases(
                            _mean_by_date(
                                moved_cache[lag] * forwards[hold]
                                - np.abs(moved_cache[lag]) * friction,
                                sorted_codes, n_groups), hold)[0]
                        for lag in lags]
                    mean_h = float(np.nanmean(turns))
                    sd_h = float(np.nanstd(turns, ddof=1))
                    z_h = (nets[hold] - mean_h) / sd_h if sd_h > 0 else float("nan")
                    # Keep the horizon that knows the most, not the one that
                    # earned the most -- otherwise the flag changes nothing.
                    if not np.isfinite(z) or (np.isfinite(z_h) and z_h > z):
                        null_mean, null_sd, z, z_hold = mean_h, sd_h, z_h, hold

            rows.append({"feature": name, "best_net": nets[best],
                         "best_hold": best, "phase_sd": spreads[best],
                         "rotated_null": null_mean, "rotated_sd": null_sd,
                         "z_vs_own_null": z, "z_hold": z_hold,
                         "net_at_z_hold": nets.get(z_hold, float("nan")),
                         "beta": betas[best], "correlation": correlations[best],
                         "beta_at_z_hold": betas.get(z_hold, float("nan")),
                         **{f"h{h}": nets[h] for h in args.holds}})
            print(f"{name:<34}"
                  + "".join(f"{nets[h]:>9.3f}" for h in args.holds)
                  + f"{nets[best]:>10.3f}{best:>9}{spreads[best]:>10.3f}"
                  + f"{betas[best]:>8.3f}"
                  + (f"{null_mean:>9.3f}{z:>8.2f}" if args.rotations else ""),
                  flush=True)
        del loaded

    report = pd.DataFrame(rows)
    if report.empty:
        print("\nnothing measurable")
        return 1

    attempts = len(report) * len(args.holds)
    bonferroni, noise_max = _thresholds(attempts)

    print("\n" + "=" * len(header))
    print("constant opponent, by hold:  "
          + "  ".join(f"h{h}={const_sharpe[h]:+.3f}" for h in args.holds))
    print(f"features measured                            {len(report)}")
    print(f"attempts (features x holds)                  {attempts}")
    print(f"best net Sharpe anywhere                     {report['best_net'].max():+.3f}")
    print(f"expected maximum of that many noise draws    {noise_max:.3f}")
    print(f"Bonferroni family-wise 5%                    {bonferroni:.3f}")
    real = report[report["best_net"] >= bonferroni]
    print(f"clear Bonferroni                             {len(real)}")
    print(f"clear the noise maximum                      "
          f"{int((report['best_net'] >= noise_max).sum())}")
    print(f"positive but inside the noise                "
          f"{int(((report['best_net'] > 0) & (report['best_net'] < noise_max)).sum())}")
    print(f"negative at every horizon                    "
          f"{int((report['best_net'] <= 0).sum())}")
    # HOW NEUTRAL THE "NEUTRAL" BOOKS ARE. Printed unconditionally because the
    # answer was assumed for a fortnight and is not zero: a third of these
    # books move with the opponent, and the one that scored best moved with it
    # most (CLAIMS R61).
    exposed = int((report["correlation"].abs() > 0.2).sum())
    print(f"\n|beta| median / 90th / max                   "
          f"{report['beta'].abs().median():.3f} / "
          f"{report['beta'].abs().quantile(0.9):.3f} / "
          f"{report['beta'].abs().max():.3f}")
    print(f"books moving with the opponent (|corr|>0.2)  {exposed} of "
          f"{len(report)}   <- dollar-neutral is not market-neutral")

    for hold in args.holds:
        column = report[f"h{hold}"]
        print(f"  best at hold {hold:>4}: {column.max():+.3f}  "
              f"({column.idxmax() in report.index and report.loc[column.idxmax(), 'feature']})")

    if args.rotations:
        # THE SECOND TEST, AND IT IS NOT THE SAME ONE. A net Sharpe above the
        # bar says the book made money. A z above the bar says the book knew
        # WHEN. A result needs both, and until 2026-09-06 this script could
        # only ask the first (CLAIMS R51, R52).
        z_bar = float(norm.ppf(1.0 - 0.025 / max(attempts, 1)))
        # AGAINST THE OPPONENT AT ITS OWN HORIZON, not against zero. A book
        # that knows something and earns +0.026 while owning the same names
        # pays +1.018 is a measured number, not a candidate -- and doing this
        # comparison by hand after the run is how the check gets skipped
        # (CLAIMS R28, R53).
        report["vs_opponent"] = (report["net_at_z_hold"]
                                 - report["z_hold"].map(const_sharpe))
        known = report[report["z_vs_own_null"] >= z_bar]
        print(f"\nz against each column's OWN rotated null, bar {z_bar:.2f}:")
        print(f"  know something about WHEN                  {len(known)}")
        print(f"  of those, also net positive                "
              f"{int((known['net_at_z_hold'] > 0).sum())}")
        print(f"  net positive but NOT above their own null  "
              f"{int(((report['best_net'] > 0) & (report['z_vs_own_null'] < z_bar)).sum())}"
              f"   <- tilts, not edges")
        if len(known):
            print(known[["feature", "z_hold", "net_at_z_hold", "rotated_null",
                         "z_vs_own_null", "vs_opponent", "beta_at_z_hold"]]
                  .sort_values("z_vs_own_null", ascending=False)
                  .head(15).to_string(index=False))
            better = known[known["vs_opponent"] > 0]
            print(f"\n  of those, BEAT BUYING EVERYTHING at the same hold  "
                  f"{len(better)}")
            if not len(better):
                print("  Every one of them knows something and is still worth "
                      "less than owning the\n  same names outright. That is a "
                      "measured number, not a candidate.")

    if len(real):
        print("\nthese clear the noise and are the first real candidates:")
        print(real[["feature", "best_net", "best_hold", "phase_sd",
                    "z_vs_own_null"]]
              .sort_values("best_net", ascending=False).to_string(index=False))
    else:
        print("\nNONE clears the multiplicity-corrected bar. Widening the "
              "universe from 46 to every\nvarying feature, and the horizon out "
              "to 120 days, did not find one. The conclusion\nis not about "
              "which feature was picked and not about the holding period: it "
              "is about\nthis universe and these features.")
    # THE FILENAME CARRIES WHAT MADE THE RUN DIFFERENT. Twice on 2026-09-06 a
    # non-standard run silently overwrote the standard record -- once a
    # two-hold smoke test over `net_test_fdr.csv`, once a lowered threshold
    # over `net_test_varying.csv` -- and both times the file still looked like
    # the artefact everything cites. A record that does not say how it was made
    # is a record that will be quoted as something else.
    suffix = "" if args.min_varies == MIN_VARIES else f"_varies{args.min_varies:g}"
    if args.holds != [1, 5, 20, 40, 60, 120]:
        suffix += "_holds" + "-".join(str(h) for h in args.holds)
    if args.rotate_all_holds:
        suffix += "_allholds"
    out = (PROJECT_ROOT / "diagnostic_reports"
           / f"net_test_{args.universe}{suffix}.csv")
    report.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
