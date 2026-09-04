"""Every net test so far ranked ONE column. This one fits a combination.

REGISTER #196: we measure cross-sectional IC and we train a temporal binary
classifier, so the thing being optimised is not the thing being counted as
evidence. Before rewiring stage 4 to train a ranker, the cheap question has to
be answered: is there anything for a ranker to find?

WHY THIS IS NOT THE SAME MEASUREMENT AT ANOTHER ANGLE. Everything measured to
date has been ONE COLUMN AT A TIME -- `net_test_every_survivor` builds a book
from a single feature's cross-sectional rank, 235 features x 6 holds, and none
clears the multiplicity bar. That closes "is any single column tradeable". It
says nothing about a COMBINATION, which is precisely what a trained ranker
produces and what no run has ever built. Two columns that individually rank
inside the noise can, together, rank outside it; that is the whole reason
models exist.

WHAT IS PRE-DECLARED, so the count of attempts is not chosen after the answer:

    representation   each feature -> per-date percentile rank minus 0.5.
                     This IS "rank within the date": it is what a ranking
                     model consumes, and it puts every column on one scale
                     without a fitted parameter.

    outcome          forward return at hold h, cross-sectionally demeaned.
                     Demeaned because a dollar-neutral book cannot earn the
                     market and must not be scored as if it could.

    split            chronological, on DATES. The first 70% of pre-seal dates
                     fit the weights, the last 30% are the measurement. The
                     sealed period is not touched by either.

    models           three, named before running:
                       top5   equal weight on the 5 largest |train IC|,
                              signed by that IC
                       top20  the same with 20
                       ridge  ridge regression on every ranked column,
                              alpha = number of features -- a scale-free
                              default, NOT tuned, because tuning alpha is a
                              search and searches have to be counted

    attempts         3 models x the holds. Printed, and the thresholds are
                     derived from it rather than stated.

THE OPPONENTS ARE PRINTED FIRST, both of them:

    buy everything   the constant book, same clock, same friction. Survivorship
                     -inflated (today's names carried back), so an upper bound.

    best single      the best SINGLE feature on the SAME test dates. This is
                     the opponent that matters here: a combination that cannot
                     beat the best of its own ingredients has added nothing.
                     It is itself a maximum over 235 draws, so it is a HARD
                     opponent -- which is the direction to err in.

    python scripts/diagnostics/does_a_combination_beat_the_best_single_feature.py
    python scripts/diagnostics/does_a_combination_beat_the_best_single_feature.py --holds 20 60
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Line-buffered from the start. Redirected to a file, Python buffers stdout in
# 8 KiB blocks, so a run that takes minutes shows NOTHING until it ends -- and
# a silent long run is indistinguishable from a hung one, which is the state
# this project has twice mistaken for a hang.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scipy.stats import norm  # noqa: E402

from net_test_every_survivor import (  # noqa: E402
    BATCH, CHUNK, ROLES, _panel, _sharpe_all_phases,
)
from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

#: Share of pre-seal DATES used to fit. Split on dates, never on rows: a row
#: split would put the same day on both sides of the line, 110 names at a
#: time, and the weights would be fitted on the answer.
TRAIN_FRACTION = 0.70

#: Model names, fixed here so the count of attempts is a property of the file
#: rather than of what was tried.
MODELS = ("top5", "top20", "ridge")

#: How many random-weight books form the null.
#:
#: It was 20, and 20 was not enough. An sd from 20 draws carries about 16%
#: relative error, and the fitted books landed at z = 2.6 against a Bonferroni
#: requirement of 2.87 -- inside the error of the threshold itself. At 200 the
#: null supports a direct empirical p-value (how many random books beat the
#: fitted one) instead of a z against an estimated sd, and each draw is one
#: matrix-vector product.
CONTROLS = 200

#: Substrings that name the volatility family. Excluded with --without-vol,
#: because the best single column at four of six holds is one of these and a
#: combination that merely rediscovers the low-volatility anomaly is a real
#: cross-sectional effect that is NOT this project's alpha -- it is sold as an
#: ETF. Matched case-insensitively on the column name.
VOL_FAMILY = ("vol", "atr", "std", "gk_", "range", "drawdown", "beta")


def _thresholds(attempts: int, years: float) -> tuple[float, float]:
    """Bonferroni and the expected maximum of noise, IN THIS SAMPLE'S UNITS.

    `net_test_every_survivor._thresholds` hardcodes SHARPE_SE = 0.193, which is
    1/sqrt(27) -- correct there, because it measures over the whole explorable
    period. Importing it here would have applied a 27-year standard error to an
    8-year test window and made the bar three times too lenient. Caught by
    asking what 0.193 was 1/sqrt of, after the first run returned numbers that
    cleared it easily.
    """
    se = 1.0 / math.sqrt(max(years, 1e-9))
    bonferroni = float(norm.ppf(1.0 - 0.025 / max(attempts, 1))) * se
    if attempts <= 1:
        return float(norm.ppf(0.975)) * se, float(norm.ppf(0.95)) * se
    log_n = math.log(attempts)
    root = math.sqrt(2.0 * log_n)
    noise_max = root - (math.log(log_n) + math.log(4.0 * math.pi)) / (2.0 * root)
    return bonferroni, max(noise_max, float(norm.ppf(0.95))) * se


def _ranked(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Per-date percentile rank, centred. NaN becomes 0 -- no position."""
    series = pd.Series(values)
    rank = series.groupby(dates).rank(pct=True).to_numpy() - 0.5
    return np.nan_to_num(rank).astype(np.float32)


def _demean(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return (series - series.groupby(dates).transform("mean")).to_numpy()


class _Dates:
    """Date codes computed ONCE.

    The panel is sorted by (ticker, datetime), so dates are not contiguous and
    every book needs a group-by. Doing that with pandas inside a loop over 235
    columns x 6 holds is a quarter of an hour of the same work; factorising
    once and using bincount is the same arithmetic in seconds.
    """

    def __init__(self, dates: np.ndarray):
        codes, uniques = pd.factorize(dates, sort=True)
        self.codes = codes.astype(np.int64)
        self.count = len(uniques)
        self.rows_per_date = np.bincount(self.codes, minlength=self.count)

    def mean_by_date(self, values: np.ndarray) -> np.ndarray:
        total = np.bincount(self.codes, weights=values, minlength=self.count)
        return total / np.maximum(self.rows_per_date, 1)

    def demean(self, values: np.ndarray) -> np.ndarray:
        return values - self.mean_by_date(values)[self.codes]


def _by_period(score: np.ndarray, index: "_Dates", forward: np.ndarray,
               friction: np.ndarray) -> np.ndarray:
    """The book's net return per DATE, before any sampling."""
    position = np.nan_to_num(np.sign(index.demean(score)))
    position = index.demean(position)
    net = position * np.nan_to_num(forward) - np.abs(position) * friction
    return index.mean_by_date(net)


def _blocks(by_date: np.ndarray, hold: int, blocks: int = 4) -> list[tuple]:
    """Does it hold up across the test period, or is it one episode?

    This is the check that killed every previous candidate in this project.
    `insider_net_value_30d` had a pooled t of -6.26 and, split by period,
    -2.74 / -3.01 / -6.01 (spring 2020) / -1.88 / -0.08. The sign never
    flipped; the effect simply stopped.

    Reported as observations, mean net per period and share positive rather
    than as a Sharpe: a quarter of eight years at a 60-day hold holds eight
    non-overlapping periods, and a Sharpe over eight points is a number about
    eight points.
    """
    sampled = by_date[::hold]
    sampled = sampled[np.isfinite(sampled)]
    out = []
    for part in np.array_split(sampled, blocks):
        if not len(part):
            out.append((0, float("nan"), float("nan")))
            continue
        out.append((len(part), float(np.mean(part)),
                    float(np.mean(part > 0))))
    return out


def _book_sharpe(score: np.ndarray, index: "_Dates", forward: np.ndarray,
                 friction: np.ndarray, hold: int) -> tuple[float, float]:
    """The same book construction as every other net test in this project.

    Dollar-neutral, or the answer is the market: the sign is taken first and
    the per-date mean subtracted after, so a score that says nothing about
    which name carries no position rather than being long everything.
    """
    position = np.nan_to_num(np.sign(index.demean(score)))
    position = index.demean(position)
    net = position * np.nan_to_num(forward) - np.abs(position) * friction
    return _sharpe_all_phases(index.mean_by_date(net), hold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", type=int, nargs="+",
                        default=[1, 5, 20, 40, 60, 120])
    parser.add_argument("--permutations", type=int, default=200,
                        help="how many times the ridge is refitted against a "
                             "target permuted WITHIN each date")
    parser.add_argument("--without-vol", action="store_true",
                        help="drop the volatility family, to see whether the "
                             "combination is anything but low-vol")
    args = parser.parse_args()

    roles = pd.read_csv(ROLES)
    names = roles[roles["varies"] > 0.5]["feature"].tolist()
    names = [n for n in dict.fromkeys(names)]
    if args.without_vol:
        before = len(names)
        names = [n for n in names
                 if not any(token in n.lower() for token in VOL_FAMILY)]
        print(f"--without-vol: {before - len(names)} of {before} columns "
              f"dropped as volatility family")
    print(f"{len(names)} features with real cross-sectional variation")
    print(f"holds: {args.holds}   models: {list(MODELS)}")

    attempts = len(MODELS) * len(args.holds)

    frame, order = _panel(names)
    dates = frame["datetime"].to_numpy()
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].nunique():,} dates, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}")

    unique_dates = np.sort(pd.unique(dates))
    cut = unique_dates[int(len(unique_dates) * TRAIN_FRACTION)]
    is_train = dates < cut
    test_years = (pd.Timestamp(unique_dates[-1]) - pd.Timestamp(cut)).days / 365.25
    print(f"split at {pd.Timestamp(cut).date()}: "
          f"{int(is_train.sum()):,} train rows, {int((~is_train).sum()):,} test, "
          f"{test_years:.1f} test years")
    bonferroni, noise_max = _thresholds(attempts, test_years)
    print(f"attempts (models x holds): {attempts}   "
          f"Sharpe SE on {test_years:.1f} years {1 / math.sqrt(test_years):.3f}   "
          f"noise maximum {noise_max:.3f}   Bonferroni {bonferroni:.3f}")
    print()

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    friction = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)
    forwards = {
        hold: (frame.groupby("ticker", sort=False)["close"]
               .transform(lambda s, h=hold: s.shift(-h) / s - 1.0).to_numpy())
        for hold in args.holds
    }

    # --- the opponents, before anything is fitted ---
    test = ~is_train
    index = _Dates(dates[test])
    constant = {}
    for hold in args.holds:
        net = np.nan_to_num(forwards[hold][test]) - friction[test]
        constant[hold], _ = _sharpe_all_phases(index.mean_by_date(net), hold)
    print(f"{'BUY EVERYTHING (opponent 1)':<30}"
          + "".join(f"{constant[h]:>9.3f}" for h in args.holds))
    print(f"{' ' * 30}survivorship-inflated: an upper bound, not the market\n")

    # --- load every ranked column once ---
    matrix = np.zeros((len(frame), len(names)), dtype=np.float32)
    kept: list[str] = []
    for start in range(0, len(names), CHUNK):
        block = names[start:start + CHUNK]
        loaded = pd.read_parquet(BATCH / "features.parquet",
                                 columns=list(dict.fromkeys(block)))
        for name in block:
            values = pd.to_numeric(loaded[name], errors="coerce").to_numpy()[order]
            if np.isfinite(values).sum() < 10_000:
                continue
            matrix[:, len(kept)] = _ranked(values, dates)
            kept.append(name)
        del loaded
        print(f"  loaded {min(start + CHUNK, len(names))}/{len(names)}", flush=True)
    matrix = matrix[:, :len(kept)]
    print(f"{len(kept)} columns usable\n")

    single_best: dict[int, tuple[float, str]] = {}
    combos: dict[tuple[str, int], float] = {}
    spreads: dict[tuple[str, int], float] = {}
    random_best: dict[int, float] = {}
    random_median: dict[int, float] = {}
    random_sd: dict[int, float] = {}
    random_mean: dict[int, float] = {}
    control_draws: dict[int, np.ndarray] = {}
    permuted: dict[int, np.ndarray] = {}
    ridge_score: dict[int, np.ndarray] = {}
    shuffled: dict[int, float] = {}
    ridge_curve: dict[int, np.ndarray] = {}

    for hold in args.holds:
        outcome = _demean(forwards[hold], dates)
        finite = np.isfinite(outcome)

        # opponent 2: the best SINGLE column on the same test dates
        best, best_name = -np.inf, "-"
        for column in range(matrix.shape[1]):
            value, _ = _book_sharpe(matrix[test, column], index,
                                    forwards[hold][test], friction[test], hold)
            if np.isfinite(value) and value > best:
                best, best_name = value, kept[column]
        single_best[hold] = (best, best_name)

        # the fit, on train only
        train = is_train & finite
        y = outcome[train]
        x = matrix[train]
        ic = np.array([
            np.corrcoef(x[:, column], y)[0, 1] if x[:, column].std() > 0 else 0.0
            for column in range(x.shape[1])
        ])
        ic = np.nan_to_num(ic)

        # THE NULL, BEFORE THE FIT IS BELIEVED. Two controls, because they
        # fail differently:
        #
        #   random weights   the fit replaced by coin flips on the same
        #                    columns. If these score what the fitted books
        #                    score, the number is the CONSTRUCTION -- a
        #                    diversified sign book over 235 ranked columns --
        #                    and the weights are decoration.
        #
        #   shuffled y       the same ridge, fitted against a train outcome
        #                    shuffled within the train rows. A real design
        #                    matrix and a meaningless answer key, so anything
        #                    it earns on test is coincidence or leakage. This
        #                    is `negative_control.py` applied to a book rather
        #                    than to the gate.
        rng = np.random.default_rng(11)
        control_scores = []
        for _ in range(CONTROLS):
            coin = rng.choice([-1.0, 1.0], size=matrix.shape[1]).astype(np.float32)
            value, _ = _book_sharpe(matrix[test] @ coin, index,
                                    forwards[hold][test], friction[test], hold)
            if np.isfinite(value):
                control_scores.append(value)
        random_best[hold] = max(control_scores) if control_scores else float("nan")
        random_median[hold] = (float(np.median(control_scores))
                               if control_scores else float("nan"))
        # The EMPIRICAL null, and it is the one that binds. The theoretical
        # Sharpe standard error assumes independent periods and a fixed book;
        # these draws are the same construction on the same data with the
        # weights replaced by coin flips, so their spread already contains
        # whatever the construction does on its own. It has been WIDER than
        # the theoretical figure at every hold measured.
        random_sd[hold] = (float(np.std(control_scores, ddof=1))
                           if len(control_scores) > 2 else float("nan"))
        random_mean[hold] = (float(np.mean(control_scores))
                             if control_scores else float("nan"))
        control_draws[hold] = np.asarray(control_scores)

        # THE PERMUTATION NULL, and it is the one that answers the question.
        #
        # Random +-1 weights are not the right null for a FITTED book: they
        # ask "is this weighting special", when the question is "does the
        # FITTING PROCEDURE find anything". Measured here at 200 draws, the
        # two disagreed -- zero of 200 random books beat the fit (empirical
        # p = 0.000) while the same fit sat at z = 2.4 against their mean and
        # sd, short of the 2.87 Bonferroni needs. They disagree because the
        # random-weight distribution is skewed, so a z computed from its mean
        # and sd is not the same statement as its tail.
        #
        # The right null runs the SAME ridge against a target that has been
        # permuted WITHIN EACH DATE: the day's cross-section of returns is
        # kept exactly, and only which name earned which return is destroyed.
        # Permuting freely across the panel would also destroy the time
        # structure, which is not the hypothesis being tested.
        #
        # The gram matrix does not depend on y, so each extra draw is one
        # matrix-vector product and a solve.
        gram = (x.T @ x).astype(np.float64)
        gram[np.diag_indices_from(gram)] += float(matrix.shape[1])
        # Within-date permutation without a Python loop over 4,760 dates:
        # `lexsort` by (random key, date code) walks the groups in the same
        # order as a stable sort by code alone, so assigning one sequence
        # from the other permutes strictly inside each date. The naive loop
        # was 2e9 comparisons per draw; this is one sort.
        train_codes = _Dates(dates[train]).codes
        base_order = np.lexsort((np.arange(len(y)), train_codes))
        permuted_scores = []
        for _ in range(args.permutations):
            shuffled_y = np.empty_like(y)
            shuffled_y[base_order] = y[np.lexsort((rng.random(len(y)),
                                                   train_codes))]
            sham = np.linalg.solve(
                gram, (x.T @ shuffled_y.astype(np.float32)).astype(np.float64))
            value, _ = _book_sharpe(matrix[test] @ sham.astype(np.float32),
                                    index, forwards[hold][test],
                                    friction[test], hold)
            if np.isfinite(value):
                permuted_scores.append(value)
        permuted[hold] = np.asarray(permuted_scores)
        shuffled[hold] = (max(permuted_scores) if permuted_scores
                          else float("nan"))

        for model in MODELS:
            if model.startswith("top"):
                k = int(model[3:])
                chosen = np.argsort(-np.abs(ic))[:k]
                weights = np.zeros(matrix.shape[1])
                weights[chosen] = np.sign(ic[chosen])
            else:
                alpha = float(matrix.shape[1])
                # float32 for the product, float64 for the solve: the full
                # 430k x 235 design cast to float64 is 800 MiB for no
                # precision that ranks in [-0.5, 0.5] can use.
                gram = (x.T @ x).astype(np.float64)
                gram[np.diag_indices_from(gram)] += alpha
                weights = np.linalg.solve(
                    gram, (x.T @ y.astype(np.float32)).astype(np.float64))
            score = matrix[test] @ weights.astype(np.float32)
            value, spread = _book_sharpe(score, index,
                                         forwards[hold][test], friction[test], hold)
            combos[(model, hold)] = value
            spreads[(model, hold)] = spread
            if model == "ridge":
                ridge_curve[hold] = _by_period(score, index,
                                               forwards[hold][test],
                                               friction[test])
                ridge_score[hold] = score
        print(f"  hold {hold} done", flush=True)

    print()
    header = (f"{'book':<30}" + "".join(f"{'h' + str(h):>9}" for h in args.holds))
    print(header)
    print("-" * len(header))
    print(f"{'BUY EVERYTHING (opponent 1)':<30}"
          + "".join(f"{constant[h]:>9.3f}" for h in args.holds))
    print(f"{'BEST SINGLE (opponent 2)':<30}"
          + "".join(f"{single_best[h][0]:>9.3f}" for h in args.holds))
    print(f"{'RANDOM WEIGHTS, best of ' + str(CONTROLS):<30}"
          + "".join(f"{random_best[h]:>9.3f}" for h in args.holds))
    print(f"{'RANDOM WEIGHTS, median':<30}"
          + "".join(f"{random_median[h]:>9.3f}" for h in args.holds))
    print(f"{'RIDGE ON A PERMUTED TARGET, best':<30}"
          + "".join(f"{shuffled[h]:>9.3f}" for h in args.holds))
    print(f"{'   the same, median':<30}"
          + "".join(f"{np.median(permuted[h]):>9.3f}" for h in args.holds))
    print()
    for model in MODELS:
        print(f"{'combination: ' + model:<30}"
              + "".join(f"{combos[(model, h)]:>9.3f}" for h in args.holds))
    print(f"{'  phase sd (ridge)':<30}"
          + "".join(f"{spreads[('ridge', h)]:>9.3f}" for h in args.holds))
    print()
    print("=== against the EMPIRICAL null, which is the bar that binds ===")
    print(f"{'random-weight books: mean':<30}"
          + "".join(f"{random_mean[h]:>9.3f}" for h in args.holds))
    print(f"{'random-weight books: sd':<30}"
          + "".join(f"{random_sd[h]:>9.3f}" for h in args.holds))
    for model in MODELS:
        z = []
        for h in args.holds:
            sd = random_sd[h]
            z.append((combos[(model, h)] - random_mean[h]) / sd
                     if np.isfinite(sd) and sd > 0 else float("nan"))
        print(f"{'  z of ' + model:<30}" + "".join(f"{v:>9.2f}" for v in z))
    z_needed = float(norm.ppf(1.0 - 0.025 / max(attempts, 1)))
    print(f"  Bonferroni z for {attempts} attempts: {z_needed:.2f}")
    print()
    print("  empirical p: share of random-weight books that BEAT the fit")
    for model in MODELS:
        cells = []
        for h in args.holds:
            draws = control_draws.get(h)
            if draws is None or not len(draws):
                cells.append(float("nan"))
                continue
            cells.append(float(np.mean(draws >= combos[(model, h)])))
        print(f"{'  p of ' + model:<30}" + "".join(f"{v:>9.3f}" for v in cells))
    print(f"  a p below {0.05 / attempts:.4f} would clear Bonferroni at "
          f"{attempts} attempts")
    print()
    print("=== the permutation null: the SAME ridge, target permuted within "
          "each date ===")
    print("Random weights ask whether this weighting is special. This asks")
    print("whether the FITTING finds anything, which is the actual question.")
    print(f"{'  permuted ridge: mean':<30}"
          + "".join(f"{np.mean(permuted[h]):>9.3f}" for h in args.holds))
    print(f"{'  permuted ridge: sd':<30}"
          + "".join(f"{np.std(permuted[h], ddof=1):>9.3f}" for h in args.holds))
    print(f"{'  permuted ridge: max':<30}"
          + "".join(f"{np.max(permuted[h]):>9.3f}" for h in args.holds))
    print(f"{'  REAL ridge':<30}"
          + "".join(f"{combos[('ridge', h)]:>9.3f}" for h in args.holds))
    print(f"{'  empirical p':<30}"
          + "".join(f"{np.mean(permuted[h] >= combos[('ridge', h)]):>9.3f}"
                   for h in args.holds))
    print()
    for hold in args.holds:
        print(f"  h{hold}: best single was {single_best[hold][1]}")

    print()
    print("=== is it a signal, or a bet on which NAMES survived? ===")
    print("The permutation null cannot answer this. Permuting returns within a")
    print("date destroys the name-return link, so a book that is simply long")
    print("the same survivors every day earns ~0 under permutation and looks")
    print("significant. This project has the exact precedent: a cross-sectional")
    print("book built without the within-name check held NVDA on 100% of days")
    print("against SPY on 100% of days and 'returned' 34.9% a year.")
    print()
    tickers_test = frame["ticker"].to_numpy()[test]
    for hold in args.holds:
        position = np.nan_to_num(np.sign(index.demean(ridge_score[hold])))
        position = index.demean(position)
        held = pd.Series(position).groupby(tickers_test).mean()
        extreme = held.reindex(held.abs().sort_values(ascending=False).index)
        top = "  ".join(f"{name} {value:+.2f}"
                        for name, value in extreme.head(6).items())
        share = float((held.abs() > 0.8).mean())
        print(f"  h{hold}: mean position per name, most extreme six")
        print(f"        {top}")
        print(f"        names held one way on >80% of dates: {share:.0%}")

        # The same book with each name's OWN average position removed: what
        # is left is only the part that changes over time. If the Sharpe
        # collapses here, the book was a static bet.
        moving = position - pd.Series(position).groupby(tickers_test).transform("mean").to_numpy()
        moving = index.demean(moving)
        net = moving * np.nan_to_num(forwards[hold][test]) - np.abs(moving) * friction[test]
        value, _ = _sharpe_all_phases(index.mean_by_date(net), hold)
        print(f"        Sharpe of the TIME-VARYING part only: {value:.3f}"
              f"   (whole book {combos[('ridge', hold)]:.3f})")

    print()
    print("=== does the ridge book hold up across the test period? ===")
    print("Four equal blocks of the test dates. An effect that lives in one")
    print("block is an episode, not a signal -- the check that killed")
    print("insider_net_value_30d, which had a pooled t of -6.26 and nothing")
    print("left in its final twenty months.")
    for hold in args.holds:
        parts = _blocks(ridge_curve[hold], hold)
        print(f"  h{hold}: " + "  ".join(
            f"[n={n} mean {mean:+.5f} pos {share:.0%}]" for n, mean, share in parts))

    print()
    best_combo = max(combos.values())
    print(f"best combination anywhere              {best_combo:>9.3f}")
    print(f"expected maximum of {attempts} noise draws   {noise_max:>9.3f}")
    print(f"Bonferroni family-wise 5%              {bonferroni:>9.3f}")
    beat_single = [f"{m} h{h}" for (m, h), v in combos.items()
                   if np.isfinite(v) and v > single_best[h][0]]
    print(f"\ncombinations beating the best single column: "
          f"{len(beat_single)} of {attempts}"
          + (f" -- {beat_single}" if beat_single else ""))
    control_ceiling = max(max(random_best.values()), max(shuffled.values()))
    print(f"best control (random weights or shuffled target)"
          f"{control_ceiling:>9.3f}")
    if best_combo <= control_ceiling:
        print()
        print("THE FIT ADDS NOTHING: a control scores what the fitted books")
        print("score, so the number comes from the construction, not from")
        print("anything learned. Read no further into it.")
    if best_combo <= noise_max:
        print("\nNo combination clears the noise maximum for this many attempts.")
        print("A ranker trained on these columns would be fitting the same")
        print("nothing the single columns already found, and #196 would be a")
        print("correct change to an apparatus with no signal to align it to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
