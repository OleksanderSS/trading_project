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

Both legs pay the friction, the Sharpe is the portfolio's, and holding periods
are sampled without overlap. The sealed period is untouched.

    python scripts/diagnostics/net_test_every_survivor.py
    python scripts/diagnostics/net_test_every_survivor.py --universe fdr
    python scripts/diagnostics/net_test_every_survivor.py --holds 1 20 120
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

from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
ROLES = PROJECT_ROOT / "diagnostic_reports" / "feature_roles_1d.csv"
SEALED = pd.Timestamp("2023-09-01", tz="UTC")

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
    parser.add_argument("--universe", choices=["varying", "fdr"],
                        default="varying",
                        help="varying: every feature with cross-sectional "
                             "variation. fdr: only the 46 that passed the "
                             "one-day screen, as R23 measured them.")
    args = parser.parse_args()

    roles = pd.read_csv(ROLES)
    if args.universe == "fdr":
        chosen = roles[(roles["passes_fdr"]) & (roles["varies"] > 0.5)]
        why = "passed FDR against target_return_1d and vary across names"
    else:
        chosen = roles[roles["varies"] > 0.5]
        why = ("have real cross-sectional variation -- selected on NOTHING "
               "about a target")
    names = chosen["feature"].tolist()
    print(f"{len(names)} features {why}")
    print(f"holds: {args.holds}\n")

    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]

    frame, order = _panel(names)
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].nunique():,} dates, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}\n")

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
    constant = np.ones(len(frame))
    const_sharpe = {}
    for hold in args.holds:
        work = pd.DataFrame({"datetime": dates})
        work["net"] = constant * forwards[hold] - np.abs(constant) * friction
        const_sharpe[hold], _ = _sharpe_all_phases(
            work.groupby("datetime")["net"].mean().sort_index().to_numpy(), hold)

    header = (f"{'feature':<34}" + "".join(f"{'h' + str(h):>9}" for h in args.holds)
              + f"{'best net':>10}{'at hold':>9}{'phase sd':>10}")
    print(f"{'BUY EVERYTHING (the opponent)':<34}"
          + "".join(f"{const_sharpe[h]:>9.3f}" for h in args.holds))
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
            position = np.sign(
                values.groupby(dates).rank(pct=True).to_numpy() - 0.5)
            position = np.nan_to_num(position)
            # DOLLAR-NEUTRAL, OR THE ANSWER IS THE MARKET.
            #
            # `sign(rank - 0.5)` on a column with heavy ties gives +1 to
            # EVERYONE: pandas ranks ties by their average, so a binary flag
            # that is 98.9% one value ranks near 0.5+ for every name and the
            # "long/short book" is long everything. Measured 04.09, seven
            # features cleared Bonferroni at ~1.00 net and the constant
            # opponent -- buy every name, rebalance on the same clock, pay the
            # same friction -- scores 1.018 at a 60-day hold. Every one of the
            # seven was the constant opponent to three decimals.
            #
            # Subtracting the per-date mean removes exactly that exposure and
            # leaves a degenerate column with no position at all, which is the
            # honest answer for a column that says nothing about which name.
            # Without it this script has no opponent ladder and compares
            # against zero -- the defect the promotion gate spent a week
            # having removed (CLAIMS R11, R17), reproduced here.
            position = position - (pd.Series(position).groupby(dates)
                                   .transform("mean").to_numpy())
            work = pd.DataFrame({"datetime": dates})
            nets, spreads = {}, {}
            for hold in args.holds:
                work["net"] = position * forwards[hold] - np.abs(position) * friction
                by_date = work.groupby("datetime")["net"].mean().sort_index()
                nets[hold], spreads[hold] = _sharpe_all_phases(
                    by_date.to_numpy(), hold)
            best = max(nets, key=lambda h: (nets[h] if np.isfinite(nets[h]) else -9))
            rows.append({"feature": name, "best_net": nets[best],
                         "best_hold": best, "phase_sd": spreads[best],
                         **{f"h{h}": nets[h] for h in args.holds}})
            print(f"{name:<34}"
                  + "".join(f"{nets[h]:>9.3f}" for h in args.holds)
                  + f"{nets[best]:>10.3f}{best:>9}{spreads[best]:>10.3f}",
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

    for hold in args.holds:
        column = report[f"h{hold}"]
        print(f"  best at hold {hold:>4}: {column.max():+.3f}  "
              f"({column.idxmax() in report.index and report.loc[column.idxmax(), 'feature']})")

    if len(real):
        print("\nthese clear the noise and are the first real candidates:")
        print(real[["feature", "best_net", "best_hold", "phase_sd"]]
              .sort_values("best_net", ascending=False).to_string(index=False))
    else:
        print("\nNONE clears the multiplicity-corrected bar. Widening the "
              "universe from 46 to every\nvarying feature, and the horizon out "
              "to 120 days, did not find one. The conclusion\nis not about "
              "which feature was picked and not about the holding period: it "
              "is about\nthis universe and these features.")
    out = PROJECT_ROOT / "diagnostic_reports" / f"net_test_{args.universe}.csv"
    report.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
