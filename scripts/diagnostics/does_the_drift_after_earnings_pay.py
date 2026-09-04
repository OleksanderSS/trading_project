"""Post-earnings drift, on data we already have, with the design fixed first.

WHY THIS AND NOT A BACKFILL. The recommendation this replaces was "backfill
the sources that have no pre-seal history". Measurement killed it twice:

    FRED credit spreads   backfillable, and worthless -- they are market-wide
                          series, so they land in "no cross-sectional
                          variation" however deep they go. They can never be
                          capable columns.

    SEC filings           needs new collector code: it reads `filings.recent`
                          only, and older batches live in `filings.files`
                          (documented in collectors.yaml, not implemented).
                          And it would yield filing DATES, not surprises.

Then the third measurement made both moot: the fundamentals are already deep.
`fund_days_since_report_1d` has 286,383 rows outside the seal, from 2009-04-30
across 95 names -- 4,480 report events, median 56 per name, which is 14 years
of four quarters. Nothing needs collecting.

WHY THE 235-COLUMN SWEEP COULD NOT SEE THIS. R28 ranked every capable column
cross-sectionally every day and held for N days. That is a STANDING book. Drift
after an announcement is an EVENT book: a position opened because something
happened to that name on that date, held, then closed. A daily rank of
`fund_earnings_yield` cannot express it, which is why zero survivors there says
nothing about this.

THE HYPOTHESIS, FIXED BEFORE THE RUN. A name whose announcement moved its price
continues in that direction for weeks. The proxy for the surprise is the
announcement return itself -- standard where analyst estimates are absent, which
is our case -- so nothing is needed but prices and the report date.

    signal    sign of the return from the close before the event to the close
              AFTER it. The position opens at that later close, so the signal
              is fully known before any money moves.
    holds     5, 20, 40, 60 days. FOUR attempts, declared here, and the
              threshold below is computed from that number rather than chosen
              after seeing the answer.
    book      dollar-neutral across the names holding a position that day.
              Without this a degenerate signal is just the market: seven such
              columns scored net Sharpe 1.016 on 2026-09-04 against a constant
              opponent's 1.018 (CLAIMS R28).
    cost      the round trip charged on the day the position opens and again
              when it closes, on both legs.
    opponent  buy every name, printed FIRST, so no result can be read without
              it.

WHY THIS HORIZON IS THE ONE WORTH TESTING. Friction is 27.6% a year at daily
turnover and 1.4% at twenty days, so the gross Sharpe needed to clear the
honest bar falls from 7.31 to about 1.04 (CLAIMS R27). A four-times-a-year
signal is the only shape this cost structure can afford, and drift after an
announcement is the best-documented one that has that shape.

The sealed period is not touched.

    python scripts/diagnostics/does_the_drift_after_earnings_pay.py
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

from src.pipeline.sealed_period import SEAL_START  # noqa: E402
from src.targets.calculators.regression_calculator import (  # noqa: E402
    RegressionCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
COUNTER = "fund_days_since_report_1d"

#: Standard error of an annualised Sharpe over the explorable span.
#: 2009-2023 is about 14 years, so 1/sqrt(14) rather than the 0.193 the
#: 27-year measurements use. A shorter history is a WIDER error bar, and
#: borrowing the longer one would quietly lower the bar.
YEARS = 14.0
SHARPE_SE = 1.0 / math.sqrt(YEARS)


def thresholds(attempts: int) -> tuple[float, float]:
    """Bonferroni at family-wise 5%, and the expected maximum of pure noise."""
    bonferroni = float(norm.ppf(1.0 - 0.025 / attempts)) * SHARPE_SE
    if attempts <= 1:
        return bonferroni, float(norm.ppf(0.95)) * SHARPE_SE
    log_n = math.log(attempts)
    root = math.sqrt(2.0 * log_n)
    gumbel = root - (math.log(log_n) + math.log(4.0 * math.pi)) / (2.0 * root)
    return bonferroni, max(gumbel, float(norm.ppf(0.95))) * SHARPE_SE


def _panel() -> pd.DataFrame:
    frame = pd.read_parquet(
        BATCH / "features.parquet",
        columns=["ticker", "datetime", "interval", "close", COUNTER])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[(frame["interval"] == "1d")
                  & (frame["datetime"] < SEAL_START)
                  & frame["close"].notna()]
    return frame.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def _sharpe(daily: np.ndarray) -> float:
    usable = daily[np.isfinite(daily)]
    if usable.size < 60 or usable.std() <= 0:
        return float("nan")
    return float(usable.mean() / usable.std() * math.sqrt(252.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holds", type=int, nargs="+", default=[5, 20, 40, 60])
    args = parser.parse_args()

    frame = _panel()
    costs = yaml.safe_load(
        (PROJECT_ROOT / "src/config/targets.yaml").read_text(encoding="utf-8")
    )["targets"]["target_return_1d"]["params"]["transaction_costs"]
    frame["friction"] = np.asarray(
        RegressionCalculator._round_trip_cost(frame["close"], costs), dtype=float)

    by_name = frame.groupby("ticker", sort=False)
    frame["ret"] = by_name["close"].transform(lambda s: s / s.shift(1) - 1.0)
    frame["row"] = by_name.cumcount()

    # An event is the day the counter resets downward: the previous bar was
    # further from a report than this one is.
    previous = by_name[COUNTER].shift(1)
    frame["event"] = (frame[COUNTER].notna() & previous.notna()
                      & (frame[COUNTER] < previous))

    events = frame.index[frame["event"]].to_numpy()
    print(f"panel      {len(frame):,} daily rows, {frame['ticker'].nunique()} names, "
          f"{str(frame['datetime'].min())[:10]} to {str(frame['datetime'].max())[:10]}")
    print(f"events     {len(events):,} report events, "
          f"{frame.loc[events, 'ticker'].nunique()} names\n")

    dates = np.sort(frame["datetime"].unique())
    date_index = {stamp: i for i, stamp in enumerate(dates)}
    frame["day"] = frame["datetime"].map(date_index).to_numpy()

    ticker_codes = pd.factorize(frame["ticker"])[0]
    frame["name"] = ticker_codes
    n_days, n_names = len(dates), ticker_codes.max() + 1

    returns = np.full((n_days, n_names), np.nan)
    returns[frame["day"], frame["name"]] = frame["ret"].to_numpy()
    friction = np.full((n_days, n_names), np.nan)
    friction[frame["day"], frame["name"]] = frame["friction"].to_numpy()

    # THE CONSTANT OPPONENT, printed before anything else.
    own_everything = np.where(np.isfinite(returns), returns, np.nan)
    constant_daily = np.nanmean(own_everything, axis=1)
    print(f"{'BUY EVERYTHING (the opponent)':<34}{_sharpe(constant_daily):>9.3f}"
          f"   annualised Sharpe, no turnover cost\n")

    header = (f"{'hold':>6}{'events used':>14}{'avg names held':>16}"
              f"{'gross Sharpe':>14}{'NET Sharpe':>12}{'net ann.ret':>13}")
    print(header)
    print("-" * len(header))

    results = {}
    for hold in args.holds:
        position = np.zeros((n_days, n_names))
        traded = np.zeros((n_days, n_names))
        used = 0
        for index in events:
            day = int(frame.at[index, "day"])
            name = int(frame.at[index, "name"])
            # The announcement return spans the bar before the event and the
            # bar after it; the position opens at the LATER close, so the
            # signal is known before any money moves.
            if day + 1 >= n_days or day - 1 < 0:
                continue
            before = returns[day, name]
            after = returns[day + 1, name]
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            side = np.sign((1.0 + before) * (1.0 + after) - 1.0)
            if side == 0:
                continue
            entry, exit_ = day + 2, min(day + 2 + hold, n_days)
            if entry >= n_days:
                continue
            position[entry:exit_, name] = side
            traded[entry, name] = 1.0
            if exit_ < n_days:
                traded[exit_ - 1, name] += 1.0
            used += 1

        # Dollar-neutral among the names held that day: without this a signal
        # that happens to be mostly long IS the market (R28).
        held = position != 0
        counts = held.sum(axis=1)
        means = np.divide(position.sum(axis=1), np.maximum(counts, 1))
        neutral = np.where(held, position - means[:, None], 0.0)
        gross_exposure = np.abs(neutral).sum(axis=1)
        scaled = np.divide(neutral, np.maximum(gross_exposure, 1e-12)[:, None])

        safe_returns = np.where(np.isfinite(returns), returns, 0.0)
        safe_friction = np.where(np.isfinite(friction), friction, 0.0)
        gross_daily = (scaled * safe_returns).sum(axis=1)
        cost_daily = (np.abs(scaled) * traded * safe_friction).sum(axis=1)
        net_daily = gross_daily - cost_daily

        live = counts > 0
        results[hold] = _sharpe(net_daily[live])
        print(f"{hold:>6}{used:>14,}{counts[live].mean():>16.1f}"
              f"{_sharpe(gross_daily[live]):>14.3f}{results[hold]:>12.3f}"
              f"{net_daily[live].mean() * 252:>12.2%}")

    attempts = len(args.holds)
    bonferroni, noise_max = thresholds(attempts)
    best = max(results, key=lambda h: (results[h] if np.isfinite(results[h]) else -9))

    print("\n" + "=" * len(header))
    print(f"attempts, declared before the run            {attempts}")
    print(f"standard error over {YEARS:.0f} years                  {SHARPE_SE:.3f}")
    print(f"expected maximum of {attempts} noise draws           {noise_max:.3f}")
    print(f"Bonferroni family-wise 5%                    {bonferroni:.3f}")
    print(f"best net Sharpe                              {results[best]:+.3f}"
          f"  (hold {best})")
    if results[best] >= bonferroni:
        print("\nCLEARS Bonferroni. This is the first candidate the project has "
              "had.\nNext step is NOT another variant: it is one pre-registered "
              "confirmation on the\nsealed period, which is what the seal was "
              "bought for.")
    elif results[best] >= noise_max:
        print("\nAbove the noise maximum, below Bonferroni. Worth stating as a "
              "measured\nnumber, not as a candidate.")
    else:
        print("\nBelow what noise alone gives from this many attempts. The "
              "drift is not\ntradeable in this universe at this cost, and "
              "trying more variants here is\nspending attempts on a null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
