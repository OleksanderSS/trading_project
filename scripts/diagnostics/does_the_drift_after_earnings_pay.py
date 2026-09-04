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


#: A quarterly EPS row, decided by the MEASURED period length rather than by
#: the `fiscal_period` label. The label cannot be trusted: measured 2026-09-04,
#: rows tagged FY have a median duration of 91 days -- a quarter -- while rows
#: tagged Q2 and Q3 sit at 167 and 165, which is a half-year. Building a
#: seasonal surprise on the label would silently subtract a half-year figure
#: from a quarterly one.
QUARTER_DAYS = (80, 100)

#: Quarters of seasonal differences used for the scaling deviation. Eight is
#: the standard window in the Foster-Olsen-Shevlin formulation and is fixed
#: here BEFORE the run so it cannot become a parameter chosen after seeing the
#: answer.
SUE_WINDOW = 8


def _quarterly_eps() -> pd.DataFrame:
    """First-filed quarterly diluted EPS per (ticker, period_end).

    `filed` is the only honest date -- the figure is private until the filing
    lands -- and the FIRST filing is the point-in-time one: a later 10-Q
    restates the same quarter as a comparative, and up to 33 filings mention a
    single quarter. Taking the earliest is what stops a restatement from
    rewriting history the book could not have known.
    """
    import duckdb

    connection = duckdb.connect(str(PROJECT_ROOT / "data" / "trading_data.duckdb"),
                                read_only=True)
    frame = connection.execute(
        """
        select ticker, period_end, min(filed) as filed, min(value) as eps
        from sec_fundamentals
        where concept = 'EarningsPerShareDiluted'
          and period_start is not null
          and date_diff('day', period_start, period_end) between ? and ?
        group by ticker, period_end
        """, list(QUARTER_DAYS)).fetch_df()
    connection.close()
    frame["filed"] = pd.to_datetime(frame["filed"], utc=True)
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True)
    return frame.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def _sue_events() -> pd.DataFrame:
    """Standardised unexpected earnings, seasonal-random-walk definition.

    surprise = EPS(q) - EPS(q-4), scaled by the deviation of the previous
    SUE_WINDOW surprises for that name. No analyst estimates: this is the
    classical formulation precisely because it needs only reported earnings,
    which SEC XBRL gives away.
    """
    eps = _quarterly_eps()
    eps["prior_year"] = eps.groupby("ticker")["period_end"].shift(4)
    eps["eps_lag4"] = eps.groupby("ticker")["eps"].shift(4)
    # The lag must actually be a year: a name with a missing quarter would
    # otherwise have its "same quarter last year" silently be a different one.
    gap = (eps["period_end"] - eps["prior_year"]).dt.days
    eps.loc[(gap < 330) | (gap > 400), "eps_lag4"] = np.nan
    eps["surprise"] = eps["eps"] - eps["eps_lag4"]
    # Scaled by PAST surprises only -- shift(1) before rolling, or the quarter
    # being judged helps set its own yardstick.
    eps["scale"] = (eps.groupby("ticker")["surprise"]
                    .transform(lambda s: s.shift(1).rolling(SUE_WINDOW, min_periods=4).std()))
    eps["sue"] = eps["surprise"] / eps["scale"].replace(0.0, np.nan)
    return eps.dropna(subset=["sue"])


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
    parser.add_argument("--skip-bars", type=int, default=2,
                        help="extra bars between the signal becoming public "
                             "and the position earning anything. `filed` is "
                             "stored with its time truncated to midnight, so "
                             "one bar cannot be shown to be enough: if the "
                             "filing landed after the close, the first bar's "
                             "return IS the announcement jump. Default 2 -- "
                             "the conservative reading, chosen because the "
                             "stored time is unusable, NOT because of what it "
                             "does to the answer. At 1 the best net Sharpe is "
                             "0.642 and at 2 it is 0.250, so the default is "
                             "the one that costs us the headline.")
    parser.add_argument("--weight", choices=["sign", "rank"], default="sign",
                        help="sign: +-1, which throws the magnitude away. "
                             "rank: the surprise's cross-sectional rank within "
                             "its own filing QUARTER, mapped to [-1, +1]. The "
                             "quarter is the natural unit for earnings, so this "
                             "adds no tuned window.")
    parser.add_argument("--signal", choices=["announcement", "sue"],
                        default="announcement",
                        help="announcement: sign of the return around the "
                             "report. sue: standardised unexpected earnings "
                             "from reported EPS, no analyst estimates needed.")
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
    # SURVIVORSHIP: these are today's names carried back, so this number is
    # an upper bound and NOT what the market gave. Measured 2026-09-04: the
    # 1996-2003 slice returns 20.55% a year at Sharpe 1.144, through the
    # dot-com crash, and only 61 of the 110 names existed in 1996. Valid as
    # a RELATIVE opponent -- both books trade the same names -- and
    # misleading as a market benchmark (CLAIMS R34).
    print(f"{chr(32)*34}survivorship-inflated: an upper bound, not the market")

    header = (f"{'hold':>6}{'events used':>14}{'avg names held':>16}"
              f"{'gross Sharpe':>14}{'NET Sharpe':>12}{'net ann.ret':>13}")
    print(header)
    print("-" * len(header))

    # THE SIGNAL, built once. Both variants produce (entry day, name, side)
    # with the side known strictly before the entry bar.
    name_of = dict(zip(frame["ticker"], frame["name"]))
    signals: list[tuple[int, int, float]] = []
    if args.signal == "announcement":
        for index in events:
            day = int(frame.at[index, "day"])
            name = int(frame.at[index, "name"])
            if day + 1 >= n_days or day - 1 < 0:
                continue
            before, after = returns[day, name], returns[day + 1, name]
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            side = float(np.sign((1.0 + before) * (1.0 + after) - 1.0))
            if side and day + 2 < n_days:
                signals.append((day + 2, name, side))
    else:
        sue = _sue_events()
        sue = sue[sue["filed"] < SEAL_START]
        print(f"SUE        {len(sue):,} quarters with a surprise, "
              f"{sue['ticker'].nunique()} names, "
              f"{str(sue['filed'].min())[:10]} to "
              f"{str(sue['filed'].max())[:10]}")
        print()
        # searchsorted compares raw datetime64, so both sides must be naive
        # UTC. Mixing an aware Timestamp with a naive array raises rather than
        # silently shifting, which is the good outcome -- but it has to be
        # handled, not caught.
        naive_dates = pd.DatetimeIndex(dates).tz_localize(None).to_numpy()             if pd.DatetimeIndex(dates).tz is not None else np.asarray(dates)
        if args.weight == "rank":
            # Rank WITHIN the filing quarter: every name in that reporting
            # season is compared with its peers, which is what "surprise
            # relative to others" means. Centred on 0 so the book is already
            # balanced before the daily demeaning.
            quarter = sue["filed"].dt.to_period("Q")
            sue = sue.assign(
                weight=2.0 * (sue.groupby(quarter)["sue"].rank(pct=True) - 0.5))
            small = sue.groupby(quarter)["sue"].transform("size") < 5
            # A quarter with four names cannot rank anything: the extremes are
            # the whole sample. Dropped rather than ranked on nothing.
            sue = sue[~small]
            print(f"           ranked within {quarter.nunique()} filing "
                  f"quarters; {int(small.sum())} events dropped from quarters "
                  f"with fewer than 5 names")
            print()
        for row in sue.itertuples():
            name = name_of.get(row.ticker)
            if name is None:
                continue
            # Enter on the first trading bar STRICTLY after the filing lands.
            filed = row.filed.tz_convert("UTC").tz_localize(None).to_datetime64()
            day = int(np.searchsorted(naive_dates, filed, side="right"))
            day += args.skip_bars - 1
            side = (float(row.weight) if args.weight == "rank"
                    else float(np.sign(row.sue)))
            if side and day < n_days:
                signals.append((day, int(name), side))

    results = {}
    for hold in args.holds:
        position = np.zeros((n_days, n_names))
        traded = np.zeros((n_days, n_names))
        used = 0
        for entry, name, side in signals:
            exit_ = min(entry + hold, n_days)
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

    # ATTEMPTS ARE CUMULATIVE ALONG A LINE OF ENQUIRY, not per run. The SUE
    # book has now been tried two ways -- by sign and by rank -- and both are
    # searches for the same thing, so the family is eight rather than four.
    # Counting each run separately is how a threshold gets quietly halved.
    attempts = len(args.holds) * (2 if args.signal == "sue" else 1)
    bonferroni, noise_max = thresholds(attempts)
    best = max(results, key=lambda h: (results[h] if np.isfinite(results[h]) else -9))

    print("\n" + "=" * len(header))
    print(f"attempts in this family                      {attempts}"
          + ("   (4 holds x sign and rank, cumulative)"
             if args.signal == "sue" else "   (declared before the run)"))
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
