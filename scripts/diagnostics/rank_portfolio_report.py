"""Is a signal this small worth money? Ask the portfolio, not the gate.

Stage 4 asks each model to beat a baseline on R2, per ticker. That is the wrong
instrument for what the features actually carry. Measured on the 2026-08-22
batch, the strongest out-of-sample rank correlation between a feature and
`target_relative_return_5d` is about 0.02-0.03. An IC of 0.03 is an R2 of
roughly 0.0009 -- no regression will show it, and a model with any capacity
will overfit and land at -0.01, which is exactly what all 22 tickers did.

A signal that size is not used by predicting a number. It is used by RANKING:
sort the names every day, hold the top against the bottom, and let breadth do
the work that accuracy cannot. This measures that directly, in money, with no
gate involved.

The rules that keep it honest, since every one of them has been got wrong here
before:

  selection    features are chosen on the TRAIN slice only, by |IC|, with the
               count fixed in advance. Choosing them on the whole sample and
               then "testing" is how a backtest lies.
  direction    the sign each feature is used with also comes from train.
  standardise  cross-sectionally, per date. A z-score over the whole sample
               would leak the future into today's ranking.
  outcome      `target_relative_return_5d` IS the forward 5-day return already
               demeaned across names, so a long-short book's return is just
               the difference of the two legs' means. Nothing is re-derived.
  overlap      a 5-day target measured every day overlaps four days in five.
               The mean is still unbiased, but the t-statistic is not, so the
               non-overlapping series is reported beside it.
  costs        from src/config/targets.yaml, applied to measured turnover.

    python scripts/diagnostics/rank_portfolio_report.py
    python scripts/diagnostics/rank_portfolio_report.py --features 20 --side 5
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import spearmanr

FEATURES = Path("data/colab/accumulated/main_database/features.parquet")
TARGETS = Path("data/colab/accumulated/main_database/targets.parquet")

TARGET = "target_relative_return_5d"
HORIZON = 5
TRAIN_FRACTION = 0.70
BLOCK = 250

#: Rows a feature needs on the daily frame before it is considered at all.
MIN_COVERAGE = 500


def _daily_targets() -> pd.DataFrame:
    frame = pd.read_parquet(
        TARGETS, columns=["datetime", "ticker", "interval", TARGET]
    )
    frame = frame[frame["interval"].astype(str).eq("1d")].reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    return frame


def _daily_mask() -> np.ndarray:
    intervals = pd.read_parquet(FEATURES, columns=["interval"])
    return intervals["interval"].astype(str).eq("1d").to_numpy()


def _train_ic(mask: np.ndarray, target: pd.Series, is_train: np.ndarray,
              names: list[str]) -> pd.DataFrame:
    """Rank correlation of every feature with the target, on TRAIN only."""
    y = target.to_numpy(dtype=float)
    found = []
    for start in range(0, len(names), BLOCK):
        columns = names[start:start + BLOCK]
        try:
            block = pd.read_parquet(FEATURES, columns=columns)
        except (OSError, ValueError):
            continue
        block = block.loc[mask].reset_index(drop=True)
        for name in columns:
            if name not in block.columns:
                continue
            values = pd.to_numeric(block[name], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).sum() < MIN_COVERAGE or np.nanstd(values) == 0:
                continue
            usable = is_train & np.isfinite(values) & np.isfinite(y)
            if usable.sum() < MIN_COVERAGE:
                continue
            ic = spearmanr(values[usable], y[usable]).statistic
            if np.isfinite(ic):
                found.append({"feature": name, "ic": float(ic)})
        del block
    return pd.DataFrame(found)


def _composite(chosen: pd.DataFrame, mask: np.ndarray, dates: pd.Series,
               tickers: pd.Series, within_ticker: bool) -> pd.Series:
    """Mean of z-scores, signed by the train IC.

    The choice of what to standardise against decides what is being measured,
    and getting it wrong produces a spectacular and completely false result.

    Cross-sectional only (`within_ticker=False`) asks "which name is high
    today". Features like price level or ATR differ between names PERMANENTLY,
    so the ranking barely moves: run this way on the 2018-2026 holdout it held
    NVDA on 100% of days against SPY on 100% of days, replaced 8.3% of the book
    between formations, and returned 34.9% a year at t=4.46. That is not a
    signal. It is long NVDA short SPY, held for eight years, discovered after
    the fact -- one bet, not 422 independent ones, and the t-statistic is
    meaningless because every "period" holds the same position.

    Within-ticker first (`within_ticker=True`) asks the question that can
    actually be traded: is this name unusual FOR ITSELF right now. Each
    feature is standardised against that ticker's own past with an EXPANDING
    window -- never a full-sample mean, which would put the future into
    today's score -- and only then compared across names.
    """
    block = pd.read_parquet(FEATURES, columns=list(chosen["feature"]))
    block = block.loc[mask].reset_index(drop=True)
    total = pd.Series(0.0, index=block.index)
    count = pd.Series(0.0, index=block.index)
    for _, row in chosen.iterrows():
        values = pd.to_numeric(block[row["feature"]], errors="coerce")
        if within_ticker:
            by_ticker = values.groupby(tickers)
            # shift(1) so today's own value is not in its own baseline.
            mean = by_ticker.transform(lambda s: s.shift(1).expanding(60).mean())
            spread = by_ticker.transform(lambda s: s.shift(1).expanding(60).std())
            values = (values - mean) / spread.replace(0.0, np.nan)
        grouped = values.groupby(dates)
        spread = grouped.transform("std")
        z = (values - grouped.transform("mean")) / spread.replace(0.0, np.nan)
        z = z.clip(-3, 3) * np.sign(row["ic"])
        total = total.add(z.fillna(0.0), fill_value=0.0)
        count = count.add(z.notna().astype(float), fill_value=0.0)
    return total / count.replace(0.0, np.nan)


def _round_trip_cost(notional: float, price: float = 150.0) -> float:
    config = yaml.safe_load(io.open("src/config/targets.yaml", encoding="utf-8"))

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                got = find(value, key)
                if got is not None:
                    return got
        return None

    profiles = find(config, "cost_profiles") or {}
    if not profiles:
        return float("nan")
    profile = next(iter(profiles.values()))
    friction = 2 * (float(profile.get("spread_pct", 0.0))
                    + float(profile.get("slippage_pct", 0.0)))
    if profile.get("model") == "per_share":
        fee = float(profile["per_share_fee"]) * (notional / price)
        fee = max(fee, float(profile.get("min_fee_per_order", 0.0)))
        return 2 * fee / notional + friction
    return 2 * float(profile.get("commission_pct", 0.0)) + friction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=20,
                        help="how many features to combine, fixed in advance")
    parser.add_argument("--side", type=int, default=5,
                        help="names held long, and the same number short")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--cross-sectional-only", action="store_true",
                        help="the WRONG construction, kept so the difference "
                             "can be shown rather than asserted")
    args = parser.parse_args()

    if not FEATURES.exists() or not TARGETS.exists():
        print("No batch on disk; run --mode prepare first.")
        return 1

    targets = _daily_targets()
    cut = targets["datetime"].quantile(TRAIN_FRACTION)
    is_train = (targets["datetime"] <= cut).to_numpy()
    print(f"daily rows {len(targets):,} | split {cut.date()} | "
          f"train {is_train.sum():,}, holdout {(~is_train).sum():,}")
    print(f"{targets['ticker'].nunique()} names\n")

    mask = _daily_mask()
    names = [
        name for name in pq.ParquetFile(FEATURES).schema_arrow.names
        if name not in {"datetime", "ticker", "interval"}
        and not name.startswith("target_")
    ]

    ic = _train_ic(mask, targets[TARGET], is_train, names)
    if ic.empty:
        print("No feature had usable coverage on the daily frame.")
        return 1
    chosen = ic.reindex(ic["ic"].abs().sort_values(ascending=False).index)
    chosen = chosen.head(args.features)
    print(f"=== {len(chosen)} features chosen on TRAIN only ===")
    for _, row in chosen.head(10).iterrows():
        print(f"  {row['feature'][:44]:44s} IC {row['ic']:+.4f}")
    if len(chosen) > 10:
        print(f"  ... and {len(chosen) - 10} more")
    print()

    signal = _composite(chosen, mask, targets["datetime"], targets["ticker"],
                        within_ticker=not args.cross_sectional_only)
    book = pd.DataFrame({
        "datetime": targets["datetime"],
        "ticker": targets["ticker"],
        "signal": signal.to_numpy(),
        "outcome": targets[TARGET].to_numpy(),
        "train": is_train,
    }).dropna(subset=["signal", "outcome"])

    holdout = book[~book["train"]]
    if holdout.empty:
        print("Nothing in the holdout slice.")
        return 1

    daily = []
    for stamp, group in holdout.groupby("datetime"):
        if len(group) < 2 * args.side:
            continue
        ordered = group.sort_values("signal")
        short_leg = ordered.head(args.side)["outcome"].mean()
        long_leg = ordered.tail(args.side)["outcome"].mean()
        daily.append({
            "datetime": stamp,
            "long_short": long_leg - short_leg,
            "long_only": long_leg,
            "passive": group["outcome"].mean(),
            "turnover_names": set(ordered.tail(args.side)["ticker"]),
        })
    series = pd.DataFrame(daily).sort_values("datetime").reset_index(drop=True)
    if series.empty:
        print(f"No date had {2 * args.side} names with a signal.")
        return 1

    # Turnover: how much of the long book is replaced from one formation to the
    # next. Costs follow from this, not from an assumed rebalance schedule.
    changes = [
        len(b - a) / max(len(b), 1)
        for a, b in zip(series["turnover_names"], series["turnover_names"][1:])
    ]
    turnover = float(np.mean(changes)) if changes else 0.0

    notional = args.capital / (2 * args.side)
    cost = _round_trip_cost(notional)
    # A book formed daily but held five days turns over roughly once per
    # horizon; charge the measured name-replacement rate over that period.
    cost_per_period = cost * turnover
    periods_per_year = 252 / HORIZON

    concentration = (
        pd.Series([n for names in series["turnover_names"] for n in names])
        .value_counts(normalize=True) * args.side
    )
    print("=== how static is the book ===")
    print("  most-held long names (share of dates in the book):")
    for name, share in concentration.head(4).items():
        print(f"    {name:6s} {share:5.0%}")
    if concentration.iloc[0] > 0.9:
        print("  A name held on essentially every date is not a signal, it is a")
        print("  position. Whatever this earns is that one name's history.")
    print()
    print(f"=== holdout: {len(series)} formation dates, "
          f"{series['datetime'].min().date()} to {series['datetime'].max().date()} ===\n")
    print(f"{'book':12s} {'mean per 5d':>13s} {'annualised':>12s} "
          f"{'hit rate':>10s} {'t (overlap)':>12s}")
    print("-" * 64)
    for label, column in (("long-short", "long_short"),
                          ("long only", "long_only"),
                          ("passive", "passive")):
        values = series[column].to_numpy(dtype=float)
        mean = np.nanmean(values)
        annual = mean * periods_per_year
        hit = float(np.nanmean(values > 0))
        tstat = mean / (np.nanstd(values, ddof=1) / np.sqrt(len(values)))
        print(f"{label:12s} {mean:12.4%} {annual:11.2%} {hit:9.1%} {tstat:12.2f}")

    # The honest statistic: one observation per horizon, no overlap.
    stagger = series.iloc[::HORIZON]
    values = stagger["long_short"].to_numpy(dtype=float)
    tstat = np.nanmean(values) / (np.nanstd(values, ddof=1) / np.sqrt(len(values)))
    print()
    print(f"Non-overlapping ({len(stagger)} independent 5-day periods): "
          f"mean {np.nanmean(values):.4%}, t = {tstat:.2f}")
    print("  The overlapping t above is inflated -- four of every five days")
    print("  share their outcome with the day before. This one is the test.")
    print()

    print(f"=== costs, {args.capital:,.0f} split over {2 * args.side} names ===")
    print(f"  measured name replacement between formations: {turnover:.1%}")
    print(f"  round trip at {notional:,.0f} per name:        {cost:.3%}")
    print(f"  charged per 5-day period:                     {cost_per_period:.3%}")
    gross = np.nanmean(series["long_short"].to_numpy(dtype=float))
    net = gross - cost_per_period
    print()
    print(f"  gross per period {gross:+.4%}  ->  net {net:+.4%}  "
          f"({net * periods_per_year:+.2%} a year)")
    if net <= 0:
        print()
        print("  Net of costs this loses money. The signal is real and too")
        print("  small to pay for its own execution at this turnover.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
