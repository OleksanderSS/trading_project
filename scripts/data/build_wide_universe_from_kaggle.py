"""Build a wide daily panel from the Kaggle US equities dump, and measure the one thing that decides whether it is worth anything.

WHY A SECOND UNIVERSE AT ALL. R33 measured this project's ceiling and it is not
features: 110 names carry 3.16 INDEPENDENT bets, because the market factor eats
the rest. Every null verdict since then (R22, R23, R28, R52) was reached on
those 3.16 bets. Widening the cross-section is the only lever that moves that
number, and R46/R47 priced the honest way to do it -- prices for delisted names,
$630/yr -- which the owner has not bought.

WHAT THIS IS AND IS NOT. The Kaggle dump
(borismarjanovic/price-volume-data-for-all-us-stocks-etfs) is 14,390 US tickers
back to 1962 with split-adjusted closes. Measured 2026-09-07: of 896 randomly
sampled series, ALL 896 end on 2017-11-10 and not one dies earlier. It is a
SURVIVOR SNAPSHOT frozen at 2017 -- the same defect as our own universe on an
earlier date, and it does NOT fix survivorship. MER, LEH, ENE, IKN, BRL, BSC,
ABK, CFC, NCC: none present.

So nothing measured here is evidence about the market. What it IS good for is
the one number R33 named: whether a cross-section 47 times wider carries more
independent bets, or whether the market factor eats those too. If it does not
move, this universe buys nothing and the work stops here -- which is why the
independence measurement runs in this same script, before any book is built.

DECLARED BEFORE THE RUN, so none of it can be chosen after seeing an answer:

    window        1996-08-26 to 2017-11-10, the overlap with our own panel's
                  start, so the comparison with R33's 3.16 is like for like on
                  the time axis.
    history       at least MIN_DAYS trading days inside that window. A name
                  present for a month contributes noise to a correlation and
                  nothing to a book.
    liquidity     median dollar volume at least MIN_DOLLAR_VOLUME. Declared
                  here as a RESEARCH filter and deliberately NOT wired to
                  `liquidity_risk.min_volume_threshold` in the risk config,
                  which happens to hold the same number for a different
                  purpose: two things that coincide today would later read as
                  one thing on purpose.
    seal          this frame ends in 2017, so the project's 2023-09-01 seal
                  seals NOTHING of it. This universe gets its own by the same
                  rule short frames get: the last 20% of its own span,
                  computed below and printed before anything else.

    python scripts/data/build_wide_universe_from_kaggle.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.sealed_period import (  # noqa: E402
    SEAL_SHARE, apply_seal, seal_start_for,
)

SOURCE = Path.home() / (
    ".cache/kagglehub/datasets/borismarjanovic/"
    "price-volume-data-for-all-us-stocks-etfs/versions/3")
OUT = PROJECT_ROOT / "data" / "wide_universe" / "kaggle_us_daily.parquet"

WINDOW = ("1996-08-26", "2017-11-10")
MIN_DAYS = 1000
MIN_DOLLAR_VOLUME = 1_000_000.0

#: Names sampled for the correlation estimate. The average pairwise correlation
#: converges long before the full cross-section, and the full 5,000x5,000 matrix
#: is 200 million pairs to answer a question 500 names answer to two decimals.
CORRELATION_SAMPLE = 600


def _load() -> pd.DataFrame:
    # THE ARCHIVE CONTAINS EVERY FILE TWICE -- once under `Stocks/` and once
    # under `Data/Stocks/`. A plain rglob therefore reports 14,390 files for
    # 7,195 tickers, reads each one twice, and builds a panel that is exactly
    # 50% duplicate rows. Caught by the run's own arithmetic disagreeing with
    # itself: it said "kept 5,368" and then counted 2,684 names. The
    # correlation estimate was unharmed (pivot_table averages duplicates, and
    # the duplicates are identical), but the panel on disk was wrong and the
    # ticker count reported to the owner was double.
    seen: dict[str, Path] = {}
    for path in sorted(SOURCE.rglob("Stocks/*.txt")):
        seen.setdefault(path.name, path)
    files = list(seen.values())
    print(f"scanning {len(files):,} distinct tickers "
          f"({len(list(SOURCE.rglob('Stocks/*.txt'))):,} files, each present twice)")
    low, high = WINDOW
    frames, skipped_short, skipped_thin = [], 0, 0
    started = time.time()
    for position, path in enumerate(files):
        if position and position % 2000 == 0:
            print(f"    {position:,} / {len(files):,}  "
                  f"kept {len(frames):,}  ({time.time() - started:.0f}s)",
                  flush=True)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        rows = [r for r in text.split("\n")[1:] if low <= r[:10] <= high]
        if len(rows) < MIN_DAYS:
            skipped_short += 1
            continue
        dates, closes, volumes = [], [], []
        for row in rows:
            parts = row.split(",")
            if len(parts) < 6:
                continue
            try:
                closes.append(float(parts[4]))
                volumes.append(float(parts[5]))
            except ValueError:
                continue
            dates.append(parts[0])
        if len(dates) < MIN_DAYS:
            skipped_short += 1
            continue
        close = np.asarray(closes, dtype=float)
        volume = np.asarray(volumes, dtype=float)
        if float(np.median(close * volume)) < MIN_DOLLAR_VOLUME:
            skipped_thin += 1
            continue
        frames.append(pd.DataFrame({
            "ticker": path.name.split(".")[0].upper(),
            "datetime": pd.to_datetime(dates),
            "close": close,
            "volume": volume,
        }))
    print(f"    kept {len(frames):,}; dropped {skipped_short:,} for history, "
          f"{skipped_thin:,} for liquidity\n")
    return pd.concat(frames, ignore_index=True)


def _effective_bets(panel: pd.DataFrame, sample: int) -> tuple[float, float, int]:
    """Average pairwise correlation and the independent-bet count it implies.

    N / (1 + (N-1) * rho_bar) -- the same arithmetic R33 applied to the 110,
    so the two numbers are comparable. Computed on a sample of names because
    the average converges long before the full matrix does.
    """
    names = panel["ticker"].unique()
    generator = np.random.default_rng(31)
    chosen = generator.choice(names, size=min(sample, len(names)), replace=False)
    wide = (panel[panel["ticker"].isin(chosen)]
            .pivot_table(index="datetime", columns="ticker", values="close"))
    returns = wide.pct_change().replace([np.inf, -np.inf], np.nan)
    # Dates where fewer than thirty names trade cannot carry a cross-section.
    returns = returns[returns.notna().sum(axis=1) >= 30]
    matrix = returns.corr(min_periods=250).to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    rho = float(np.nanmean(upper))
    count = int(matrix.shape[0])
    return rho, count / (1 + (count - 1) * rho), count


def main() -> int:
    panel = _load()
    panel = panel.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    dates = panel["datetime"]
    print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique():,} names, "
          f"{dates.min().date()} to {dates.max().date()}")

    # THE PROJECT'S RULE, ASKED RATHER THAN REIMPLEMENTED. The first version
    # computed the percentile here with its own SEAL_SHARE -- a second copy of
    # a rule that already exists, which is exactly how the seal came to have
    # nine definitions on 2026-09-04. `seal_start_for` already handles this
    # case: when a frame ENDS before the declared date it falls back to the
    # frame's own tail instead of withholding nothing. It returns the same
    # 2013-08-16 the hand-rolled version did.
    seal = seal_start_for(dates)
    kept, withheld = apply_seal(panel)
    print(f"SEAL for this universe: {seal.date()}, from `seal_start_for` -- the "
          f"last {SEAL_SHARE:.0%} of its own span.\n    The project's 2023-09-01 "
          f"seal would hold back NOTHING here, since the frame\n    ends in 2017. "
          f"Explorable: {len(kept):,} rows; held back: {withheld:,}.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"written to {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)\n")

    print("=" * 78)
    print("\nTHE QUESTION THIS UNIVERSE EXISTS TO ANSWER (R33)\n")
    explorable = panel[panel["datetime"] < seal]
    rho, bets, counted = _effective_bets(explorable, CORRELATION_SAMPLE)
    print(f"    names in the correlation estimate      {counted:,}")
    print(f"    average pairwise correlation           {rho:.4f}")
    print(f"    EFFECTIVE INDEPENDENT BETS             {bets:.2f}")
    print(f"    ceiling 1/rho_bar                      {1 / rho:.2f}")
    print(f"\n    R33 measured the 110-name universe at 0.3104 and 3.16 bets.")
    print(f"    This one: {rho:.4f} and {bets:.2f} -- "
          f"{bets / 3.16:.1f}x the independence.\n")
    print("    If that multiple is near one, a wider cross-section buys nothing "
          "and the market\n    factor eats it, which is the answer R33 would "
          "predict. If it is large, the ceiling\n    this project has been "
          "hitting was the universe and not the features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
