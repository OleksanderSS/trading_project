"""The one surviving feature, turned into an actual equity curve.

R20 found exactly one feature of 455 that survives every check:
`CDL_UPPER_WICK_RATIO_1d`, IC 0.0166 per date, t 4.33 over 1,763 dates, four
quarters of four agreeing. What that is WORTH was then estimated through the
fundamental law -- and the estimate fell apart in both directions:

    breadth from RAW returns      rho_bar +0.2903 -> 3.37  -> IR 0.48
    breadth from RESIDUALS        rho_bar -0.0131 -> N-1   -> IR 2.75

The first understates it: a cross-sectional ranking book is market-neutral by
construction, so the correlation that governs its breadth is what remains after
the common factor, and on these names the common factor is ALL of it (-0.0131
is the mechanical artefact of demeaning, -1/(N-1) = -0.0092). The second
overstates it: 110 names are not 109 independent bets once costs, a
non-constant IC and estimation error are paid, and the fundamental law is
known for flattering exactly this quantity.

So stop estimating breadth. A realised curve needs no breadth term at all.

The book is built through `build_holdout_equity`, the pipeline's own money
path, rather than a second implementation -- that path takes
`position = sign(prediction)` and averages `position * actual` across every row
sharing a timestamp, which IS an equal-weight market-neutral portfolio when the
prediction is a cross-sectionally centred rank. It is also the path CLAIMS R9
calibrated against a planted edge of known size, and the one that has never
computed a profit on real data: every run so far ended at
`holdout_equity: no_return_targets`. This gives it its first real input.

THE SEALED PERIOD IS NOT TOUCHED. Everything here is exploration data, before
the date declared in docs/SEALED_HOLDOUT.md, so the sealed years remain
available for the one test that matters later.

    python scripts/diagnostics/book_from_one_feature.py
    python scripts/diagnostics/book_from_one_feature.py --feature RSI_14_1d --top-decile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.stages.evaluation.holdout_equity import (  # noqa: E402
    build_holdout_equity,
)
from src.pipeline.stages.evaluation.metrics_calculator import (  # noqa: E402
    EvaluationMetricsCalculator,
)

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
FEATURE = "CDL_UPPER_WICK_RATIO_1d"
TARGET = "target_return_1d"
SEALED_FROM = pd.Timestamp("2023-09-01", tz="UTC")


def _panel(feature: str) -> pd.DataFrame:
    ident = ["ticker", "datetime", "interval"]
    values = pd.read_parquet(BATCH / "features.parquet", columns=ident + [feature])
    values = values[values["interval"] == "1d"]
    outcome = pd.read_parquet(BATCH / "targets.parquet", columns=ident + [TARGET])
    outcome = outcome[outcome["interval"] == "1d"]
    frame = values.merge(outcome[["ticker", "datetime", TARGET]],
                         on=["ticker", "datetime"], how="inner")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame[frame["datetime"] < SEALED_FROM]
    return frame.dropna(subset=[feature, TARGET])


def _positions(frame: pd.DataFrame, feature: str, decile: bool) -> pd.DataFrame:
    """A cross-sectionally centred rank, so `sign` splits long from short.

    With `decile`, everything between the extremes is set to exactly zero and
    `sign` makes it flat -- a concentrated book rather than a median split.
    """
    ranks = frame.groupby("datetime")[feature].rank(pct=True)
    if decile:
        signal = np.where(ranks >= 0.9, 1.0, np.where(ranks <= 0.1, -1.0, 0.0))
    else:
        signal = (ranks - 0.5).to_numpy()
    return pd.DataFrame({
        "target": TARGET,
        "context": "BOOK::1d::" + feature,
        "ticker": frame["ticker"].to_numpy(),
        "datetime": frame["datetime"].to_numpy(),
        "prediction": signal,
        "actual": frame[TARGET].to_numpy(),
    })


def _report(label: str, predictions: pd.DataFrame) -> None:
    curve = build_holdout_equity(predictions)
    if curve.get("status") != "built":
        print(f"{label}: no curve -- {curve}")
        return
    metrics = EvaluationMetricsCalculator(None)._calculate_basic_metrics(
        curve["portfolio_history"]
    )
    held = int((predictions["prediction"] != 0).sum())
    print(f"\n=== {label} ===")
    print(f"  bars {curve['bar_count']:,}   positions taken {held:,} of "
          f"{len(predictions):,} rows")
    for key in ("sharpe_ratio", "total_return", "max_drawdown",
                "annualized_return", "volatility", "periods_per_year_used"):
        if key in metrics:
            print(f"  {key:<22}{metrics[key]:>12.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", default=FEATURE)
    parser.add_argument("--top-decile", action="store_true")
    args = parser.parse_args()

    frame = _panel(args.feature)
    print(f"{args.feature}: {len(frame):,} rows, "
          f"{frame['ticker'].nunique()} names, "
          f"{frame['datetime'].min().date()} -> {frame['datetime'].max().date()} "
          f"(sealed from {SEALED_FROM.date()} untouched)")

    _report("median split, long above / short below",
            _positions(frame, args.feature, decile=False))
    _report("deciles, long top 10% / short bottom 10%",
            _positions(frame, args.feature, decile=True))

    # The control that says whether the curve is the FEATURE or the period.
    rng = np.random.default_rng(20260903)
    shuffled = frame.copy()
    shuffled[args.feature] = shuffled.groupby("datetime")[args.feature].transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    _report("CONTROL: same feature shuffled within each date",
            _positions(shuffled, args.feature, decile=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
