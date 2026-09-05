"""Is there a clock to subtract, on the frame we are allowed to study?

REGISTER #195 says the clock should be a variable we REMOVE from the target,
not an opponent the model has to beat: `residual = target - E[target | hour,
weekday]`, and then any finding is by construction not the clock. The argument
is strong -- a signal worth 0.02 sitting on top of a clock worth 0.26 is
invisible while the model must clear 0.5615, and visible against a residual
whose baseline is 0.5.

But the clock that killed four of nine targets was the INTRADAY one, and #253
closed intraday research under the seal. So before doing the work the
precondition has to be measured: is there a clock on the DAILY frame at all?

Three questions, in the order that makes the third meaningful:

    1  How much of a raw daily return is the DATE itself -- the market factor
       common to every name that day? This is not a calendar effect; it is the
       thing a dollar-neutral book removes by construction, and it is measured
       first so the other two are read against it.

    2  After the date mean is removed, does the CALENDAR explain anything --
       weekday, month, turn-of-month, or the day's position in the week? This
       is the daily analogue of the intraday clock, and it is what #195 would
       residualise.

    3  On the TIME-SERIES side, where #195's argument was made: does the
       calendar alone beat a constant at predicting the sign of tomorrow's
       return? That is the opponent the ladder actually runs.

R-squared is reported against zero, and the share of variance is compared with
what pure noise would give for the same number of dummies -- k/n, which at
these sizes is small but not zero and would otherwise be read as signal.

    python scripts/diagnostics/is_there_a_clock_on_the_daily_frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.sealed_period import SEAL_START, describe  # noqa: E402

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = y - fitted
    total = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan")


def _group_means(y: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Fitted values of a one-way fixed effect, without building a design."""
    count = int(codes.max()) + 1
    total = np.bincount(codes, weights=y, minlength=count)
    rows = np.bincount(codes, minlength=count)
    return (total / np.maximum(rows, 1))[codes]


def main() -> int:
    frame = pd.read_parquet(
        BATCH / "features.parquet",
        columns=["ticker", "datetime", "interval", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    keep = ((frame["interval"] == "1d")
            & (frame["datetime"] < SEAL_START)
            & frame["close"].notna())
    frame = frame.loc[keep].sort_values(["ticker", "datetime"]).reset_index(drop=True)
    print(describe())

    frame["ret"] = (frame.groupby("ticker", sort=False)["close"]
                    .transform(lambda s: s.shift(-1) / s - 1.0))
    frame = frame.loc[frame["ret"].notna()].reset_index(drop=True)
    print(f"panel: {len(frame):,} rows, {frame['ticker'].nunique()} names, "
          f"{frame['datetime'].nunique():,} dates, "
          f"{frame['datetime'].min().date()} to {frame['datetime'].max().date()}\n")

    y = frame["ret"].to_numpy(dtype=float)
    date_codes = pd.factorize(frame["datetime"], sort=True)[0]

    # --- 1. the date itself ------------------------------------------------
    date_fit = _group_means(y, date_codes)
    date_r2 = _r_squared(y, date_fit)
    print("1. THE DATE (the market factor common to every name that day)")
    print(f"   R^2 of date fixed effects: {date_r2:.4f}")
    print(f"   noise would give about k/n = "
          f"{frame['datetime'].nunique() / len(frame):.4f}")
    print("   A dollar-neutral cross-sectional book removes exactly this by")
    print("   construction, so for the books measured in Р41 it is already gone.\n")

    residual = y - date_fit

    # --- 2. the calendar, after the date is removed ------------------------
    when = frame["datetime"].dt
    calendars = {
        "weekday": when.dayofweek.to_numpy(),
        "month": (when.month - 1).to_numpy(),
        "day of month": (when.day - 1).to_numpy(),
        "turn of month (last 3 / first 3)": np.where(
            when.day.to_numpy() >= 26, 0,
            np.where(when.day.to_numpy() <= 3, 1, 2)),
    }
    print("2. THE CALENDAR, on the residual after the date mean is removed")
    print(f"   {'effect':<34}{'levels':>8}{'R^2':>10}{'noise k/n':>12}")
    for label, codes in calendars.items():
        codes = codes.astype(np.int64)
        levels = int(codes.max()) + 1
        r2 = _r_squared(residual, _group_means(residual, codes))
        print(f"   {label:<34}{levels:>8}{r2:>10.5f}{levels / len(frame):>12.5f}")
    print("   Nothing here can survive the per-date demeaning either: a")
    print("   weekday is a property OF THE DATE, so it is inside the effect")
    print("   removed above. Measured anyway, because 'obviously zero' is the")
    print("   phrase that precedes most of this project's corrections.\n")

    # --- 3. the calendar as a time-series opponent -------------------------
    print("3. AS AN OPPONENT, on the raw return -- can the calendar predict")
    print("   the sign of tomorrow, the way the intraday clock did?")
    up = (y > 0).astype(float)
    base = float(up.mean())
    print(f"   base rate (always say 'up'): {max(base, 1 - base):.4f}")
    for label, codes in calendars.items():
        codes = codes.astype(np.int64)
        fitted = _group_means(up, codes)
        accuracy = float(((fitted > 0.5).astype(float) == up).mean())
        spread = float(np.ptp(_group_means(up, codes)))
        print(f"   {label:<34} accuracy {accuracy:.4f}   "
              f"spread across levels {spread:.4f}")

    print()
    print("READING IT. #195 proposes removing E[target | clock]. On this frame")
    print("the clock IS the date, and every book in Р41 already removes the")
    print("date. What is left for the calendar to explain is what these")
    print("numbers say -- and a residualisation that subtracts approximately")
    print("nothing changes approximately nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
