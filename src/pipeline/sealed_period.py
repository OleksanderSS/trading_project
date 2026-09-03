"""The stretch of history nobody is allowed to look at yet.

Every measurement spends the data it reads. On 2026-08-29 the leading-feature
report was run repeatedly against the post-2018 slice, the conditional report
before it, and the invariant checks over everything -- so by the end of that
day the batch contained no unseen period at all. A "holdout" that has been
consulted a dozen times is a training set with a different name.

So a period is sealed by declaration and enforced by code: the diagnostics
truncate their input here, print that they did, and cannot be talked out of it
by a flag that is easy to type. What is measured before the seal is
exploration, however careful. The sealed stretch buys exactly one honest
confirmation, and only if it stays untouched until there is something worth
confirming.

Chosen 2026-08-29: three years, roughly 750 sessions across 110 names --
enough to confirm a cross-sectional coefficient, small enough to leave the
1996-2023 stretch for exploration. Moving this date EARLIER is always safe.
Moving it later destroys the guarantee and must be recorded in the register
with the reason.
"""

from __future__ import annotations

import pandas as pd

#: Nothing at or after this timestamp may be read during exploration.
#:
#: Kept for the DAILY frame, which is what it was chosen for.
SEAL_START = pd.Timestamp("2023-09-01", tz="UTC")

#: Share of a frame's own timeline that stays sealed.
#:
#: An absolute date cannot serve frames of different lengths. Measured
#: 2026-08-30, one day after the seal was declared:
#:
#:     1d   1996-08-26 .. 2026-08-28    11.6% sealed
#:     60m  2024-08-19 .. 2026-08-28   100.0% sealed
#:     15m  2026-06-09 .. 2026-08-28   100.0% sealed
#:
#: Two frames of three were sealed WHOLE, because their history is shorter
#: than the distance back to the chosen date. That does not protect them, it
#: deletes them: `opponent_ladder.py --interval 15m` returned "159,149 rows
#: withheld; 0 remain". A seal that leaves nothing to explore is not a
#: stricter seal, it is a broken one.
#:
#: A fraction of each frame's own span holds the same promise -- an unseen
#: tail nobody has measured on -- at every length.
SEAL_SHARE = 0.20

#: When the seal was declared, so a later reader can tell how much of the
#: sealed period was already visible in work that predates it.
SEALED_ON = pd.Timestamp("2026-08-29", tz="UTC")


def seal_start_for(stamps: pd.Series) -> pd.Timestamp:
    """Where this particular frame's seal begins.

    The later of the absolute date and the frame's own tail, so the daily
    frame keeps exactly the seal it was given while a short intraday frame
    still gets one. Uses distinct timestamps rather than rows: a frame holds
    110 names per bar, and sealing by row count would move with the number of
    tickers rather than with time.
    """
    stamps = pd.to_datetime(stamps, errors="coerce", utc=True).dropna()
    if stamps.empty:
        return SEAL_START
    distinct = pd.Series(stamps.unique()).sort_values()
    by_span = distinct.quantile(1.0 - SEAL_SHARE)
    return max(SEAL_START, pd.Timestamp(by_span)) if distinct.iloc[-1] >= SEAL_START         else pd.Timestamp(by_span)


def apply_seal(frame: pd.DataFrame, column: str = "datetime",
               allow_sealed: bool = False) -> tuple[pd.DataFrame, int]:
    """Drop rows at or after the seal. Returns the frame and how many went.

    `allow_sealed` exists for the single confirmation run, and the caller has
    to say so out loud in its own output.
    """
    if allow_sealed or column not in frame.columns:
        return frame, 0
    stamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
    keep = stamps < SEAL_START
    return frame.loc[keep], int((~keep).sum())


def describe() -> str:
    return (f"sealed from {SEAL_START.date()} onward "
            f"(declared {SEALED_ON.date()}); exploration sees earlier data only")
