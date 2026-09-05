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

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

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
               allow_sealed: bool = False,
               by: str | None = None) -> tuple[pd.DataFrame, int]:
    """Drop rows at or after the seal. Returns the frame and how many went.

    `allow_sealed` exists for the single confirmation run, and the caller has
    to say so out loud in its own output.

    USES THE PER-FRAME RULE, not the bare date (REGISTER #264). It used the
    bare `SEAL_START`, which is right for the daily frame and deletes a short
    one outright. Measured on the live batch 2026-09-05:

        1d    705,492 rows   absolute withholds  82,094   per-frame  82,094
        60m   380,938 rows   absolute withholds 380,938   per-frame  69,502

    One hundred percent of the hourly frame, and a diagnostic handed zero rows
    reports "nothing to measure" -- which reads as a statement about the
    market rather than about the seal. The module's own note has said so since
    the seal was declared: "a seal that leaves nothing to explore is not a
    stricter seal, it is a broken one".

    `by` seals each group on its own span -- pass "interval" for a frame that
    stacks several cadences, where one seal for all of them is the same
    mistake at a smaller scale.
    """
    if allow_sealed or column not in frame.columns:
        return frame, 0
    stamps = pd.to_datetime(frame[column], errors="coerce", utc=True)

    if by and by in frame.columns:
        keep = pd.Series(False, index=frame.index)
        for _, rows in frame.groupby(by, sort=False).groups.items():
            group = stamps.loc[rows]
            keep.loc[rows] = group < seal_start_for(group)
    else:
        keep = stamps < seal_start_for(stamps)

    return frame.loc[keep], int((~keep).sum())


def describe() -> str:
    return (f"sealed from {SEAL_START.date()} onward "
            f"(declared {SEALED_ON.date()}); exploration sees earlier data only")


def describe_for(stamps: pd.Series) -> str:
    """What was ACTUALLY withheld from this frame, which is not always the date.

    Caught 2026-09-05, one command after `conditional_pattern_report` was
    routed through `apply_seal`: it printed "sealed from 2023-09-01 onward"
    while keeping 60m rows through 2026-04-22. The sentence was false about
    the run that printed it -- the same defect family as the seal itself, a
    decision taken in one place and described by another that does not know
    about it. A reader checking the output would have concluded the hourly
    report was reading sealed data.
    """
    boundary = pd.Timestamp(seal_start_for(stamps))
    if boundary == SEAL_START:
        return describe()
    return (f"sealed from {boundary.date()} onward -- the last "
            f"{SEAL_SHARE:.0%} of THIS frame's own span, because its history "
            f"begins after the declared {SEAL_START.date()} and that date "
            f"would withhold all of it (declared {SEALED_ON.date()})")


def seal_and_describe(frame: pd.DataFrame, column: str = "datetime",
                      by: str | None = None,
                      allow_sealed: bool = False,
                      ) -> tuple[pd.DataFrame, int, str]:
    """Seal a frame and say what was done, in one call that cannot be misordered.

    Two functions in the right order is a convention, and this module exists
    because conventions do not hold. Caught immediately after `describe_for`
    was written: the first call site passed the ALREADY-SEALED frame, whose
    own tail is now the boundary, so it would have printed a date later than
    the one actually applied -- a wrong number produced by the very function
    added to stop wrong numbers.

    Returns the sealed frame, how many rows went, and one line per cadence.
    """
    if by and by in frame.columns:
        stamps = pd.to_datetime(frame[column], errors="coerce", utc=True)
        lines = [f"[{cadence}] {describe_for(stamps.loc[rows])}"
                 for cadence, rows in frame.groupby(by, sort=True).groups.items()]
        note = "\n".join(lines)
    else:
        note = describe_for(frame[column]) if column in frame.columns else describe()

    kept, withheld = apply_seal(frame, column=column, by=by,
                                allow_sealed=allow_sealed)
    if allow_sealed:
        note = "SEAL DELIBERATELY OPENED -- " + describe()
    return kept, withheld, note

#: The file that carries the seal to a machine without this repository.
#:
#: REGISTER #153. The Colab cell is a standalone 2,077-line file pasted into a
#: notebook, and its own comment describes the real workflow: "one real
#: workflow copies features.parquet and targets.parquet from a local machine
#: to Drive". In that workflow there IS no `src/`, so the cell cannot import
#: this module -- and it has been splitting off the last 20% of each series as
#: validation with no notion of a seal at all.
#:
#: Measured 2026-09-05: the batch it reads spans 1996-08-26 to 2026-09-01 and
#: holds 82,094 sealed daily rows (11.6%), and the last 20% of the daily rows
#: begins 2021-07-12 -- so the cell's validation window runs straight through
#: the sealed period. Champions selected there were selected on the one
#: un-spent confirmation this project has.
#:
#: Writing the date beside the parquets is the only route that survives the
#: copy. It is NOT a tenth definition (#264 found nine): the value comes from
#: SEAL_START above and nowhere else, and the reader is told to refuse rather
#: than guess when the file is absent.
SEAL_FILE = "sealed_period.json"


def export_to(directory: "Path | str") -> "Path | None":
    """Write the seal beside a batch, so a machine without src/ can honour it.

    Returns the path written, or None when the directory does not exist --
    which is logged rather than raised, because a batch write must not fail
    for want of a sidecar. The cell's refusal is what makes the absence
    matter; this end only has to try.
    """
    from pathlib import Path as _Path

    target = _Path(directory)
    if not target.is_dir():
        logger.warning(
            "Cannot write %s beside %s: the directory does not exist, so a "
            "reader without this repository has nothing to honour "
            "(REGISTER #153).", SEAL_FILE, target,
        )
        return None

    path = target / SEAL_FILE
    payload = {
        "seal_start": SEAL_START.isoformat(),
        "seal_share": SEAL_SHARE,
        "sealed_on": SEALED_ON.isoformat(),
        "source": "src/pipeline/sealed_period.py",
        "meaning": (
            "Rows at or after seal_start are the untouched holdout. Anything "
            "that trains, validates or SELECTS on them spends the one "
            "confirmation this project has. A reader that cannot find this "
            "file must refuse to train, not assume there is no seal."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Seal exported to %s (%s)", path, SEAL_START.date())
    return path
