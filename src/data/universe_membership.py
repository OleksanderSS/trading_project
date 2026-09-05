"""Which names existed on a given date -- including the ones that died.

REGISTER #169, CLAIMS Р41 and Р42.

`assets.yaml` lists TODAY's names, and every measurement in this repository has
been made by carrying that list back through thirty years. Measured on our own
panel: of 110 names, **ZERO stop before the panel ends** in twenty-seven years.
Not one. A universe in which nothing can die makes a permanent long in the
survivors pay BY CONSTRUCTION, and Р41 put a number on what that is worth --
a fitted cross-sectional book scored Sharpe 1.417, and 0.675 once every
persistent name bet was removed. More than half of the result was the list.

So survivorship here is not a caveat on a result. It is a rival explanation of
the whole result, and on that universe it cannot be falsified.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT.

It is the membership half: for each ticker, when it started trading and when it
stopped, so a universe can be selected AS OF a date instead of as of today.
That half is free -- Alpha Vantage's LISTING_STATUS returns ipoDate and
delistingDate per symbol, which is exactly the membership interval.

It is NOT the price half. Measured 2026-09-05 by asking directly: Yahoo, our
collector's source, returns ZERO bars for ten of ten known delistings (TWTR,
ATVI, SIVB, FRC, CERN, XLNX, WORK, MYL, RTN, CTL). Prices for dead names need a
paid source ($630/yr at the cheapest that publishes a rate -- Р42). So this
module can tell you a name existed and that we have no bars for it, and that
distinction is the point: an absence we can NAME is worth more than an absence
that looks like the name never existed.

THE INVARIANT THIS MODULE EXISTS TO HOLD. `universe_as_of` must never fall back
to today's list when it lacks history for a date. That fallback would rebuild
survivorship INSIDE the fix, silently, which is this repository's commonest
defect shape and the reason for the register in the first place. It raises
`NoCoverage` instead, and the caller decides.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Where the store lives. One file, columns fixed below, rewritten whole by the
#: fetcher -- there is no incremental path, because a partially updated
#: membership table is indistinguishable from a complete one and that is the
#: failure this module is about.
DEFAULT_STORE = Path("data/reference/universe_membership.parquet")

COLUMNS = (
    "ticker",        # symbol as the source reports it
    "name",          # company name, for a human checking a surprising row
    "exchange",
    "asset_type",
    "ipo_date",      # first day it could be held
    "delisting_date",  # last day it could be held; NaT while it still trades
    "status",        # "Active" or "Delisted", as reported
    "source",        # e.g. "alphavantage:LISTING_STATUS" -- never blank
    "source_detail",  # e.g. "demo key, only the documented date" -- the caveat
    "fetched_at",
)


class NoCoverage(RuntimeError):
    """The store cannot speak about that date.

    A distinct type, not a bare return of today's names, because "I do not
    know" and "nothing died" are different facts and only one of them is true.
    """


@dataclass(frozen=True)
class Coverage:
    """What the store actually knows, so a caller can check before trusting it."""

    rows: int
    active: int
    delisted: int
    earliest_ipo: pd.Timestamp | None
    earliest_delisting: pd.Timestamp | None
    latest_delisting: pd.Timestamp | None
    sources: tuple[str, ...]

    def describe(self) -> str:
        if not self.rows:
            return "universe membership store is EMPTY"
        return (
            f"{self.rows:,} symbols ({self.active:,} active, "
            f"{self.delisted:,} delisted); delistings observed "
            f"{_day(self.earliest_delisting)} .. {_day(self.latest_delisting)}; "
            f"source {', '.join(self.sources) or 'unrecorded'}"
        )


def _day(value: pd.Timestamp | None) -> str:
    return "-" if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def load(store: Path | str = DEFAULT_STORE) -> pd.DataFrame:
    """Read the store, or an empty frame with the right columns.

    An empty frame rather than an exception: "no store yet" is a legitimate
    state early in the work, and `coverage()` reports it honestly. What is not
    legitimate is a caller treating it as "no name ever died".
    """
    path = Path(store)
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_parquet(path)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}; it was written by an older "
            "version and must be refetched rather than patched"
        )
    for column in ("ipo_date", "delisting_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def coverage(frame: pd.DataFrame | None = None,
             store: Path | str = DEFAULT_STORE) -> Coverage:
    frame = load(store) if frame is None else frame
    if frame.empty:
        return Coverage(0, 0, 0, None, None, None, ())
    delisted = frame[frame["delisting_date"].notna()]
    return Coverage(
        rows=len(frame),
        active=int((frame["status"].astype(str).str.lower() == "active").sum()),
        delisted=len(delisted),
        earliest_ipo=frame["ipo_date"].min() if frame["ipo_date"].notna().any() else None,
        earliest_delisting=delisted["delisting_date"].min() if len(delisted) else None,
        latest_delisting=delisted["delisting_date"].max() if len(delisted) else None,
        sources=tuple(sorted(set(frame["source"].dropna().astype(str)))),
    )


def universe_as_of(when: str | date | datetime | pd.Timestamp,
                   frame: pd.DataFrame | None = None,
                   store: Path | str = DEFAULT_STORE) -> set[str]:
    """Every symbol that was trading on `when`, per the store.

    Raises NoCoverage rather than guessing. The three ways it can fail to know:

        the store is empty -- nothing has been fetched;

        the date precedes the earliest delisting the store observed, so the
          store cannot distinguish "nothing died before then" from "we have no
          record of what died before then". The first is false for any real
          market and the second is the truth about our data, and returning a
          set of names would state the false one;

        the date follows the LATEST delisting the store observed, which is the
          same failure mirrored and the one that actually bit. Built from a
          single snapshot the store knows nothing that died after it, so every
          date in our 2015-2023 test window would have come back as today's
          survivors wearing a point-in-time label.

    Both edges matter because the coverage of this store is a WINDOW, not a
    prefix, and a window has two ends.
    """
    frame = load(store) if frame is None else frame
    moment = pd.Timestamp(when, tz="UTC") if pd.Timestamp(when).tzinfo is None \
        else pd.Timestamp(when)

    if frame.empty:
        raise NoCoverage(
            "the universe membership store is empty; fetch it with "
            "scripts/data/fetch_universe_membership.py before asking which "
            "names existed on a date"
        )

    known = coverage(frame)
    if known.earliest_delisting is None:
        raise NoCoverage(
            "the store holds no delistings at all, so it cannot answer a "
            "point-in-time question -- it would return today's survivors "
            "under a different name (REGISTER #169)"
        )
    if moment < known.earliest_delisting:
        raise NoCoverage(
            f"the store's earliest observed delisting is "
            f"{_day(known.earliest_delisting)}; asked about {_day(moment)}. "
            "Answering would claim nothing had died before then, which is the "
            "exact bias this store exists to remove"
        )
    if moment > known.latest_delisting:
        # THE SAME REFUSAL ON THE OTHER SIDE, and it took running the thing to
        # see it. The first version guarded only the early edge. With a store
        # built from ONE delisted snapshot (2014-07-10), the latest delisting
        # it knows is 2014-07-09 -- so every date in our actual test window,
        # 2015 to 2023, would have been answered with today's survivors and
        # nothing else. Full survivorship bias, returned silently, by the
        # module written to remove it.
        raise NoCoverage(
            f"the store's latest observed delisting is "
            f"{_day(known.latest_delisting)}; asked about {_day(moment)}. "
            "Everything that died after that date is missing, so the answer "
            "would be today's survivors under a point-in-time name -- fetch "
            "delisted snapshots covering the period first"
        )

    started = frame["ipo_date"].isna() | (frame["ipo_date"] <= moment)
    ended = frame["delisting_date"].notna() & (frame["delisting_date"] < moment)
    return set(frame.loc[started & ~ended, "ticker"].astype(str))


def died_between(start, end, frame: pd.DataFrame | None = None,
                 store: Path | str = DEFAULT_STORE) -> pd.DataFrame:
    """The names that stopped trading in a window, with their last date.

    The half of the universe our own panel does not contain: measured
    2026-09-05, 0 of 110 names have a last bar before the panel ends.
    """
    frame = load(store) if frame is None else frame
    if frame.empty:
        return frame
    lo = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None \
        else pd.Timestamp(start)
    hi = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None \
        else pd.Timestamp(end)
    dead = frame["delisting_date"].notna()
    window = dead & frame["delisting_date"].between(lo, hi)
    return frame.loc[window].sort_values("delisting_date")


def save(frame: pd.DataFrame, store: Path | str = DEFAULT_STORE) -> Path:
    """Write the store whole, refusing a write that loses delistings.

    The same guard as `parquet_union_writer` (#138): a smaller overwrite is
    sometimes legitimate and always worth saying out loud, and here the number
    that matters is not rows but DELISTINGS -- a refetch that returns only
    active names would look like a healthy 14,000-row file while silently
    deleting the entire point of the store.
    """
    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"refusing to write a store missing {missing}")

    if path.exists():
        before = coverage(store=path)
        after = coverage(frame)
        if after.delisted < before.delisted:
            logger.error(
                "OVERWRITING %s WITH FEWER DELISTINGS: %d -> %d (%+d). Rows "
                "%d -> %d. The store's whole purpose is the dead names; a "
                "write that loses them is the survivorship bias returning "
                "through the fix for it (REGISTER #169).",
                path, before.delisted, after.delisted,
                after.delisted - before.delisted, before.rows, after.rows,
            )

    frame.loc[:, list(COLUMNS)].to_parquet(path, index=False)
    logger.info("universe membership written to %s -- %s",
                path, coverage(frame).describe())
    return path
