"""The gate every price source must pass BEFORE its rows reach the database.

This lived as a private method on the Yahoo collector -- one of 22 -- so it
protected exactly one source. `BaseCollector` has no validation hook at all,
which means any new price feed (a Kaggle dump, a second API, a CSV import)
would have written straight into market_data_raw with nothing between it and
the table.

That is not hypothetical. The 15m timeframe was destroyed by precisely this
class of defect: a yfinance shared-cache race filed one instrument's bars
under another ticker, and 63,038 rows sat in the database for four months
until Stage 2's PriceFilter -- three stages downstream -- refused the whole
timeframe. 4,668 of them carried impossible prices (KO above 900, INTC above
900). The collector-side gate, once it existed, stopped it at the door:
zero contaminated rows after 2026-07-22.

The checks, in the order they run:

  missing_columns / empty_market_data   nothing to validate
  invalid_datetime_rows                 unparseable timestamps
  datetime_timezone_unresolved          naive timestamps cannot be aligned
  duplicate_identity_rows               the same (ticker, datetime, interval)
                                        twice
  cross_identity_ohlcv_rows             IDENTICAL open/high/low/close/volume
                                        under DIFFERENT ticker or interval --
                                        the shape that broke 15m
  cadence_mismatch                      bar spacing inconsistent with the
                                        declared interval

Returns a sorted list of issue strings, empty when the frame is sound. It
does not raise: the caller decides whether a source gate failure is fatal,
because a live collector and a bulk historical import can reasonably differ
on that.
"""
from __future__ import annotations

import pandas as pd

from src.pipeline.timeframe_lineage import timeframe_lineage_report


def price_source_issues(frame: pd.DataFrame) -> list[str]:
    """Validate source identity and cadence before cache/database writes."""
    required = {
        "datetime",
        "ticker",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return ["missing_columns=" + ",".join(missing)]
    if frame.empty:
        return ["empty_market_data"]

    issues: list[str] = []
    timestamps = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
    )
    if timestamps.isna().any():
        issues.append(
            f"invalid_datetime_rows={int(timestamps.isna().sum())}"
        )
    if getattr(timestamps.dt, "tz", None) is None:
        issues.append("datetime_timezone_unresolved")

    identity_duplicates = int(
        frame.duplicated(
            ["ticker", "datetime", "interval"],
            keep=False,
        ).sum()
    )
    if identity_duplicates:
        issues.append(
            f"duplicate_identity_rows={identity_duplicates}"
        )

    price_identity_columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    identity_frame = frame.assign(
        datetime=timestamps,
        _source_identity=(
            frame["ticker"].astype(str).str.upper()
            + "|"
            + frame["interval"].astype(str).str.lower()
        ),
    )
    duplicate_price_mask = identity_frame.duplicated(
        price_identity_columns,
        keep=False,
    )
    duplicate_price_rows = identity_frame.loc[duplicate_price_mask]
    if not duplicate_price_rows.empty:
        cross_identity = duplicate_price_rows.groupby(
            price_identity_columns,
            dropna=False,
        )["_source_identity"].transform("nunique") > 1
        contaminated_rows = int(cross_identity.sum())
        if contaminated_rows:
            issues.append(
                f"cross_identity_ohlcv_rows={contaminated_rows}"
            )

    cross_interval = _cross_interval_duplicate_mask(frame)
    if cross_interval.any():
        issues.append(
            f"cross_interval_ohlcv_rows={int(cross_interval.sum())}"
        )

    for interval, interval_frame in frame.groupby(
        "interval",
        dropna=False,
    ):
        report = timeframe_lineage_report(
            interval_frame,
            declared_timeframe=interval,
        )
        if report.get("status") in {
            "timeframe_cadence_mismatch",
            "timeframe_cadence_ambiguous",
        }:
            issues.append(
                "cadence_mismatch="
                f"{interval}:observed="
                f"{report.get('observed_timeframe')}"
            )

    expected_minutes = {
        "15m": 15.0,
        "1h": 60.0,
        "60m": 60.0,
        "1d": 1440.0,
    }
    timing = frame.assign(_datetime=timestamps)
    for (ticker, interval), group in timing.groupby(
        ["ticker", "interval"],
        dropna=False,
    ):
        expected = expected_minutes.get(str(interval).lower())
        if expected is None:
            issues.append(f"unsupported_interval={interval}")
            continue
        deltas = (
            group.sort_values("_datetime")["_datetime"]
            .dropna()
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60.0)
        )
        if deltas.empty:
            continue
        ratios = deltas / expected
        invalid = (
            ratios.lt(1.0 - 1e-6)
            | (ratios - ratios.round()).abs().gt(1e-6)
        )
        if invalid.any():
            issues.append(
                "cadence_mismatch="
                f"{ticker}/{interval}:"
                f"{int(invalid.sum())}/{len(deltas)}"
            )

    return sorted(set(issues))


def _cross_interval_duplicate_mask(frame: pd.DataFrame) -> "pd.Series[bool]":
    """One ticker's bar appearing under two different intervals.

    The existing cross-identity check compares rows sharing a TIMESTAMP, so
    it catches a daily bar written into the hourly series at the daily bar's
    own time. It does not catch the same bar copied to a different hour --
    and both shapes were present: AAPL's 2026-03-16 daily bar appeared at
    02:00 as `1h` (same timestamp, caught) AND at 06:00 (different timestamp,
    missed). 80 such rows sat in the hourly series, each carrying a full
    day's range and volume under an hourly label.

    Matching on open/high/low/close/volume without the timestamp is safe at
    this precision: five floats at full double precision plus an exact share
    count do not coincide between two genuinely different bars.

    The copy in the FINER series is the one flagged, and the direction
    matters: the daily bar is a genuine independent observation, while an
    hourly row carrying the whole day's range AND the whole day's share count
    is that daily bar wearing an hourly label. A real hour cannot contain a
    full day's volume unless it was the day's only hour of trading.

    (The first version of this had it backwards -- it kept the finer series
    and dropped the daily bar, which would have deleted the real observation
    and left the intruder in place.)
    """
    ohlcv = ["open", "high", "low", "close", "volume"]
    mask = pd.Series(False, index=frame.index)
    intervals = frame["interval"].astype(str).str.lower()
    if intervals.nunique() < 2:
        return mask

    # Rows per interval decides which side is the coarser series.
    order = intervals.value_counts()
    for ticker, group in frame.groupby(frame["ticker"].astype(str), sort=False):
        group_intervals = intervals.loc[group.index]
        if group_intervals.nunique() < 2:
            continue
        duplicated = group.duplicated(ohlcv, keep=False)
        if not duplicated.any():
            continue
        candidates = group.loc[duplicated]
        candidate_intervals = group_intervals.loc[candidates.index]
        spanning = (
            candidates.groupby(ohlcv, dropna=False).apply(
                lambda block: candidate_intervals.loc[block.index].nunique() > 1,
                include_groups=False,
            )
        )
        if spanning.empty or not spanning.any():
            continue
        for key, is_spanning in spanning.items():
            if not is_spanning:
                continue
            block = candidates[
                (candidates[ohlcv] == pd.Series(key, index=ohlcv)).all(axis=1)
            ]
            block_intervals = candidate_intervals.loc[block.index]
            # Keep the COARSER series' row -- it is the genuine bar -- and
            # drop its copy from the finer series, where a full day's volume
            # cannot belong.
            coarsest = min(block_intervals.unique(), key=lambda i: order.get(i, 0))
            mask.loc[block.index[block_intervals != coarsest]] = True
    return mask


#: Issues that describe the FRAME, not particular rows. Nothing can be
#: quarantined away from these, so they stay fatal.
FRAME_LEVEL_PREFIXES = (
    "missing_columns",
    "empty_market_data",
    "datetime_timezone_unresolved",
)


def quarantine_bad_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split a price frame into (clean, rejected, frame_level_issues).

    Rejecting a whole download because a fraction of it is bad is the wrong
    trade, and this project has now paid for it twice over. On 2026-08-05 the
    Yahoo collector gathered 202,713 rows and the gate refused every one of
    them over `cross_identity_ohlcv_rows=102` -- 0.05% of the batch. The
    database had already gone six days without a new row; that refusal made
    it seven, and the log said only that a source gate had failed.

    Those 102 were real, and worth keeping out: the same instrument's daily
    bar appearing in its hourly series, identical OHLCV and a full day's
    volume under a 1h label. But 202,611 sound rows should not follow them
    into the bin.

    Row-level defects are dropped by row:

      duplicate identity      the same (ticker, datetime, interval) twice
      cross-identity OHLCV    identical open/high/low/close/volume under a
                              different (ticker, interval) -- one series'
                              bars filed under another's name
      unparseable datetime    a row that cannot be placed in time

    Frame-level defects still fail everything, because there is nothing to
    quarantine them away from: absent columns, an empty frame, or timestamps
    with no timezone at all.

    Cadence is deliberately NOT enforced here. It describes a SERIES rather
    than a row, so acting on it means dropping a whole ticker/interval, and
    that decision belongs to the caller with the collection context in hand.
    `price_source_issues` still reports it.
    """
    frame_issues = [
        issue for issue in price_source_issues(frame)
        if issue.startswith(FRAME_LEVEL_PREFIXES)
    ]
    if frame_issues or frame.empty:
        return frame.iloc[0:0], frame, frame_issues

    timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
    bad = timestamps.isna()

    bad |= frame.duplicated(["ticker", "datetime", "interval"], keep="first")

    # A daily bar copied into the hourly series -- see
    # _cross_interval_duplicate_mask. The coarser bar is kept.
    bad |= _cross_interval_duplicate_mask(frame)

    price_columns = ["datetime", "open", "high", "low", "close", "volume"]
    identity = (
        frame["ticker"].astype(str).str.upper()
        + "|"
        + frame["interval"].astype(str).str.lower()
    )
    working = frame.assign(_identity=identity, _datetime=timestamps)
    duplicated_prices = working.duplicated(price_columns, keep=False)
    if duplicated_prices.any():
        distinct_identities = (
            working.loc[duplicated_prices]
            .groupby(price_columns, dropna=False)["_identity"]
            .transform("nunique")
        )
        # Keep the FIRST identity's copy rather than discarding every side:
        # one of them is the genuine bar, and dropping both loses real data
        # to punish the duplicate.
        cross = distinct_identities > 1
        offenders = working.loc[duplicated_prices].loc[cross]
        losers = offenders.duplicated(price_columns, keep="first")
        bad.loc[offenders.index[losers]] = True

    return frame.loc[~bad].copy(), frame.loc[bad].copy(), []
