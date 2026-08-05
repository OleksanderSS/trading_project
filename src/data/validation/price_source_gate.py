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
