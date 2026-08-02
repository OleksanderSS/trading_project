from __future__ import annotations

from collections import Counter
from typing import Any

import logging
import pandas as pd

logger = logging.getLogger(__name__)

_TIMEFRAME_ALIASES = {
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "60m",
    "60min": "60m",
    "daily": "1d",
}
_TIMEFRAME_DURATIONS = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}
_MIN_CADENCE_OBSERVATIONS = 4
_MIN_MATCH_SHARE = 0.30
_MIN_SCORE_MARGIN = 0.10


def normalize_timeframe(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return _TIMEFRAME_ALIASES.get(text, text)


def is_timeframe_token(value: Any) -> bool:
    """True when ``value`` spells one of the timeframes this project uses.

    normalize_timeframe answers "what is this called canonically" and will
    happily hand back any string it does not recognise. Callers that need to
    decide whether a trailing name fragment IS a timeframe -- rather than
    part of a longer feature name -- need this instead. Without it,
    ``MARKET_REGIME_ENCODED_1d`` reads as ``MARKET_REGIME`` suffixed with
    ``ENCODED_1d``, and a confidence float gets mistaken for a regime label.
    """
    normalized = normalize_timeframe(value)
    return bool(normalized) and normalized in _TIMEFRAME_DURATIONS


def timeframe_lineage_report(
    frame: pd.DataFrame,
    *,
    declared_timeframe: Any = None,
) -> dict[str, Any]:
    declared = (
        normalize_timeframe(declared_timeframe)
        or _single_declared_timeframe(frame)
    )
    cadence = infer_observed_timeframe(frame)
    observed = cadence.get("inferred_timeframe")
    cadence_status = cadence.get("status")

    if cadence_status == "ambiguous":
        status = "timeframe_cadence_ambiguous"
        resolved = None
    elif declared and observed and declared != observed:
        status = "timeframe_cadence_mismatch"
        resolved = None
    elif declared and observed:
        status = "timeframe_cadence_verified"
        resolved = declared
    elif declared:
        status = "timeframe_declared_cadence_unverified"
        resolved = declared
    elif observed:
        status = "timeframe_inferred_from_cadence"
        resolved = observed
    else:
        status = "timeframe_missing"
        resolved = None

    return {
        "status": status,
        "declared_timeframe": declared,
        "observed_timeframe": observed,
        "resolved_timeframe": resolved,
        "cadence": cadence,
        "safe_for_prediction_lineage": resolved is not None,
    }


def infer_observed_timeframe(frame: pd.DataFrame) -> dict[str, Any]:
    datetime_column = _datetime_column(frame)
    if datetime_column is None:
        return _empty_cadence("datetime_missing")

    group_column = next(
        (
            column
            for column in ("ticker", "symbol")
            if column in frame.columns
        ),
        None,
    )
    if group_column:
        grouped = frame.groupby(group_column, dropna=False, sort=True)
    else:
        grouped = [("__all__", frame)]

    group_reports = []
    for group_name, group in grouped:
        timestamps = pd.to_datetime(
            group[datetime_column],
            errors="coerce",
            utc=True,
        ).dropna()
        timestamps = pd.Series(timestamps.unique()).sort_values()
        deltas = timestamps.diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        scores = {
            timeframe: _cadence_match_share(deltas, duration)
            for timeframe, duration in _TIMEFRAME_DURATIONS.items()
        }
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        best_timeframe, best_score = ranked[0]
        second_score = ranked[1][1]
        inferred = None
        if (
            len(deltas) >= _MIN_CADENCE_OBSERVATIONS
            and best_score >= _MIN_MATCH_SHARE
            and best_score - second_score >= _MIN_SCORE_MARGIN
        ):
            inferred = best_timeframe
        group_reports.append(
            {
                "group": str(group_name),
                "observation_count": int(len(timestamps)),
                "delta_count": int(len(deltas)),
                "inferred_timeframe": inferred,
                "best_match_share": round(float(best_score), 6),
                "score_margin": round(
                    float(best_score - second_score), 6
                ),
                "match_shares": {
                    key: round(float(value), 6)
                    for key, value in scores.items()
                },
            }
        )

    inferred_counts = Counter(
        item["inferred_timeframe"]
        for item in group_reports
        if item["inferred_timeframe"]
    )
    unresolved_count = sum(
        item["inferred_timeframe"] is None
        for item in group_reports
    )
    if len(inferred_counts) > 1:
        status = "ambiguous"
        inferred_timeframe = None
    elif len(inferred_counts) == 1 and unresolved_count == 0:
        status = "inferred"
        inferred_timeframe = next(iter(inferred_counts))
    elif len(inferred_counts) == 1:
        status = "partially_inferred"
        inferred_timeframe = next(iter(inferred_counts))
    else:
        status = "insufficient_observations"
        inferred_timeframe = None
    return {
        "status": status,
        "datetime_column": datetime_column,
        "inferred_timeframe": inferred_timeframe,
        "group_count": len(group_reports),
        "unresolved_group_count": unresolved_count,
        "inferred_group_counts": dict(sorted(inferred_counts.items())),
        "group_reports": group_reports,
        "minimum_observations": _MIN_CADENCE_OBSERVATIONS,
        "minimum_match_share": _MIN_MATCH_SHARE,
        "minimum_score_margin": _MIN_SCORE_MARGIN,
    }


def partition_market_frame_by_timeframe(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    declared_column = next(
        (
            column
            for column in ("interval", "timeframe")
            if column in frame.columns
        ),
        None,
    )
    if declared_column:
        normalized = frame[declared_column].map(normalize_timeframe)
        if normalized.isna().any():
            raise ValueError(
                f"{declared_column} contains missing timeframe values"
            )
        unsupported = sorted(
            set(normalized) - set(_TIMEFRAME_DURATIONS)
        )
        if unsupported:
            raise ValueError(
                "Unsupported market-data timeframe(s): "
                + ", ".join(unsupported)
            )
        groups = {}
        for timeframe in sorted(set(normalized)):
            selected = frame.loc[normalized.eq(timeframe)].copy()
            report = timeframe_lineage_report(
                selected,
                declared_timeframe=timeframe,
            )
            if report["status"] == "timeframe_cadence_mismatch":
                raise ValueError(
                    f"Declared {timeframe} market data has observed "
                    f"{report['observed_timeframe']} cadence. "
                    "This usually indicates multi-timeframe upsampling (forward-fill)."
                )
            selected["interval"] = timeframe
            selected.attrs["timeframe_lineage"] = report
            selected.attrs["timeframe_source"] = (
                f"declared_{declared_column}"
            )
            groups[timeframe] = selected
        return groups

    report = timeframe_lineage_report(frame)
    inferred = report.get("resolved_timeframe")
    if not inferred:
        raise ValueError(
            "Market data has no declared timeframe and cadence cannot "
            "be inferred unambiguously"
        )
    selected = frame.copy()
    selected["interval"] = inferred
    selected.attrs["timeframe_lineage"] = report
    selected.attrs["timeframe_source"] = (
        "inferred_from_observed_cadence"
    )
    return {inferred: selected}


def _single_declared_timeframe(
    frame: pd.DataFrame,
) -> str | None:
    for column in ("timeframe", "interval"):
        if column not in frame.columns:
            continue
        values = {
            normalize_timeframe(value)
            for value in frame[column].dropna().unique()
            if normalize_timeframe(value)
        }
        if len(values) == 1:
            return values.pop()
    return None


def _datetime_column(frame: pd.DataFrame) -> str | None:
    return next(
        (
            column
            for column in ("datetime", "timestamp", "date")
            if column in frame.columns
        ),
        None,
    )


def _cadence_match_share(
    deltas: pd.Series,
    duration: pd.Timedelta,
) -> float:
    if deltas.empty:
        return 0.0
    tolerance = max(
        pd.Timedelta(seconds=1),
        duration * 0.02,
    )
    matches = (deltas - duration).abs() <= tolerance
    return float(matches.mean())


def _empty_cadence(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "datetime_column": None,
        "inferred_timeframe": None,
        "group_count": 0,
        "unresolved_group_count": 0,
        "inferred_group_counts": {},
        "group_reports": [],
        "minimum_observations": _MIN_CADENCE_OBSERVATIONS,
        "minimum_match_share": _MIN_MATCH_SHARE,
        "minimum_score_margin": _MIN_SCORE_MARGIN,
    }


__all__ = [
    "infer_observed_timeframe",
    "normalize_timeframe",
    "partition_market_frame_by_timeframe",
    "timeframe_lineage_report",
]
