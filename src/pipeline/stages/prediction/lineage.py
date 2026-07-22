from __future__ import annotations

from typing import Any

import pandas as pd

from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    timeframe_lineage_report,
)

_PLACEHOLDER_FINGERPRINTS = {
    "default",
    "normal",
    "unknown",
    "unknown_context",
    "batch_training",
}


def source_lineage_attrs(frame: pd.DataFrame) -> dict[str, Any]:
    """Extract prediction-time lineage before metadata columns are removed."""
    timeframe_report = timeframe_lineage_report(frame)
    return {
        "prediction_observed_at": _last_observed_at(frame),
        "prediction_timeframe": timeframe_report.get(
            "resolved_timeframe"
        ),
        "prediction_timeframe_lineage": timeframe_report,
    }


def apply_lineage_attrs(
    frame: pd.DataFrame,
    attrs: dict[str, Any],
) -> pd.DataFrame:
    frame.attrs.update(
        {
            key: value
            for key, value in attrs.items()
            if value is not None
        }
    )
    return frame


def prediction_observed_at(frame: pd.DataFrame) -> str | None:
    value = frame.attrs.get("prediction_observed_at")
    if value is not None:
        return str(value)
    return _last_observed_at(frame)


def prediction_timeframe(frame: pd.DataFrame) -> str | None:
    value = frame.attrs.get("prediction_timeframe")
    if value is not None:
        return _normalize_timeframe(value)
    return timeframe_lineage_report(frame).get("resolved_timeframe")


def prediction_timeframe_lineage(
    frame: pd.DataFrame,
    declared_timeframe: Any = None,
) -> dict[str, Any]:
    value = frame.attrs.get("prediction_timeframe_lineage")
    report = (
        dict(value)
        if isinstance(value, dict)
        else timeframe_lineage_report(frame)
    )
    model_declared = normalize_timeframe(declared_timeframe)
    if not model_declared:
        return report
    frame_declared = normalize_timeframe(
        report.get("declared_timeframe")
    )
    observed = normalize_timeframe(
        report.get("observed_timeframe")
    )
    conflict = (
        (frame_declared and frame_declared != model_declared)
        or (observed and observed != model_declared)
        or report.get("status")
        in {
            "timeframe_cadence_mismatch",
            "timeframe_cadence_ambiguous",
        }
    )
    report["model_declared_timeframe"] = model_declared
    if conflict:
        report["status"] = "timeframe_cadence_mismatch"
        report["resolved_timeframe"] = None
        report["safe_for_prediction_lineage"] = False
    else:
        report["resolved_timeframe"] = model_declared
        report["safe_for_prediction_lineage"] = True
        if observed:
            report["status"] = "timeframe_cadence_verified"
        else:
            report["status"] = (
                "timeframe_declared_cadence_unverified"
            )
    return report


def trusted_context_fingerprint(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = text.lower()
        if (
            normalized in _PLACEHOLDER_FINGERPRINTS
            or normalized.startswith("legacy_")
        ):
            continue
        return text
    return None


def _last_observed_at(frame: pd.DataFrame) -> str | None:
    for column in ("datetime", "timestamp", "date"):
        if column not in frame.columns:
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        return _timestamp_text(values.iloc[-1])
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
        return _timestamp_text(frame.index[-1])
    return None


def _single_timeframe(frame: pd.DataFrame) -> str | None:
    for column in ("timeframe", "interval"):
        if column not in frame.columns:
            continue
        values = {
            _normalize_timeframe(value)
            for value in frame[column].dropna().astype(str)
            if str(value).strip()
        }
        if len(values) == 1:
            return values.pop()
    return None


def _timestamp_text(value: Any) -> str | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _normalize_timeframe(value: Any) -> str:
    normalized = str(value).strip().lower()
    return {
        "15min": "15m",
        "1h": "60m",
        "60min": "60m",
        "daily": "1d",
    }.get(normalized, normalized)


__all__ = [
    "apply_lineage_attrs",
    "prediction_observed_at",
    "prediction_timeframe",
    "prediction_timeframe_lineage",
    "source_lineage_attrs",
    "trusted_context_fingerprint",
]
