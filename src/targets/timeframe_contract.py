from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

_TIMEFRAME_DURATION = {
    "15m": pd.Timedelta(minutes=15),
    "60m": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}
_HORIZON_DURATION = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}
_PARTITION_COLUMNS = (
    "partition_id",
    "source_partition",
    "data_partition",
    "segment_id",
)


@dataclass(frozen=True)
class TargetTimeframeContract:
    timeframe: str | None
    horizon: str | None
    shift_bars: int
    expected_elapsed: pd.Timedelta | None
    maximum_elapsed: pd.Timedelta | None


def target_applies_to_timeframe(
    target: dict[str, Any],
    timeframe: str,
) -> bool:
    """Return whether a configured target has valid semantics on a timeframe."""
    normalized_timeframe = _normalize_timeframe(timeframe)
    name = str(target["name"]).lower()
    params = target.get("params", {})
    horizon = str(params.get("horizon", "")).lower()
    source_timeframe = _target_source_timeframe(target)

    if (
        source_timeframe is not None
        and source_timeframe != normalized_timeframe
    ):
        return False

    if horizon == "15m":
        return normalized_timeframe == "15m"
    if horizon == "1h":
        return normalized_timeframe in {"15m", "60m"}
    if horizon == "1d":
        return normalized_timeframe == "1d"
    if "intraday" in name or "_15m" in name:
        return normalized_timeframe == "15m"
    if "hourly" in name or "_1h" in name:
        return normalized_timeframe in {"15m", "60m"}
    if (
        "daily" in name
        or "weekly" in name
        or name.endswith(("_1d", "_5d", "_1w"))
    ):
        return normalized_timeframe == "1d"
    return True


def _target_source_timeframe(
    target: dict[str, Any],
) -> str | None:
    params = target.get("params", {})
    explicit = params.get("source_timeframe")
    if explicit:
        return _normalize_timeframe(explicit)
    if str(target.get("type") or "") != "indicator_prediction":
        return None
    indicator = str(params.get("indicator_col") or "").lower()
    for suffix, timeframe in (
        ("_15m", "15m"),
        ("_60m", "60m"),
        ("_1h", "60m"),
        ("_1d", "1d"),
    ):
        if indicator.endswith(suffix):
            return timeframe
    return None


#: Suffixes the feature stage appends to indicator columns. It uses whatever
#: the frame's `interval` column says, so the hourly frame yields `_1h` while
#: the target config and the contract both speak `60m`. Both spellings live
#: here so a column can be found under either.
_COLUMN_TIMEFRAME_SUFFIXES = ("_15m", "_60m", "_1h", "_1d")

_SUFFIX_FOR_TIMEFRAME = {
    "15m": ("_15m",),
    "60m": ("_1h", "_60m"),
    "1d": ("_1d",),
}


def resolve_column_for_frame(
    name: str,
    frame: pd.DataFrame,
    timeframe: str | None,
) -> str | None:
    """Find the column a target means, on the frame it is actually running on.

    A target names one indicator, `BB_Upper_60m`, and then runs wherever its
    horizon is valid -- a "1h" horizon is valid on 15-minute bars (four of
    them) and on hourly bars (one). The column is not the same on both: the
    feature stage suffixes by the frame's own `interval`, so it is
    `BB_Upper_15m` on one and `BB_Upper_1h` on the other, and
    `BB_Upper_60m` exists on neither.

    `target_hourly_breakout_1h` and `target_volatility_spike_1h` failed this
    way on the 15-minute frame on every run, two ERROR lines each time.

    Corrected after checking the 18.08 batch rather than trusting the log:
    they are NOT absent from it. They carry 64,203 and 64,206 values -- all at
    interval `60m`, where the hardcoded `_60m` suffix happens to match. So the
    loss was the 15-minute slice of two targets, not the targets themselves,
    and the config comment claiming the hourly frame is labelled 60m was right
    about that batch. What this removes is the dependence on that label:
    whether the frame calls itself `60m` or `1h`, the column is found.

    The horizon already resolves against the frame; this is the same idea for
    the column. An exact match always wins, so a target that deliberately
    names one timeframe's indicator keeps it.
    """
    if not name:
        return None
    if name in frame.columns:
        return name

    stem = name
    for suffix in _COLUMN_TIMEFRAME_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidates: list[str] = []
    normalized = _normalize_timeframe(timeframe) if timeframe else None
    for suffix in _SUFFIX_FOR_TIMEFRAME.get(normalized or "", ()):
        candidates.append(f"{stem}{suffix}")
    candidates.append(stem)

    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def resolve_target_timeframe_contract(
    params: dict[str, Any],
    frame: pd.DataFrame,
    *,
    default_timeframe: str | None = None,
) -> tuple[dict[str, Any], TargetTimeframeContract]:
    """Resolve a semantic target horizon against the frame's actual cadence."""
    resolved = dict(params)
    horizon = resolved.pop("horizon", None)
    timeframe = _frame_timeframe(frame, default_timeframe)
    configured_shift = int(resolved.get("shift", -1))
    if configured_shift >= 0:
        raise ValueError(f"Target shift must be negative, got {configured_shift}.")

    duration = _TIMEFRAME_DURATION.get(timeframe or "")
    if horizon is not None:
        horizon_duration = _HORIZON_DURATION.get(str(horizon).lower())
        if horizon_duration is None:
            raise ValueError(f"Unsupported semantic target horizon: {horizon}.")
        if duration is None:
            raise ValueError(
                f"Cannot resolve target horizon {horizon} without a supported timeframe."
            )
        ratio = horizon_duration / duration
        if ratio < 1 or float(ratio) != int(ratio):
            raise ValueError(
                f"Target horizon {horizon} is not an exact bar multiple of {timeframe}."
            )
        shift_bars = int(ratio)
        resolved["shift"] = -shift_bars
        expected_elapsed = horizon_duration
    else:
        shift_bars = abs(configured_shift)
        expected_elapsed = duration * shift_bars if duration is not None else None

    maximum_elapsed = _maximum_elapsed(timeframe, expected_elapsed, shift_bars)
    return resolved, TargetTimeframeContract(
        timeframe=timeframe,
        horizon=str(horizon).lower() if horizon is not None else None,
        shift_bars=shift_bars,
        expected_elapsed=expected_elapsed,
        maximum_elapsed=maximum_elapsed,
    )


def mask_targets_across_time_boundaries(
    frame: pd.DataFrame,
    target: pd.Series,
    contract: TargetTimeframeContract,
) -> pd.Series:
    """Blank labels whose future endpoint crosses a gap or partition boundary."""
    valid = pd.Series(True, index=frame.index)
    bars = contract.shift_bars

    temporal_column = next(
        (name for name in ("datetime", "timestamp", "date") if name in frame.columns),
        None,
    )
    if temporal_column is not None:
        observed_at = pd.to_datetime(frame[temporal_column], errors="coerce", utc=True)
        future_at = observed_at.shift(-bars)  # audit-ignore: NEGATIVE_SHIFT_INTENTIONAL target horizon, used to BLANK labels
        elapsed = future_at - observed_at
        valid &= observed_at.notna() & future_at.notna() & elapsed.gt(pd.Timedelta(0))
        if contract.maximum_elapsed is not None:
            valid &= elapsed.le(contract.maximum_elapsed)

    for column in _PARTITION_COLUMNS:
        if column in frame.columns:
            future_partition = frame[column].shift(-bars)  # audit-ignore: NEGATIVE_SHIFT_INTENTIONAL target horizon, used to BLANK labels
            valid &= frame[column].notna() & frame[column].eq(future_partition)

    return target.where(valid)


def _frame_timeframe(
    frame: pd.DataFrame,
    default_timeframe: str | None,
) -> str | None:
    values: list[str] = []
    if "interval" in frame.columns:
        values = [
            _normalize_timeframe(value)
            for value in frame["interval"].dropna().astype(str).unique()
        ]
        values = sorted(set(values))
    if len(values) > 1:
        raise ValueError(f"Target frame mixes timeframes: {values}.")
    if values:
        return values[0]
    return _normalize_timeframe(default_timeframe) if default_timeframe else None


def _normalize_timeframe(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "15min": "15m",
        "1h": "60m",
        "60min": "60m",
        "daily": "1d",
    }
    return aliases.get(normalized, normalized)


def _maximum_elapsed(
    timeframe: str | None,
    expected_elapsed: pd.Timedelta | None,
    shift_bars: int | None = None,
) -> pd.Timedelta | None:
    """How much wall-clock time a horizon may span before it looks like a gap.

    The daily allowance was a flat three days, which covers one weekend. A
    horizon of n trading days crosses one weekend when it starts on a Monday
    and two when it starts on a Thursday, so a constant slack silently
    removes the later weekdays as n grows.

    Measured on the 2026-08-13 batch, AAPL daily, target_weekly_up_1w
    (shift 7): expected 7 days, allowance 10 days.

        from Mon/Tue/Wed   7 trading days =  9 calendar days   kept
        from Thu/Fri       7 trading days = 11 calendar days   blanked

        Mon  93 of 100    Tue  98 of 107    Wed  69 of 104
        Thu   0 of 101    Fri   0 of 102

    260 labels survived of the 507 the data supports. Recomputing the target
    directly agrees with every stored value, so nothing was wrong with the
    labels -- they were simply thrown away, and thrown away by weekday, which
    left the model unable to learn from a Thursday or Friday at all. That is
    worse than the missing half: it is a systematic bias.

    `target_up_5d` (shift 5) escaped only by luck of arithmetic -- from a
    Thursday it spans 7 calendar days against an allowance of 8.

    The slack now scales with the number of weekends the span can contain.
    A run of n trading days starting on a Friday touches ceil((n + 4) / 5)
    weekends; three days each keeps the holiday tolerance the original had.
    """
    if expected_elapsed is None:
        return None
    if timeframe == "1d":
        bars = max(int(shift_bars or 1), 1)
        weekends = -(-(bars + 4) // 5)          # ceil((bars + 4) / 5)
        return expected_elapsed + pd.Timedelta(days=3 * weekends)
    return expected_elapsed * 1.5
