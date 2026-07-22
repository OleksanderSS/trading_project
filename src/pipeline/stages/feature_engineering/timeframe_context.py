from __future__ import annotations

from typing import Any

import pandas as pd

from src.pipeline.target_column_utils import is_target_like_column

_TIMEFRAME_ALIASES = {
    "15min": "15m",
    "1h": "60m",
    "60min": "60m",
    "daily": "1d",
}
_TIMEFRAME_DURATION = {
    "15m": pd.Timedelta(minutes=15),
    "60m": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}
_CONTEXT_TOLERANCE = {
    "60m": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=4),
}
_PARTITION_COLUMNS = (
    "partition_id",
    "source_partition",
    "data_partition",
    "segment_id",
)
_SERVICE_COLUMNS = {
    "datetime",
    "timestamp",
    "date",
    "ticker",
    "symbol",
    "interval",
    "timeframe",
    "hash",
}


class BackwardTimeframeContextAssembler:
    """Attach completed higher-timeframe observations to each base row."""

    def assemble(
        self,
        frames_by_timeframe: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized = self._normalize_frames(frames_by_timeframe)
        if not normalized:
            return pd.DataFrame(), self._empty_report()

        assembled_frames: list[pd.DataFrame] = []
        base_reports: list[dict[str, Any]] = []
        total_future_violations = 0

        for base_timeframe, base_frame in normalized.items():
            assembled = self._prepare_base(base_frame, base_timeframe)
            joins: list[dict[str, Any]] = []
            for context_timeframe in self._higher_timeframes(
                base_timeframe,
                normalized,
            ):
                assembled, join_report = self._join_context(
                    assembled,
                    normalized[context_timeframe],
                    base_timeframe=base_timeframe,
                    context_timeframe=context_timeframe,
                )
                joins.append(join_report)
                total_future_violations += join_report["future_context_violations"]

            row_identity_preserved = len(assembled) == len(base_frame)
            if not row_identity_preserved:
                raise ValueError(
                    f"Context assembly changed {base_timeframe} row count "
                    f"from {len(base_frame)} to {len(assembled)}."
                )
            assembled_frames.append(self._finalize_base(assembled))
            base_reports.append(
                {
                    "base_timeframe": base_timeframe,
                    "input_rows": int(len(base_frame)),
                    "output_rows": int(len(assembled)),
                    "row_identity_preserved": row_identity_preserved,
                    "context_joins": joins,
                }
            )

        combined = pd.concat(assembled_frames, ignore_index=True, sort=False)
        if total_future_violations:
            raise ValueError(
                "Backward timeframe context assembly produced future-context matches."
            )

        report = {
            "status": "causal_timeframe_context_ready",
            "join_direction": "backward",
            "allow_future_context": False,
            "bar_availability_policy": {
                "15m": "bar_start_plus_15_minutes",
                "60m": "bar_start_plus_60_minutes",
                "1d": "daily_label_plus_1_calendar_day",
            },
            "partition_isolation_columns": list(_PARTITION_COLUMNS),
            "input_timeframes": list(normalized),
            "base_contexts": base_reports,
            "summary": {
                "input_rows": int(sum(len(frame) for frame in normalized.values())),
                "output_rows": int(len(combined)),
                "base_context_count": len(base_reports),
                "future_context_violations": total_future_violations,
                "row_identity_preserved": all(
                    item["row_identity_preserved"] for item in base_reports
                ),
            },
        }
        return combined, report

    def _normalize_frames(
        self,
        frames_by_timeframe: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        normalized: dict[str, pd.DataFrame] = {}
        for raw_timeframe, frame in frames_by_timeframe.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            timeframe = _normalize_timeframe(raw_timeframe)
            if timeframe not in _TIMEFRAME_DURATION:
                raise ValueError(f"Unsupported timeframe for context assembly: {raw_timeframe}.")
            if timeframe in normalized:
                raise ValueError(f"Duplicate normalized timeframe: {timeframe}.")
            normalized[timeframe] = frame.copy()
        return normalized

    def _prepare_base(
        self,
        frame: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        result = frame.copy()
        datetime_column = _datetime_column(result)
        if "ticker" not in result.columns:
            raise ValueError(f"{timeframe} frame is missing ticker.")
        if "interval" in result.columns:
            intervals = {
                _normalize_timeframe(value)
                for value in result["interval"].dropna().astype(str).unique()
            }
            if intervals and intervals != {timeframe}:
                raise ValueError(
                    f"{timeframe} frame contains incompatible intervals: {sorted(intervals)}."
                )
        result["interval"] = timeframe
        result["__base_order"] = range(len(result))
        result["__base_time_utc"] = _as_utc_nanoseconds(
            result[datetime_column],
        )
        if result["__base_time_utc"].isna().any():
            raise ValueError(f"{timeframe} frame contains invalid datetimes.")
        result["__base_available_utc"] = (
            result["__base_time_utc"] + _TIMEFRAME_DURATION[timeframe]
        )
        return result

    def _higher_timeframes(
        self,
        base_timeframe: str,
        frames: dict[str, pd.DataFrame],
    ) -> list[str]:
        base_duration = _TIMEFRAME_DURATION[base_timeframe]
        return sorted(
            (
                timeframe
                for timeframe in frames
                if _TIMEFRAME_DURATION[timeframe] > base_duration
            ),
            key=_TIMEFRAME_DURATION.__getitem__,
        )

    def _join_context(
        self,
        base: pd.DataFrame,
        context: pd.DataFrame,
        *,
        base_timeframe: str,
        context_timeframe: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        prepared_context, payload_columns = self._prepare_context(
            context,
            context_timeframe,
        )
        mismatched_partition_columns = [
            column
            for column in _PARTITION_COLUMNS
            if (column in base.columns) != (column in prepared_context.columns)
        ]
        if mismatched_partition_columns:
            raise ValueError(
                "Base and context frames must declare the same partition metadata: "
                f"{mismatched_partition_columns}."
            )
        group_columns = ["ticker", *[
            column
            for column in _PARTITION_COLUMNS
            if column in base.columns and column in prepared_context.columns
        ]]
        context_output_columns = [
            f"ctx_{context_timeframe}_source_datetime",
            f"ctx_{context_timeframe}_available_at",
            *payload_columns,
        ]
        collisions = sorted(set(context_output_columns).intersection(base.columns))
        if collisions:
            raise ValueError(
                f"Context columns already exist before {context_timeframe} join: {collisions}."
            )

        merged_groups: list[pd.DataFrame] = []
        for identity, base_group in base.groupby(
            group_columns,
            sort=False,
            dropna=False,
        ):
            identity_values = identity if isinstance(identity, tuple) else (identity,)
            context_group = prepared_context
            for column, value in zip(group_columns, identity_values, strict=True):
                if pd.isna(value):
                    context_group = context_group.loc[context_group[column].isna()]
                else:
                    context_group = context_group.loc[context_group[column].eq(value)]

            if context_group.empty:
                unmatched = base_group.copy()
                for column in context_output_columns:
                    unmatched[column] = pd.Series(
                        index=unmatched.index,
                        dtype=prepared_context[column].dtype,
                    )
                merged_groups.append(unmatched)
                continue

            right_columns = [
                "__context_available_utc",
                *context_output_columns,
            ]
            right = (
                context_group[right_columns]
                .sort_values("__context_available_utc", kind="mergesort")
                .drop_duplicates("__context_available_utc", keep="last")
            )
            left = base_group.sort_values("__base_time_utc", kind="mergesort")
            merged = pd.merge_asof(
                left,
                right,
                left_on="__base_available_utc",
                right_on="__context_available_utc",
                direction="backward",
                allow_exact_matches=True,
                tolerance=_CONTEXT_TOLERANCE.get(context_timeframe),
            )
            merged_groups.append(merged)

        result = (
            pd.concat(merged_groups, ignore_index=True, sort=False)
            .sort_values("__base_order", kind="mergesort")
            .reset_index(drop=True)
        )
        available_column = f"ctx_{context_timeframe}_available_at"
        source_column = f"ctx_{context_timeframe}_source_datetime"
        matched = result[source_column].notna()
        future_violations = (
            result.loc[matched, available_column]
            > result.loc[matched, "__base_available_utc"]
        ).sum()
        result = result.drop(columns=["__context_available_utc"], errors="ignore")
        return result, {
            "base_timeframe": base_timeframe,
            "context_timeframe": context_timeframe,
            "direction": "backward",
            "input_context_rows": int(len(context)),
            "matched_base_rows": int(matched.sum()),
            "unmatched_base_rows": int((~matched).sum()),
            "context_feature_count": len(payload_columns),
            "future_context_violations": int(future_violations),
            "tolerance": str(_CONTEXT_TOLERANCE.get(context_timeframe)),
        }

    def _prepare_context(
        self,
        frame: pd.DataFrame,
        timeframe: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        result = frame.copy()
        datetime_column = _datetime_column(result)
        if "ticker" not in result.columns:
            raise ValueError(f"{timeframe} context frame is missing ticker.")

        source_time = _as_utc_nanoseconds(result[datetime_column])
        if source_time.isna().any():
            raise ValueError(f"{timeframe} context frame contains invalid datetimes.")
        result["__context_available_utc"] = (
            source_time + _TIMEFRAME_DURATION[timeframe]
        )
        result[f"ctx_{timeframe}_source_datetime"] = source_time
        result[f"ctx_{timeframe}_available_at"] = result[
            "__context_available_utc"
        ]

        excluded = _SERVICE_COLUMNS.union(_PARTITION_COLUMNS)
        raw_payload_columns = [
            column
            for column in result.columns
            if column not in excluded
            and not column.startswith("__")
            and not column.startswith(f"ctx_{timeframe}_")
            and not is_target_like_column(column)
        ]
        rename_map = {
            column: f"ctx_{timeframe}_{column}"
            for column in raw_payload_columns
        }
        result = result.rename(columns=rename_map)
        return result, list(rename_map.values())

    def _finalize_base(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(
            columns=[
                "__base_order",
                "__base_time_utc",
                "__base_available_utc",
            ],
            errors="ignore",
        )

    def _empty_report(self) -> dict[str, Any]:
        return {
            "status": "no_timeframe_context",
            "join_direction": "backward",
            "allow_future_context": False,
            "input_timeframes": [],
            "base_contexts": [],
            "summary": {
                "input_rows": 0,
                "output_rows": 0,
                "base_context_count": 0,
                "future_context_violations": 0,
                "row_identity_preserved": True,
            },
        }


def _normalize_timeframe(value: object) -> str:
    normalized = str(value).strip().lower()
    return _TIMEFRAME_ALIASES.get(normalized, normalized)


def _datetime_column(frame: pd.DataFrame) -> str:
    for column in ("datetime", "timestamp", "date"):
        if column in frame.columns:
            return column
    raise ValueError("Timeframe frame requires datetime, timestamp, or date.")


def _as_utc_nanoseconds(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce", utc=True)
    return converted.astype("datetime64[ns, UTC]")
