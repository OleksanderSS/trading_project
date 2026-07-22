"""Utilities for keeping target labels out of model features."""

from __future__ import annotations

from collections.abc import Iterable

_DIRECT_TARGET_PREFIXES = ("target_", "target.", "target:", "target-", "target ")
_DERIVED_TARGET_MARKERS = ("_target_", "_target.", "_target:", "_target-", "_target ")


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower()


def is_direct_target_column(column: object) -> bool:
    """Return True for direct label columns such as target_up_1d or TARGET_RETURN_1P."""
    text = _normalize_column_name(column)
    return text.startswith(_DIRECT_TARGET_PREFIXES)


def is_target_like_column(column: object) -> bool:
    """Return True for direct labels or features derived from labels."""
    text = _normalize_column_name(column)
    return is_direct_target_column(text) or any(marker in text for marker in _DERIVED_TARGET_MARKERS)


def split_model_features_and_targets(columns: Iterable[object]) -> tuple[list[object], list[object], list[object]]:
    """
    Split columns into safe model features, direct targets, and dropped target-derived columns.

    Direct targets belong in targets. Target-derived columns such as state_TARGET_RETURN_1P
    are excluded from features but are not promoted to targets.
    """
    feature_columns: list[object] = []
    target_columns: list[object] = []
    dropped_target_derived_columns: list[object] = []
    for column in columns:
        if is_direct_target_column(column):
            target_columns.append(column)
        elif is_target_like_column(column):
            dropped_target_derived_columns.append(column)
        else:
            feature_columns.append(column)
    return feature_columns, target_columns, dropped_target_derived_columns
