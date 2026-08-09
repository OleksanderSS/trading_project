"""Utilities for keeping target labels out of model features."""

from __future__ import annotations

from collections.abc import Iterable

from src.pipeline.timeframe_lineage import is_timeframe_token

_DIRECT_TARGET_PREFIXES = ("target_", "target.", "target:", "target-", "target ")
_DERIVED_TARGET_MARKERS = ("_target_", "_target.", "_target:", "_target-", "_target ")

#: Columns that identify a row or describe its context, rather than
#: describing the market. They must never become model features -- see
#: is_identity_column for what happens when they do.
_IDENTITY_STEMS = (
    "hash",
    "record_hash",
    "context_fingerprint",
    "context_pattern_seq",
    "context_pattern_id",
    "context_schema_id",
    "symbol",
    "ticker",
    "interval",
    "timeframe",
)


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower()


def is_identity_column(column: object) -> bool:
    """True for row identifiers and context keys, with or without a suffix.

    These are strings, so `handle_categorical_features` label-encodes any
    with more than five distinct values before feature selection sees them --
    turning them into perfectly ordinary-looking integers. Three consequences,
    all observed on the 2026-08-04 run:

    - `hash` is unique per row. Label-encoded it becomes a dense row index,
      and a tree model can split on it to memorise the training set. An
      identifier is the purest form of leakage there is.
    - `context_fingerprint` and `context_pattern_seq` carry tri-state
      structure ('0|1|-1|...'). Label encoding maps them to an arbitrary
      integer ordered alphabetically, which is meaningless as a magnitude and
      destroys the structure the KNN context search depends on.
    - The LabelEncoder is local to that function and never persisted, so no
      unseen string can be mapped back at prediction time. Models trained on
      these columns cannot be scored on new data even in principle.

    Matched by stem so suffixed forms are covered: the enrichers emit
    context_fingerprint_1d and context_pattern_seq_60m, and a check against
    the bare names alone would miss every one of them -- the same
    one-thing-two-names trap this project has hit repeatedly.

    A leading ctx_<timeframe>_ is stripped first, because the cross-timeframe
    assembler copies a lower timeframe's columns forward under that prefix --
    identity columns included. ctx_1d_context_fingerprint_1d is the same
    string as context_fingerprint_1d, one prefix later, and the stem check
    anchored at the start of the name did not see it.

    That gap cost the 2026-08-09 run 313 of 660 contexts. The columns are not
    empty in features.parquet -- they hold strings, so an all-NaN scan finds
    nothing wrong -- but Stage 5 coerces with pd.to_numeric before checking,
    which turns every one of them into NaN, and a row with a NaN in a
    required feature is dropped. Three such columns emptied the entire
    50-row prediction window, per context, silently:

        3 of 777 required feature(s) are null in EVERY one of the 50
        candidate rows (e.g. ctx_1d_context_fingerprint_1d,
        ctx_1d_context_pattern_seq_1d, ctx_1d_context_pattern_id_1d)
    """
    text = _normalize_column_name(column)
    if text.startswith("ctx_"):
        parts = text.split("_", 2)
        # Only a genuine timeframe segment is a prefix. A column merely
        # beginning with "ctx_" keeps its name -- stripping two fields from
        # anything would turn ctx_volume_ratio into ratio.
        if len(parts) == 3 and is_timeframe_token(parts[1]):
            text = parts[2]
    for stem in _IDENTITY_STEMS:
        if text == stem or text.startswith(f"{stem}_"):
            return True
    return False


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
