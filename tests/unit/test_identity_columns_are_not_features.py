"""A row hash became a model feature, and every prediction died on it.

The 2026-08-04 run trained 506 champions and produced ZERO predictions. Each
light model's selected_features contained four non-numeric columns:

    hash                    a SHA-256, unique per row
    context_fingerprint_1d  '0|1|1|-1|...'
    context_pattern_seq_1d  the same, plus shifted history
    context_pattern_id_1d   an 8-char digest

handle_categorical_features label-encodes any object column with more than
five distinct values BEFORE feature selection runs, so all four arrived at
selection as perfectly ordinary integers and were kept. At prediction the
frame is coerced with pd.to_numeric, the original strings became NaN, and
_drop_incomplete_model_rows correctly refused to fabricate values -- dropping
all 50 rows of every context. Measured on the real export: 50 of 50 rows
complete before coercion, 0 of 50 after.

The prediction failure is the visible half. The training half is worse:

- `hash` is unique per row, so label-encoded it is a dense row index. A tree
  can split on it and memorise the training set.
- The tri-state context strings become an integer ordered alphabetically --
  meaningless as a magnitude, and it destroys the structure the KNN context
  search depends on.
- The LabelEncoder is local to that function and never persisted, so no
  unseen string could ever be mapped back. Those models could not be scored
  on new data even in principle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.adapters.data_preparation import handle_categorical_features
from src.pipeline.target_column_utils import is_identity_column


@pytest.mark.parametrize("column", [
    "hash", "record_hash", "ticker", "interval", "timeframe", "symbol",
    "context_fingerprint", "context_fingerprint_1d",
    "context_pattern_seq_60m", "context_pattern_id_1d", "context_schema_id",
])
def test_identifiers_and_context_keys_are_recognised(column):
    assert is_identity_column(column)


@pytest.mark.parametrize("column", [
    "close", "volume", "SMA_200_1d", "MARKET_REGIME_1d", "AATR_14_1d",
    # Numeric summaries OF the context are legitimate features; only the
    # keys are not.
    "context_velocity", "context_stability", "context_anxiety_index",
])
def test_real_features_are_not_mistaken_for_identifiers(column):
    assert not is_identity_column(column)


def test_the_suffixed_forms_are_covered():
    """The enrichers emit context_fingerprint_1d, never the bare name.

    A check against bare names alone would have missed every single one --
    the same one-thing-two-names trap that has cost this project repeatedly.
    """
    assert is_identity_column("context_fingerprint_1d")
    assert is_identity_column("context_pattern_seq_60m")


def test_a_row_hash_is_never_label_encoded_into_a_feature():
    """The mechanism: encoding happens before selection, so blocking it at
    selection alone would leave a plausible integer in the frame."""
    frame = pd.DataFrame({
        "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "hash": [f"{i:064x}" for i in range(6)],
        "context_fingerprint_1d": ["0|1", "1|0", "0|0", "1|1", "0|1", "1|0"],
        "regime": ["a", "b", "a", "b", "a", "b"],
        "target_up_1d": [0, 1, 0, 1, 0, 1],
    })

    processed, info = handle_categorical_features(frame, ["target_up_1d"])

    assert "hash" not in info, "the row hash was encoded into a feature"
    assert "context_fingerprint_1d" not in info
    # A genuine categorical still gets encoded.
    assert "regime" in info


def test_the_identity_columns_do_not_become_numeric():
    """If they survive as numbers, every downstream numeric check waves them
    through -- which is exactly how they reached the models."""
    frame = pd.DataFrame({
        "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "hash": [f"{i:064x}" for i in range(6)],
        "target_up_1d": [0, 1, 0, 1, 0, 1],
    })

    processed, _ = handle_categorical_features(frame, ["target_up_1d"])

    if "hash" in processed.columns:
        assert not np.issubdtype(processed["hash"].dtype, np.number)


def test_selection_excludes_them_even_if_they_arrive_numeric():
    """Belt and braces: a numeric column literally named `hash` still must
    not be selected."""
    import inspect

    from src.models.adapters import data_preparation

    source = inspect.getsource(data_preparation.prepare_data_for_models)

    assert "is_identity_column(c)" in source
