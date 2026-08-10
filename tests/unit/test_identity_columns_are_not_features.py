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


# --------------------------------------------------------------------------
# The cross-timeframe prefix defeated the stem check.
#
# The assembler copies a lower timeframe's columns forward as ctx_<tf>_*,
# identity columns included. ctx_1d_context_fingerprint_1d is the same
# string as context_fingerprint_1d one prefix later, and a stem check
# anchored at the start of the name did not see it.
#
# It cost the 2026-08-09 run 313 of 660 contexts. What made it hard to find:
# the columns are NOT empty in features.parquet -- they hold strings, so an
# all-NaN scan comes back clean (it did, and the hypothesis was wrongly
# discarded). Stage 5 coerces with pd.to_numeric first, which turns them
# into NaN, and a row with a NaN in a required feature is dropped. Three
# such columns emptied every 50-row prediction window.
# --------------------------------------------------------------------------


import pytest

from src.pipeline.target_column_utils import is_identity_column


@pytest.mark.parametrize("column", [
    "ctx_1d_context_fingerprint_1d",
    "ctx_1d_context_pattern_seq_1d",
    "ctx_1d_context_pattern_id_1d",
    "ctx_60m_context_fingerprint_60m",
    "ctx_60m_context_pattern_id_60m",
    "ctx_1d_ticker",
    "ctx_15m_hash",
])
def test_a_prefixed_identity_column_is_still_an_identity_column(column):
    assert is_identity_column(column) is True


@pytest.mark.parametrize("column", [
    "ctx_1d_open",
    "ctx_1d_high",
    "ctx_1d_SMA_100_1d",
    "ctx_60m_ATR_14_60m",
    "ctx_1d_market_context_put_call_ratio_1d",
])
def test_a_prefixed_real_feature_is_left_alone(column):
    """The daily context's genuine market columns are the whole point of
    carrying it forward."""
    assert is_identity_column(column) is False


def test_only_a_genuine_timeframe_is_stripped():
    """Stripping two fields from anything beginning with ctx_ would turn
    ctx_volume_ratio into 'ratio' and ctx_ticker_count into 'count'."""
    assert is_identity_column("ctx_volume_ratio") is False
    assert is_identity_column("ctx_something_ticker") is False


def test_the_three_columns_named_in_the_2026_08_09_log_are_caught():
    """The exact strings Stage 5 reported as null in every candidate row."""
    for column in (
        "ctx_1d_context_fingerprint_1d",
        "ctx_1d_context_pattern_seq_1d",
        "ctx_1d_context_pattern_id_1d",
    ):
        assert is_identity_column(column), column


def test_the_real_export_has_exactly_the_six_prefixed_identity_columns():
    """Pins the measurement: 6 of 1,978 columns, none of them a feature."""
    from pathlib import Path

    import pandas as pd

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    columns = list(pd.read_parquet(path).columns)
    prefixed = [c for c in columns if c.startswith("ctx_") and is_identity_column(c)]

    assert len(prefixed) == 6, sorted(prefixed)
    assert all("context_" in c for c in prefixed), sorted(prefixed)


# --------------------------------------------------------------------------
# A feature that cannot be reproduced at prediction time is not a feature.
#
# handle_categorical_features label-encoded any categorical with more than
# five values, using an encoder built and discarded in the same call.
# Training then saw MARKET_REGIME_1d as integers while the prediction path
# read the raw frame, where it is still 'TRENDING_UP' -- pd.to_numeric turns
# that into NaN and drops every candidate row.
#
# Measured on the 2026-08-10 run: 101 contexts blocked, every one naming a
# MARKET_REGIME_* column as null in all 50 rows. All five columns reaching
# that branch are MARKET_REGIME variants, and each already has a numeric
# MARKET_REGIME_ENCODED_* counterpart, so dropping them loses nothing.
# --------------------------------------------------------------------------


def test_a_high_cardinality_categorical_is_dropped_not_label_encoded():
    import numpy as np
    import pandas as pd

    from src.models.adapters.data_preparation import handle_categorical_features

    frame = pd.DataFrame({
        "MARKET_REGIME_1d": ["A", "B", "C", "D", "E", "F"] * 3,
        "close": np.arange(18, dtype=float),
    })

    out, info = handle_categorical_features(frame, exclude_cols=[])

    assert "MARKET_REGIME_1d" not in out.columns
    assert info["MARKET_REGIME_1d"] == "dropped_unpersisted_encoding"


def test_the_numeric_counterpart_is_kept():
    """MARKET_REGIME_ENCODED_* carries the same information and survives the
    prediction path, which is the whole reason dropping is safe."""
    import numpy as np
    import pandas as pd

    from src.models.adapters.data_preparation import handle_categorical_features

    frame = pd.DataFrame({
        "MARKET_REGIME_1d": ["A", "B", "C", "D", "E", "F"] * 3,
        "MARKET_REGIME_ENCODED_1d": np.arange(18, dtype=float),
    })

    out, _ = handle_categorical_features(frame, exclude_cols=[])

    assert "MARKET_REGIME_ENCODED_1d" in out.columns


def test_a_low_cardinality_categorical_is_still_one_hot_encoded():
    """One-hot creates NEW named columns, so it does not have the same
    problem -- do not sweep it up with the fix."""
    import numpy as np
    import pandas as pd

    from src.models.adapters.data_preparation import handle_categorical_features

    frame = pd.DataFrame({
        "volatility_regime_1d": ["low", "mid", "high"] * 6,
        "close": np.arange(18, dtype=float),
    })

    out, info = handle_categorical_features(frame, exclude_cols=[])

    assert info["volatility_regime_1d"] == "one-hot"
    assert any(c.startswith("volatility_regime_1d_") for c in out.columns)


def test_no_label_encoder_remains_in_the_path():
    """The encoder was the mechanism; leaving it in invites its return."""
    import inspect

    from src.models.adapters import data_preparation

    source = inspect.getsource(data_preparation.handle_categorical_features)

    assert "LabelEncoder()" not in source, (
        "label encoding is back, and its mapping is still not persisted"
    )


def test_the_real_batch_has_exactly_the_market_regime_columns_affected():
    """Pins the measurement the fix rests on: 5 columns, all MARKET_REGIME."""
    from pathlib import Path

    import pandas as pd

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(path)
    categorical = [
        c for c in frame.select_dtypes(include=["object", "category"]).columns
        if not is_identity_column(c)
        and "ticker" not in c.lower() and "timeframe" not in c.lower()
    ]
    high_cardinality = [c for c in categorical if frame[c].nunique() > 5]

    assert high_cardinality, "nothing reaches the dropped branch any more"
    assert all("MARKET_REGIME" in c for c in high_cardinality), high_cardinality
    for column in high_cardinality:
        twin = column.replace("MARKET_REGIME", "MARKET_REGIME_ENCODED")
        assert twin in frame.columns, f"{column} has no numeric counterpart"
