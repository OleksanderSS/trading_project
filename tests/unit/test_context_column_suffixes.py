"""Context columns arrive with a timeframe suffix; the lookup wanted them bare.

ContextMapEnricher runs per timeframe, so what reaches Stage 4 is
context_pattern_seq_1d / context_pattern_seq_60m and
context_fingerprint_1d / _60m. Verified against the exported
features.parquet from the 2026-08-02 prepare run: 1,189 columns, containing

    context_fingerprint_1d, context_pattern_seq_1d, context_pattern_id_1d
    context_fingerprint_60m, context_pattern_seq_60m, context_pattern_id_60m

and NOT one of the bare names.

_latest_context_value looked only for the bare form, so it always returned
its default. That had two consequences, and the second explains a finding
from earlier in this audit:

- context_pattern_seq never reached the diary, so KNN similarity had nothing
  to search (fc30bfb6, and the plumbing fix 9e4cccff was therefore still
  inert);
- _build_context_fingerprint's "reuse the existing fingerprint" branch never
  matched either, so every fingerprint fell through to the SHA-256 payload
  hash. That is why all 19,305 diary rows carry an identity hash instead of
  the tri-state string, which a hash cannot be vectorised into.

The suffixed forms are now accepted, preferring the timeframe being
processed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage


def _frame(**columns):
    return pd.DataFrame({name: [value] for name, value in columns.items()})


def test_the_suffixed_column_is_found():
    frame = _frame(context_pattern_seq_1d="1|1|0>>1|0|0")

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None, timeframe="1d"
    ) == "1|1|0>>1|0|0"


def test_the_bare_column_still_wins_when_present():
    frame = _frame(
        context_pattern_seq="BARE",
        context_pattern_seq_60m="SUFFIXED",
    )

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None
    ) == "BARE"


def test_the_requested_timeframe_is_preferred():
    frame = _frame(
        context_pattern_seq_1d="DAILY",
        context_pattern_seq_60m="HOURLY",
    )

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None, timeframe="60m"
    ) == "HOURLY"


def test_another_timeframe_beats_no_context_at_all():
    """A neighbouring timeframe's context is worth more than falling through
    to a hash that cannot be compared to anything."""
    frame = _frame(context_pattern_seq_15m="INTRADAY")

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None, timeframe="1d"
    ) == "INTRADAY"


def test_a_frame_without_any_variant_returns_the_default():
    frame = _frame(close=100.0)

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default="fallback", timeframe="1d"
    ) == "fallback"


def test_nulls_are_skipped_in_favour_of_the_last_real_value():
    frame = pd.DataFrame({"context_pattern_seq_1d": ["1|0", None, "0|1", None]})

    assert ModelingStage._latest_context_value(
        frame, ("context_pattern_seq",), default=None, timeframe="1d"
    ) == "0|1"


def test_the_exported_features_carry_only_suffixed_names():
    """The observation this fix rests on, checked against the real artifact."""
    from pathlib import Path

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    columns = set(pd.read_parquet(path).columns)

    assert "context_pattern_seq" not in columns
    assert "context_fingerprint" not in columns
    assert any(c.startswith("context_pattern_seq_") for c in columns)
    assert any(c.startswith("context_fingerprint_") for c in columns)


def test_the_real_export_now_yields_a_vectorisable_fingerprint():
    """The end this serves: a fingerprint KNN can measure distance on."""
    from pathlib import Path

    from src.meta_learning.memory.contextual_weight_calculator import (
        ContextualWeightCalculator,
    )

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(path).head(200)
    fingerprint = ModelingStage._latest_context_value(
        frame, ("context_fingerprint",), default=None, timeframe="1d"
    )

    assert fingerprint, "no fingerprint found in the exported features"
    assert ContextualWeightCalculator.fingerprint_to_vec(fingerprint), (
        "the fingerprint carries no vector, which is what left KNN starved"
    )
