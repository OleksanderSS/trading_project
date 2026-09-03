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
import pyarrow.parquet as pq
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


def test_a_longer_feature_name_does_not_answer_for_the_shorter_one():
    """MARKET_REGIME_ENCODED_1d holds detector confidence, not a regime.

    A plain startswith("MARKET_REGIME_") match accepts it, and 0.72 would
    be stringified and filed as the market regime for that model.
    """
    frame = _frame(MARKET_REGIME_ENCODED_1d=0.72)

    assert ModelingStage._latest_context_value(
        frame, ("MARKET_REGIME",), default="unknown", timeframe="60m"
    ) == "unknown"


def test_the_regime_column_is_read_under_the_name_its_producer_writes():
    """technical_analysis_enricher writes MARKET_REGIME, upper case."""
    frame = _frame(
        MARKET_REGIME_1d="TRENDING_UP",
        MARKET_REGIME_ENCODED_1d=0.72,
    )

    assert ModelingStage._latest_context_value(
        frame,
        ("MARKET_REGIME", "market_regime", "regime"),
        default="unknown",
        timeframe="1d",
    ) == "TRENDING_UP"


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


def _columns_for(path, bases):
    """Only the columns a lookup can possibly answer from, read from the schema.

    `_latest_context_value` builds its candidates from the base names plus
    whatever timeframe-suffixed variants exist in `frame.columns`, in column
    order. Restricting the read to names that start with one of those bases
    therefore leaves the candidate set -- and its order -- identical, while
    turning a 7.0 GiB allocation into a few megabytes.
    """
    names = pq.read_schema(path).names
    return [
        name for name in names
        if any(name == base or name.startswith(f"{base}_") for base in bases)
    ]


def test_the_exported_features_carry_only_suffixed_names():
    """The observation this fix rests on, checked against the real artifact."""
    from pathlib import Path

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    # The schema, not the data. Reading the frame to look at its column
    # names asked pyarrow for a 7.0 GiB allocation and failed on a machine
    # with 5 free -- the same whole-frame-where-a-slice-would-do shape that
    # killed the pipeline run of 2026-08-31.
    columns = set(pq.read_schema(path).names)

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

    frame = pd.read_parquet(
        path, columns=_columns_for(path, ("context_fingerprint",))
    ).head(200)
    fingerprint = ModelingStage._latest_context_value(
        frame, ("context_fingerprint",), default=None, timeframe="1d"
    )

    assert fingerprint, "no fingerprint found in the exported features"
    assert ContextualWeightCalculator.fingerprint_to_vec(fingerprint), (
        "the fingerprint carries no vector, which is what left KNN starved"
    )


@pytest.mark.parametrize("timeframe", ["1d", "60m"])
def test_the_real_export_yields_a_regime_label_not_a_number(timeframe):
    """The artifacts recorded market_regime='unknown' with the answer in hand."""
    from pathlib import Path

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    bases = ("MARKET_REGIME", "market_regime", "regime")
    wanted = _columns_for(path, bases)
    if not wanted:
        # Not a stale assertion to delete: MARKET_REGIME was switched off at
        # the source on 2026-08-28 because it cost 5.4 hours of a twelve-hour
        # rebuild and failed its own leading-feature test. The absence is a
        # decision, and this test cannot judge it -- what it CAN judge is the
        # shape of the value when one exists. The consequence of the absence
        # (every champion keyed by the literal 'normal') is register #182 and
        # is now warned about in ModelingStage.run.
        pytest.skip(
            "no MARKET_REGIME column in this batch; it is opt-in behind "
            "MARKET_REGIME_FEATURES since 2026-08-28 (register #182)"
        )

    frame = pd.read_parquet(path, columns=wanted)
    regime = ModelingStage._latest_context_value(
        frame,
        bases,
        default="unknown",
        timeframe=timeframe,
    )

    assert regime != "unknown"
    # A label, not the confidence float from MARKET_REGIME_ENCODED_*.
    assert regime.replace("_", "").isalpha(), regime


def test_the_training_pattern_is_read_with_its_timeframe_suffix():
    """The axis the "Regime-Aware Training Arena" is built on never varied.

    ModelingStage.run set

        current_pattern = df['context_pattern_id'].iloc[-1]
                          if 'context_pattern_id' in df.columns else 'normal'

    against the BARE name, while ContextMapEnricher emits
    context_pattern_id_1d / _60m. The condition was never true, so every
    champion was filed under the literal 'normal' -- confirmed on the
    2026-08-04 run: all 506 of them. Reading it with the timeframe yields 44
    distinct (timeframe, pattern) pairs on the same export.
    """
    import inspect
    import textwrap

    from tests.contracts._lookahead_scan import _code_only

    source = textwrap.dedent(inspect.getsource(ModelingStage.run))
    # Comments and docstrings stripped: the replacement QUOTES the old
    # expression to explain itself, and this assertion failed on that
    # comment the first time it ran. Prose is not code -- the same
    # distinction the lookahead scanner had to learn.
    code = "\n".join(_code_only(source).values())

    assert "_latest_context_value(" in code
    assert "df['context_pattern_id'].iloc[-1]" not in code


def test_the_training_pattern_uses_a_key_that_actually_groups():
    """context_pattern_id is unique per row, so it cannot key a champion.

    It is a SHA-256 of five concatenated fingerprints, each ~185 tri-state
    drivers, so the space is astronomically large. Measured on the export:
    15m 14,209 rows -> 14,201 distinct; 60m 5,652 -> 5,647; 1d 7,128 ->
    7,112. One observation per pattern carries the same information as the
    constant 'normal' it replaced -- both extremes, neither a regime.

    MARKET_REGIME has 6-8 values per timeframe and is well spread, which is
    what a champion keyed by regime needs.

    Read from the model's OWN timeframe: a 15m model's edge is a 15m
    phenomenon, and a daily regime would hold nearly constant across a
    session. The daily view is still available to the model as a feature via
    the assembler's ctx_1d_* columns.
    """
    import inspect
    import io
    import textwrap
    import tokenize

    source = textwrap.dedent(inspect.getsource(ModelingStage.run))

    # COMMENTS stripped, string literals KEPT. The lookahead scanner's
    # _code_only blanks strings as well, which is correct when hunting for
    # operations and wrong here: the thing being asserted on IS a string
    # literal, so that helper erases the evidence. It did, on the first run
    # of this test.
    code = " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    )

    assert "MARKET_REGIME" in code
    assert "context_pattern_id" not in code, (
        "the champion key is back on a row-unique hash"
    )
