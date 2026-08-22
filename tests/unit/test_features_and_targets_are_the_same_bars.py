"""Row i of the features must be the same bar as row i of the targets.

`_initialize_stage_outputs` pairs them with `reset_index(drop=True)` and
`pd.concat(axis=1)`, and nothing checked that the two frames were in the same
order. Verified on the 18.08 batch -- 256,062 rows a side, ticker, datetime
and interval identical on every one -- so the assumption has been holding. It
was unchecked, not wrong.

It is one reordering away from silence. This pipeline reorders: `Enricher
'nlp_features' returned the same 28856 rows in a DIFFERENT ORDER` appears in
its own logs. Equal row counts survive that; the pairing does not, and a model
trained on the result learns one bar's features against another bar's outcome.

Gemini's audit called this a fatal misalignment. It is not one today. It is an
invariant that was being assumed, and is now asserted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


def _frames(shuffle_targets: bool = False, drop_a_target_row: bool = False):
    stamps = pd.date_range("2026-01-01", periods=6, freq="D")
    features = pd.DataFrame({
        "ticker": ["AAPL"] * 3 + ["MSFT"] * 3,
        "datetime": list(stamps[:3]) * 2,
        "interval": ["1d"] * 6,
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    targets = features[["ticker", "datetime", "interval"]].copy()
    targets["target_up_1d"] = [0, 1, 0, 1, 1, 0]
    if shuffle_targets:
        targets = targets.iloc[[3, 4, 5, 0, 1, 2]].reset_index(drop=True)
    if drop_a_target_row:
        targets = targets.iloc[1:].reset_index(drop=True)
    return features, targets


def _merge(features, targets):
    orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orchestrator.stages = []
    orchestrator.stages_to_run = None
    import logging
    orchestrator.logger = logging.getLogger("alignment-test")
    context = {"tickers": ["AAPL"], "timeframes": ["1d"],
               "enriched_data": features, "targets_df": targets}
    return orchestrator._initialize_stage_outputs(context)["enriched_data"]


def test_aligned_frames_are_paired_as_before():
    features, targets = _frames()
    merged = _merge(features, targets)
    assert len(merged) == 6
    assert merged["target_up_1d"].tolist() == [0, 1, 0, 1, 1, 0]


def test_a_reordered_target_frame_is_repaired_not_pasted():
    """The failure the row count cannot show."""
    features, targets = _frames(shuffle_targets=True)
    merged = _merge(features, targets)

    assert len(merged) == 6
    # Each row keeps ITS OWN target, not the one that happened to sit at its
    # position in the shuffled frame.
    assert merged["target_up_1d"].tolist() == [0, 1, 0, 1, 1, 0]


def test_mismatched_row_counts_do_not_silently_truncate():
    features, targets = _frames(drop_a_target_row=True)
    merged = _merge(features, targets)
    assert len(merged) == 6
    assert merged["target_up_1d"].isna().sum() == 1


def test_no_shared_keys_and_unequal_rows_is_refused():
    """Nothing left to align on: pairing would be a guess."""
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    targets = pd.DataFrame({"target_up_1d": [0, 1]})
    with pytest.raises(ValueError, match="Cannot align"):
        _merge(features, targets)
