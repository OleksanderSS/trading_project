"""A machine without `src/` must still be able to honour the seal.

REGISTER #153, and the half of it that was still live. The champion-metric half
was closed on 2026-09-04 (`test_a_gated_model_is_not_outranked_by_an_ungated_one`).
This is the other half: the Colab cell knew nothing about the seal.

MEASURED 2026-09-05, which is what turned this from a worry into a defect:

    the batch the cell reads spans 1996-08-26 .. 2026-09-01
    it holds 82,094 sealed daily rows -- 11.6% of the daily frame
    the cell validates on the LAST 20% of each series
    the last 20% of the daily rows begins 2021-07-12

So the cell's validation window ran straight through the sealed period, and
every champion it selected was selected on the one un-spent confirmation this
project has.

WHY A SIDECAR AND NOT AN IMPORT. The cell is a standalone file pasted into a
notebook, and its own comment describes the real workflow: "one real workflow
copies features.parquet and targets.parquet from a local machine to Drive". In
that workflow there is no `src/` to import from. The seal therefore has to
travel WITH the batch.

WHY THAT IS NOT A TENTH DEFINITION. #264 found the seal date written nine
times. `sealed_period.export_to` reads `SEAL_START` and writes it out; the
value still exists once. A delivery is not a definition -- what makes the
difference is that the reader REFUSES when the delivery is missing instead of
assuming there is no seal, which would be #182 in a new place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.parquet_union_writer import write_union
from src.pipeline.sealed_period import SEAL_FILE, SEAL_START, export_to

CELL = Path(__file__).resolve().parents[2] / "scripts" / "colab" / "colab_clean_cell.py"


def test_the_export_carries_the_date_and_its_meaning(tmp_path):
    path = export_to(tmp_path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    assert pd.Timestamp(payload["seal_start"]) == SEAL_START
    assert payload["source"].endswith("sealed_period.py"), (
        "the sidecar does not name where the value came from, so a reader "
        "cannot tell a delivery from a tenth copy"
    )
    assert "refuse" in payload["meaning"], (
        "the file does not tell its reader what to do when it is absent, "
        "which is the only instruction that matters"
    )


def test_a_missing_directory_is_reported_not_raised(tmp_path, caplog):
    """A batch write must not fail for want of a sidecar; the refusal that
    matters is at the reading end."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert export_to(tmp_path / "absent") is None
    assert "#153" in " ".join(r.getMessage() for r in caplog.records)


def test_writing_a_batch_ships_the_seal_beside_it(tmp_path):
    """Both writers of this artefact go through `write_union` (#138), so the
    sidecar cannot be attached to one path and missed by the other."""
    destination = tmp_path / "features.parquet"
    write_union({"1d": pd.DataFrame({"a": [1, 2, 3]})}, destination)

    assert (tmp_path / SEAL_FILE).exists(), (
        "the batch was written without the seal beside it, so a machine that "
        "copies the parquet files has nothing to honour"
    )


def test_the_cell_refuses_when_the_sidecar_is_absent():
    """Reading a missing file as "there is no seal" restores the defect the
    moment someone copies two files instead of three."""
    source = CELL.read_text(encoding="utf-8")
    assert "raise FileNotFoundError(" in source
    marker = source.index("def _apply_seal")
    window = source[marker:marker + 2600]
    # Matched on a phrase that lies inside ONE string literal: the message is
    # built from several f-string fragments, so "This cell will not train"
    # never appears contiguously in the source and asserting on it tests the
    # line wrapping rather than the behaviour.
    assert "not train without it" in window, (
        "the cell no longer refuses on a missing seal file"
    )
    assert "#153" in window


def test_the_cell_drops_the_sealed_tail_from_both_frames():
    source = CELL.read_text(encoding="utf-8")
    marker = source.index("def _apply_seal")
    window = source[marker:marker + 4200]
    assert 'for name in ("features_df", "targets_df")' in window, (
        "the seal is applied to one frame and not the other, so the targets "
        "still carry the holdout"
    )
    # Per-frame comparison: one absolute date would delete a short frame
    # whole, which is what the first version did to the 60m rows.
    assert 'frame["datetime"] < next(iter(seals.values()))' in window
    assert 'row["datetime"] < seals.get(' in window


def test_the_cell_stops_when_the_seal_leaves_nothing():
    """An empty frame after sealing is not a small batch -- it is a batch that
    begins inside the holdout, and training on it teaches nothing while
    spending the seal."""
    source = CELL.read_text(encoding="utf-8")
    # Bounded by the NEXT definition rather than a character count: the window
    # had to be widened twice as the method grew, and a test that fails
    # because a comment was added is a test measuring the wrong thing.
    marker = source.index("def _apply_seal")
    end = source.index("\n    def ", marker + 10)
    window = source[marker:end]
    assert "removed every row" in window


def test_the_cell_applies_the_seal_before_anything_splits():
    """Order is the whole point: a seal applied after the split protects
    nothing."""
    source = CELL.read_text(encoding="utf-8")
    applied = source.index("self._apply_seal()")
    split = source.index("def _chronological_split")
    load_end = source.index("return self.features_df, self.targets_df")
    assert applied < load_end, "the seal is applied after load_data returns"
    assert applied < split or "def _chronological_split" in source[split:], (
        "the split runs on unsealed frames"
    )


@pytest.mark.parametrize("token", ["sealed_period.json", "SEAL_FILE"])
def test_the_two_ends_name_the_same_file(token):
    """The writer and the reader must agree on the filename, and neither may
    hold a second spelling of it."""
    cell = CELL.read_text(encoding="utf-8")
    module = (Path(__file__).resolve().parents[2] / "src" / "pipeline"
              / "sealed_period.py").read_text(encoding="utf-8")
    assert token in cell and token in module


def test_the_cell_seals_each_frame_by_its_own_span():
    """One absolute date is not a seal for a short frame -- it is a deletion.

    Measured on the live batch: sealing everything at 2023-09-01 withheld
    380,938 of 380,938 sixty-minute rows, ONE HUNDRED PERCENT, because that
    frame's history is shorter than the distance back to the date. The module
    already carried the note -- "a seal that leaves nothing to explore is not
    a stricter seal, it is a broken one" -- and the first version of the cell
    reproduced the defect the note was written about.

    Under the module's rule the same batch withholds 12% of the daily frame
    and 18% of the hourly one: 14.0% overall against 42.6%.
    """
    source = CELL.read_text(encoding="utf-8")
    marker = source.index("def _apply_seal")
    window = source[marker:marker + 4200]
    assert "seal_share" in window, (
        "the cell reads only the absolute date, so any frame shorter than the "
        "distance back to it is withheld entirely"
    )
    assert "quantile(1.0 - share)" in window
    assert "groupby(column)" in window, (
        "one seal is applied to every timeframe at once"
    )


def test_the_cell_and_the_module_agree_on_the_rule():
    """Two implementations of one decision is family C. The cell cannot import
    the module, so the test compares their arithmetic on the same input."""
    import numpy as np

    from src.pipeline.sealed_period import SEAL_SHARE, seal_start_for

    stamps = pd.Series(pd.date_range("2024-08-19", "2026-08-28", freq="h",
                                     tz="UTC"))
    module_answer = pd.Timestamp(seal_start_for(stamps)).tz_localize(None)

    naive = stamps.dt.tz_localize(None)
    distinct = pd.Series(naive.dropna().unique()).sort_values()
    by_span = pd.Timestamp(distinct.quantile(1.0 - SEAL_SHARE))
    cell_answer = (max(SEAL_START.tz_localize(None), by_span)
                   if distinct.iloc[-1] >= SEAL_START.tz_localize(None)
                   else by_span)

    assert abs((module_answer - cell_answer).total_seconds()) < 3600, (
        f"the cell would seal at {cell_answer} and the module at "
        f"{module_answer}; one decision with two answers is family C"
    )
