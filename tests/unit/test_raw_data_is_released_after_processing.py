"""The raw tables were carried through stage 3 for nothing.

`stage_outputs['raw_data']` holds every collected table. Nothing after stage 2
reads it -- checked across all of src/, where every other mention of the name
is a local variable inside a collector -- and it is already on disk as
`main_database_stage1_raw_data_*.parquet`, 360 MiB compressed. So a full second
copy of the data rode along through stage 3's two and a quarter hours.

Why it matters beyond tidiness: stage 3 peaks at 2.67 GiB and 2.04 of that is
held BEFORE its first phase, left behind by collection and processing.
Enrichment itself costs about 0.13 GiB. This is the part that scales with
tickers, so it is what stands between 22 names and 110.

It is released only after ProcessingStage produces cleaned_data. Processing is
the one thing that reads it, and a run that skips stage 2 must keep it.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


def _orchestrator():
    instance = PipelineOrchestrator.__new__(PipelineOrchestrator)
    instance.logger = logging.getLogger("probe")
    return instance


def _outputs():
    return {
        "raw_data": {"market_data_raw": pd.DataFrame({"x": range(100)})},
        "cleaned_data": {"market_data": pd.DataFrame({"x": range(90)})},
    }


def test_it_is_dropped_once_processing_has_consumed_it():
    outputs = _outputs()
    _orchestrator()._release_raw_data("ProcessingStage", outputs)
    assert "raw_data" not in outputs
    assert outputs["cleaned_data"], "cleaned_data must survive"


def test_it_survives_every_other_stage():
    """Collection has just produced it; stage 3 onward never sees this call."""
    for stage in ("CollectionStage", "FeatureEngineeringStage", "ModelingStage"):
        outputs = _outputs()
        _orchestrator()._release_raw_data(stage, outputs)
        assert "raw_data" in outputs, f"dropped by {stage}"


def test_it_is_kept_when_processing_produced_nothing(caplog):
    """Then the raw tables are the only way anything could recover."""
    outputs = {"raw_data": {"t": pd.DataFrame({"x": [1]})}, "cleaned_data": {}}
    with caplog.at_level(logging.WARNING):
        _orchestrator()._release_raw_data("ProcessingStage", outputs)

    assert "raw_data" in outputs
    assert any("keeping raw_data" in record.message for record in caplog.records)


def test_a_run_without_raw_data_is_not_an_error():
    outputs = {"cleaned_data": {"x": pd.DataFrame({"x": [1]})}}
    _orchestrator()._release_raw_data("ProcessingStage", outputs)   # must not raise


def test_bookkeeping_never_ends_the_run():
    """Reporting how much was freed must not be able to fail the pipeline."""
    class _Awkward(pd.DataFrame):
        @property
        def _constructor(self):
            return _Awkward

        def memory_usage(self, *args, **kwargs):
            raise RuntimeError("no")

    outputs = {
        "raw_data": {"t": _Awkward({"x": [1]})},
        "cleaned_data": {"t": pd.DataFrame({"x": [1]})},
    }
    _orchestrator()._release_raw_data("ProcessingStage", outputs)
    assert "raw_data" not in outputs, (
        "the release itself must happen even when measuring it fails"
    )
