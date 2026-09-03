"""A requested cadence that produced nothing must fail the run, not shrink it.

Run 14 (2026-09-02) asked for ['15m', '1d', '1h'] and delivered ['1d', '60m'].
Every part of that was detected and recorded:

    ERROR  Batch 'main_database' was asked for timeframe(s) ['15m'] and
           produced NONE. Delivered: ['1d', '60m']
    batch_metadata.json:  "timeframes_missing": ["15m"]

and two seconds later:

    INFO   Pipeline completed successfully for batch: main_database

Nothing was hidden. What was missing was a VERDICT: evidence collected at the
stage that noticed it, and no one turning it into a pass or a fail at the
boundary. That is the same defect as REGISTER #201 and #211, in a third place.

The cost was measured, not imagined. 15m was dropped twelve minutes into that
run; stage 3 then spent two hours enriching the two cadences that survived, and
the batch was planned on as though it held three. It was found by reading the
log by hand (REGISTER #228, #229).

WHAT IS TESTED HERE, and why in this shape:

The wiring, not the function. `_create_batch_metadata` had computed
`timeframes_missing` correctly all along -- the field was then dropped by
`_assemble_preparation_result` on the way out, so the caller that decides pass
or fail never saw it. A test of either half alone passes while the run still
lies. So one test walks the real `ColabManager` from metadata to returned
result, and another feeds that result to the real verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runner():
    """The run script, imported by path -- it is not a package module."""
    spec = importlib.util.spec_from_file_location(
        "run_hybrid_pipeline", PROJECT_ROOT / "run_hybrid_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**kwargs) -> argparse.Namespace:
    base = {"batch_name": "test_batch", "allow_missing_timeframes": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


def _result(missing: list[str], **extra) -> dict:
    result = {
        "batch_dir": "/tmp/x",
        "batch_name": "test_batch",
        "timeframes_requested": ["15m", "1d", "1h"],
        "timeframes_delivered": ["1d", "60m"],
        "timeframes_missing": missing,
    }
    result.update(extra)
    return result


def test_a_missing_timeframe_fails_the_run(runner):
    """The exact shape of run 14."""
    assert runner.run_failed(_result(["15m"]), _args()) is True


def test_all_timeframes_delivered_passes(runner):
    assert runner.run_failed(_result([]), _args()) is False


def test_the_gap_can_be_tolerated_but_only_on_purpose(runner):
    """Deliberate is fine; accidental is the whole problem. The flag puts the
    intent in the command instead of in nobody's memory."""
    assert runner.run_failed(_result(["15m"]), _args(allow_missing_timeframes=True)) is False


def test_the_older_verdicts_still_hold(runner):
    """This check is added to the boundary, not substituted for it."""
    assert runner.run_failed(None, _args()) is True
    assert runner.run_failed(_result([], status="failed"), _args()) is True
    assert runner.run_failed(
        _result([], stage_6={"execution_status": "no_predictions"}), _args()
    ) is True


def test_prepare_carries_the_gap_out_to_the_caller(tmp_path):
    """The half that was broken: the number existed and was thrown away.

    `_create_batch_metadata` computed `timeframes_missing` and logged it;
    `_assemble_preparation_result` then built a dictionary without it, so the
    boundary had nothing to judge. Both real methods are used here, because
    that seam is exactly where the fact was lost.
    """
    from src.pipeline.hybrid.colab_manager import BatchPreparationConfig, ColabManager

    manager = ColabManager.__new__(ColabManager)
    features = pd.DataFrame({
        "interval": ["1d"] * 3 + ["60m"] * 3,
        "ticker": ["AAPL"] * 6,
        "datetime": pd.date_range("2026-01-01", periods=6, tz="UTC"),
    })
    config = BatchPreparationConfig(
        tickers=["AAPL"], timeframes=["15m", "1d", "1h"],
    )

    # Real files: the metadata step hashes them.
    f_path, t_path = tmp_path / "features.parquet", tmp_path / "targets.parquet"
    features.to_parquet(f_path)
    features.to_parquet(t_path)

    metadata = manager._create_batch_metadata(
        "test_batch", "20260902", config, features, features,
        f_path, t_path, None,
    )
    assert metadata["timeframes_missing"] == ["15m"], metadata

    result = manager._assemble_preparation_result(
        tmp_path, "test_batch", tmp_path / "m.json", metadata,
        {}, config, None,
    )
    assert result["timeframes_missing"] == ["15m"], (
        "prepare computed the gap and dropped it before returning; the "
        "boundary cannot fail on a fact it never receives"
    )
    assert result["timeframes_requested"] == ["15m", "1d", "1h"]
    assert result["timeframes_delivered"] == ["1d", "60m"]


def test_1h_and_60m_are_not_reported_as_a_gap():
    """The request says 1h, the data says 60m. A raw set difference would
    report a phantom gap and hide the real one."""
    from src.pipeline.hybrid.colab_manager import ColabManager

    manager = ColabManager.__new__(ColabManager)
    assert manager._missing_timeframes(["1d", "1h"], {"1d", "60m"}) == []


def test_the_flag_exists_on_the_command_line():
    """The failure message tells the user to pass it; it has to be real."""
    from src.cli.argument_parser import create_argument_parser

    args = create_argument_parser().parse_args(["--allow-missing-timeframes"])
    assert args.allow_missing_timeframes is True
    assert create_argument_parser().parse_args([]).allow_missing_timeframes is False
