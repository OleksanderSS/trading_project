"""Each timeframe walked all 27 targets and skipped the ~20 it cannot have.

Splitting the heavy branch by timeframe (bb7faa06) left the target list
taken once from the whole merged frame, so every timeframe then announced

    🎯 Таргет: target_up_1d [15m]
      ⚠️ Лише 0 зразків, занадто мало.

for targets that do not exist at that cadence. They do not exist by DESIGN
-- a one-day-ahead direction target has no meaning on a 15-minute bar, and
the export partitions them accordingly. Measured on the 2026-08-06 batch,
every one of the 22 tickers has exactly 18 trainable targets on 1d, 7 on
15m and 5 on 60m: 660 real combinations and 1,122 announcements of nothing.

Cheap in time -- the sample check returns before any work -- but it buried
the real lines in a log read after a multi-hour run, and wrote an empty
entry per skipped target into colab_results.json.

No model changes. The set trained is identical; only the announcing stops.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _controller():
    path = Path("scripts/colab/colab_clean_cell.py")
    if not path.exists():
        pytest.skip("colab trainer script not present")
    spec = importlib.util.spec_from_file_location("colab_clean_cell", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"colab trainer imports unavailable here: {exc}")
    return module.ColabTrainingController


def test_the_threshold_has_one_definition():
    """The per-timeframe list and the per-target guard must agree.

    Two copies of a threshold are two thresholds the moment one is tuned --
    and the disagreement would be invisible: a target admitted by one and
    refused by the other simply produces no model and no explanation.
    """
    import inspect
    import textwrap

    controller = _controller()
    assert isinstance(controller._MIN_TRAINING_SAMPLES, int)

    for method in (controller._process_ticker, controller._process_target):
        source = textwrap.dedent(inspect.getsource(method))
        assert "_MIN_TRAINING_SAMPLES" in source, method.__name__
        assert "< 50" not in source and ">= 50" not in source, (
            f"{method.__name__} has its own copy of the threshold again"
        )


def test_the_real_export_partitions_targets_by_timeframe():
    """The observation behind the fix, against the artifact itself."""
    features = Path("data/colab/accumulated/main_database/features.parquet")
    targets = Path("data/colab/accumulated/main_database/targets.parquet")
    if not (features.exists() and targets.exists()):
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(targets)
    target_cols = [c for c in frame.columns if c.startswith("target_")]
    controller = _controller()
    minimum = controller._MIN_TRAINING_SAMPLES

    live_per_tf = {
        tf: [c for c in target_cols if rows[c].notna().sum() >= minimum]
        for tf, rows in frame.groupby("interval", sort=True)
    }

    # Every timeframe carries some targets, and none carries all of them --
    # which is exactly why iterating the full list per timeframe was noise.
    for tf, live in live_per_tf.items():
        assert live, f"{tf} carries no trainable target at all"
        assert len(live) < len(target_cols), (
            f"{tf} carries every target; the partition this fix rests on is gone"
        )


def test_an_hourly_horizon_target_is_kept_on_the_intraday_timeframe():
    """target_hourly_volume_spike_1h on 15m rows is NOT a mistake.

    It is an hourly HORIZON observed at a 15-minute CADENCE: given the state
    at this bar, is there a volume spike within the next hour. 10,003
    non-null values on 15m rows in the 2026-08-06 batch. Dropping it as
    "wrong timeframe" would delete a legitimate model -- and these three
    _1h targets are precisely the ones that used to be trained as one fit
    over two bar sizes.
    """
    targets = Path("data/colab/accumulated/main_database/targets.parquet")
    if not targets.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(targets)
    if "target_hourly_volume_spike_1h" not in frame.columns:
        pytest.skip("that target is not in this export")

    counts = (
        frame.groupby("interval")["target_hourly_volume_spike_1h"]
        .apply(lambda s: s.notna().sum())
        .to_dict()
    )

    assert counts.get("15m", 0) > 0
    assert counts.get("60m", 0) > 0
