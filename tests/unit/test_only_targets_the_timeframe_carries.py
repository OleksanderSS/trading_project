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

from src.pipeline.sealed_period import SEAL_START


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

    The 15m half of this went red when the batch stopped carrying 15m rows,
    and the reason is structural rather than a loss -- see
    `test_fifteen_minute_bars_can_never_reach_a_measurement` below. The
    assertion is kept for the day 15m rows are explorable again, and skips
    with the measured reason meanwhile rather than pretending to watch.
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

    assert counts.get("60m", 0) > 0, (
        "the hourly-cadence half is gone too, which is a real loss rather "
        "than the structural 15m one")
    if not counts.get("15m", 0):
        pytest.skip(
            "this batch carries no 15m rows, and batch_metadata records "
            "timeframes_missing: ['15m']. Yahoo serves 15m for 60 days, so "
            "every 15m bar obtainable sits inside the seal -- measured 0 of "
            "159,149 explorable")


def test_fifteen_minute_bars_can_never_reach_a_measurement():
    """The 15m timeframe is configured, collected, and structurally dead here.

    Measured 2026-09-07 on `features_15m.parquet`: 159,149 rows spanning
    2026-06-09 to 2026-08-28 -- eighty days, and the seal starts 2023-09-01.
    ZERO rows are explorable, and no future run can change that: Yahoo serves
    15m for sixty days, so the whole window this project can ever obtain lies
    inside the held-back period by construction.

    This is why `timeframes_missing: ['15m']` in the batch metadata is correct
    rather than a loss, and it is the same family as R62 -- a source whose
    collector works perfectly and reaches no measurement.

    It is NOT a reason to stop collecting 15m: paper trading happens in the
    present, where those bars are the live ones. It is a reason to stop
    reading their absence from a research batch as a defect.
    """
    intraday = Path("data/colab/accumulated/main_database/features_15m.parquet")
    if not intraday.exists():
        pytest.skip("no 15m export on disk")

    stamps = pd.to_datetime(
        pd.read_parquet(intraday, columns=["datetime"])["datetime"], utc=True
    ).dt.tz_localize(None)
    seal = pd.Timestamp(SEAL_START).tz_localize(None)
    explorable = int((stamps < seal).sum())

    assert explorable == 0, (
        f"{explorable:,} of {len(stamps):,} 15m rows are now BEFORE the seal. "
        "That would be new -- a longer intraday history has appeared from "
        "somewhere -- and the 15m timeframe would stop being structurally "
        "unmeasurable. Re-enable the assertion above and re-measure.")
    assert stamps.max() > seal, "the 15m export is stale rather than sealed"
