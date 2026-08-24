"""What stage 3 holds for its whole duration, measured rather than guessed.

The peak of stage 3 is 2.67 GiB and the feature frames account for under half
of it: 15m is 29,097 x 1,386 and daily 154,069 x 466, well under a gigabyte
together. The rest is the input tables, held from the first enricher to the
last -- wikipedia_attention_data at 596,444 rows, market_data_raw at 441,839,
fred_data at 140,733. Nothing measured them, so which were worth holding could
only be argued about.

The alias is why this needs care. Stage 3 hands out `cftc` and `cftc_data` as
the SAME object, created with `setdefault(key[:-5], value)`. A scan of which
enrichers read which key suggested 15 of 25 inputs were unused; most were
aliases of frames that are read, and acting on that number would have argued
for deleting live inputs. Counting keys overstates what is held; counting
objects does not.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.stages.feature_engineering.orchestrator import (
    FeatureEngineeringStage,
)


def test_aliases_of_one_frame_are_counted_once():
    """`cftc` and `cftc_data` are one object under two names, not two frames."""
    shared = pd.DataFrame({"x": range(1000)})
    other = pd.DataFrame({"y": range(50)})

    groups = FeatureEngineeringStage._distinct_frames({
        "cftc": shared,
        "cftc_data": shared,
        "news": other,
    })

    assert len(groups) == 2, "the alias was counted as a second frame"
    by_size = {len(frame): sorted(names) for names, frame in groups.values()}
    assert by_size[1000] == ["cftc", "cftc_data"]
    assert by_size[50] == ["news"]


def test_equal_but_separate_frames_are_counted_separately():
    """Identity, not equality: two copies of the same data cost twice."""
    first = pd.DataFrame({"x": range(100)})
    second = pd.DataFrame({"x": range(100)})

    groups = FeatureEngineeringStage._distinct_frames({
        "market_data": first,
        "market_data_raw": second,
    })
    assert len(groups) == 2, (
        "two separate objects holding equal data were merged; that would hide "
        "exactly the duplicate this measurement exists to find"
    )


def test_non_frames_are_ignored():
    groups = FeatureEngineeringStage._distinct_frames({
        "frame": pd.DataFrame({"x": [1]}),
        "flag": True,
        "name": "offline_only",
        "nothing": None,
    })
    assert len(groups) == 1


def test_the_report_survives_a_frame_that_cannot_be_sized(caplog):
    """Instrumentation must never be the thing that ends a five-hour run."""
    class _Awkward(pd.DataFrame):
        @property
        def _constructor(self):
            return _Awkward

        def memory_usage(self, *args, **kwargs):
            raise RuntimeError("no")

    stage = FeatureEngineeringStage.__new__(FeatureEngineeringStage)
    stage.logger = logging.getLogger("probe")

    with caplog.at_level(logging.INFO):
        stage._log_enrichment_input_cost("1d", {"awkward": _Awkward({"x": [1, 2]})})

    assert any("Enrichment inputs for 1d" in record.message
               for record in caplog.records)
