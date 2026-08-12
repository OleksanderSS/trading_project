"""Predicting tomorrow's moving average is arithmetic, not forecasting.

`target_sma_20_f1` asks for the 20-period SMA one bar ahead. Nineteen of its
twenty terms are known today, which is why persistence alone scores R2
0.998-0.9994 across this family. A model that beats persistence there earns
~0.999 and lands in the same champion table as a directional model at 0.55
balanced accuracy — so any ranking that compares champions across targets
picks the arithmetic every time.

On the 2026-08-12 run, 12 of 65 champions sat here: volume_ratio_f1 (10) and
macd_hist_f1 (2). Not one of them names a price move anyone can trade.

They are still trained, still scored, still written to the holdout artifact.
The family stays available as evidence and as a feature source. It just does
not reach Stage 5 as something to act on.
"""
import pytest

from src.config.target_type_registry import load_target_types
from src.pipeline.stages.stage_4_modeling import ModelingStage


@pytest.mark.parametrize(
    "target",
    [
        "target_volume_ratio_f1",
        "target_macd_hist_f1",
        "target_sma_20_f1",
        "target_ema_20_f1",
        "target_rsi_14_f1",
        "target_atr_14_f5",
        "target_bb_upper_f1",
    ],
)
def test_every_indicator_target_is_refused_promotion(target):
    assert ModelingStage._is_indicator_prediction(target) is True


@pytest.mark.parametrize(
    "target",
    [
        "target_return_1d",
        "target_up_1d",
        "target_volatility_spike_15m",
        "target_hourly_volume_spike_1h",
        "target_hourly_breakout_1h",
        "target_weekly_up_1w",
    ],
)
def test_tradeable_targets_are_untouched(target):
    """The 53 champions that were not indicator predictions must still pass."""
    assert ModelingStage._is_indicator_prediction(target) is False


def test_the_filter_matches_the_registry_rather_than_a_name_pattern():
    """A `_f1` suffix is a convention; the registry is the declaration.

    Filtering on the suffix would be a second copy of a fact targets.yaml
    already states, and would miss any indicator target named differently.
    """
    declared = {
        name for name, kind in load_target_types().items()
        if kind == "indicator_prediction"
    }
    assert declared, "targets.yaml must declare the indicator family"

    for name in declared:
        assert ModelingStage._is_indicator_prediction(name) is True

    for name, kind in load_target_types().items():
        if kind != "indicator_prediction":
            assert ModelingStage._is_indicator_prediction(name) is False


def test_an_unknown_target_is_promoted_rather_than_refused():
    """Refusing what is not declared would silently drop new targets."""
    assert ModelingStage._is_indicator_prediction("target_invented_today") is False
    assert ModelingStage._is_indicator_prediction("") is False
    assert ModelingStage._is_indicator_prediction(None) is False
