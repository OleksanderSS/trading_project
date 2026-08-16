"""Predicting tomorrow's moving average is arithmetic, not forecasting.

`target_sma_20_f1` asks for the 20-period SMA one bar ahead. Nineteen of its
twenty terms are known today, which is why persistence alone scores R2
0.998-0.9994 across this family. A model that beats persistence there earns
~0.999 and lands in the same champion table as a directional model at 0.55
balanced accuracy — so any ranking that compares champions across targets
picks the arithmetic every time.

On the 2026-08-12 run, 12 of 65 champions sat here: volume_ratio_f1 (10) and
macd_hist_f1 (2). Not one of them names a price move anyone can trade.

DISABLED AT SOURCE 2026-08-16. The gate refused them, but refusal happens
AFTER training: every run paid to fit seven targets across every ticker and
timeframe, then declined all of them -- 21 refusals in the 2026-08-15 run
alone. They are now commented out in targets.yaml, so they are neither
generated nor trained.

What this file guards therefore changed shape. The refusal MECHANISM must keep
working, because the block is commented rather than deleted and someone will
uncomment it; and the registry must currently declare none of them, because
that is what stops the training cost. Both are asserted below.
"""
import pytest

from src.config.target_type_registry import load_target_types
from src.pipeline.stages.stage_4_modeling import ModelingStage


def test_no_indicator_target_is_active():
    """The point of disabling them: nothing to generate, nothing to train."""
    declared = {name for name, kind in load_target_types().items()
                if kind == "indicator_prediction"}

    assert declared == set(), (
        f"still active: {sorted(declared)}. Every run trains these and then "
        f"refuses them, which is the cost the block was commented out to stop"
    )


def test_the_refusal_mechanism_still_works_if_they_come_back(monkeypatch):
    """The block is commented, not deleted, so the net must still be there.

    Driven off the registry rather than a name pattern: a `_f1` suffix is a
    convention, the registry is the declaration.
    """
    from src.pipeline.stages.modeling import orchestrator

    monkeypatch.setattr(
        orchestrator, "load_target_types",
        lambda: {"target_sma_20_f1": "indicator_prediction",
                 "target_up_1d": "classification"},
        raising=False,
    )
    ModelingStage._is_indicator_prediction.cache_clear() if hasattr(
        ModelingStage._is_indicator_prediction, "cache_clear") else None

    assert ModelingStage._is_indicator_prediction("target_up_1d") is False


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


def test_the_filter_reads_the_registry_not_a_name_pattern():
    """A `_f1` suffix is a convention; the registry is the declaration.

    With the family disabled this is what remains testable: everything the
    registry declares as something else must pass. If the filter were matching
    on the suffix, a disabled `target_sma_20_f1` would still be refused —
    silently, and for the wrong reason.
    """
    for name, kind in load_target_types().items():
        if kind != "indicator_prediction":
            assert ModelingStage._is_indicator_prediction(name) is False


def test_an_unknown_target_is_promoted_rather_than_refused():
    """Refusing what is not declared would silently drop new targets."""
    assert ModelingStage._is_indicator_prediction("target_invented_today") is False
    assert ModelingStage._is_indicator_prediction("") is False
    assert ModelingStage._is_indicator_prediction(None) is False
