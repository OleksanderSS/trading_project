"""A config block that nothing reaches looks exactly like one that works.

This is #165's shape and it has now cost this project twice. PredictionAdjuster
asks for `analysis.prediction_adjustment`, which exists in no file, while
strategy.yaml declares `context_prediction_adjustment` in a different shape --
so the component has logged "initialized with no rules" on every run for
months while looking like a configured layer.

The second time was mine, on 2026-09-08, wiring PostInferenceFilter to
`strategy.post_inference_filter`. strategy.yaml has no top-level `strategy:`
key -- its blocks ARE the top level, which is why VirtualPortfolio reads
`backtesting.transaction_costs` and not `strategy.backtesting.transaction_costs`
-- so the path returned {} and the filter kept using its code defaults.

WHAT MAKES THIS INVISIBLE, and why comparing values cannot catch it: an
unreachable block falls through to the same defaults the code already had, so
every value matches and every test comparing values passes. The only thing that
distinguishes a wired block from an unwired one is whether the block comes back
NON-EMPTY. That is what these assert.
"""
from __future__ import annotations

import pytest

from src.config.unified_config_manager import UnifiedConfigManager

#: (config path, keys the reader expects). Each entry is a block some component
#: reads by path; the test proves the path resolves to something.
WIRED_BLOCKS = {
    "post_inference_filter": {
        "macro_weight", "rsi_weight", "sentiment_weight", "chaos_weight",
        "min_confidence", "max_confidence"},
    # Read off the resolved block rather than guessed: my first version named
    # `commission_per_share`, which this block does not carry, and the test
    # failed on my expectation instead of on the config. `market_impact_y` is
    # here because it replaced the market_impact_coefficient constant on
    # 2026-09-08 (R66) and must stay reachable.
    "backtesting.transaction_costs": {
        "commission_pct", "spread_pct", "slippage_pct", "market_impact_y"},
    "strategy.risk_management": {"max_position_size_pct", "max_total_risk_pct"},
    "strategy.risk_management.position_sizer": {
        "max_position_size_pct", "kelly_fraction", "max_active_positions"},
}


@pytest.fixture(scope="module")
def manager():
    return UnifiedConfigManager()


@pytest.mark.parametrize("path,expected", sorted(WIRED_BLOCKS.items()))
def test_the_block_a_component_reads_by_path_is_not_empty(manager, path, expected):
    block = manager.get(path, {})
    assert block, (
        f"{path!r} resolves to nothing. Whatever reads it is running on its "
        "own code defaults, and every value will still look correct -- an "
        "unreachable block falls through to exactly those defaults. This is "
        "#165's shape.")
    missing = expected - set(block)
    assert not missing, (
        f"{path!r} resolves, but without {sorted(missing)}. A block that is "
        "reached and incomplete is the same defect one level down: the reader "
        "falls back to code for the keys that are absent.")


def test_the_filter_weights_sum_to_one(manager):
    """Load-bearing, not decorative.

    A missing input column makes PostInferenceFilter use a multiplier of 1.0
    for that term, so a missing column contributes its weight NEUTRALLY only
    while the weights sum to one. Change one without changing another and a
    missing column starts pushing confidence up or down.
    """
    block = manager.get("post_inference_filter", {})
    total = sum(value for key, value in block.items()
                if key.endswith("_weight"))
    assert total == pytest.approx(1.0), (
        f"the filter weights sum to {total}, not 1.0. A missing input column "
        "is no longer neutral.")


def test_declaring_the_block_changed_no_behaviour(manager):
    """The values were declared at what was already acting, and this says so."""
    from src.trading.post_inference_filter import PostInferenceFilter

    declared = PostInferenceFilter(
        config=manager.get("post_inference_filter", {})).params
    from_code = PostInferenceFilter().params
    assert declared == from_code, (
        f"declared {declared} differs from the code defaults {from_code}. That "
        "may be intended, but it is a change in what the filter DOES and may "
        "not arrive as a visibility fix.")
