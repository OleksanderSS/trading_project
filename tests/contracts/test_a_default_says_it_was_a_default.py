""""The column is absent" and "the column says normal" must not be one answer.

REGISTER #182. The arena is called "Regime-Aware Training Arena" and its
regime axis carries one value: every champion key ends in `_normal`, because
`MARKET_REGIME` is deliberately off (5.4 hours of the twelve to rebuild, and
it failed its own out-of-sample sign check) and the batch holds zero
MARKET_REGIME columns.

The decision recorded there was option (a): keep the key constant on purpose,
but stop the fallback being silent. That was implemented -- the stage warns,
names the cost of enabling the feature, and points here.

Reading it on 2026-09-04 showed the warning's own condition was the same
defect one level down. `_latest_context_value(..., default='normal')` returns
the default when nothing is found, so the caller could not tell an ABSENT
column from one that genuinely says 'normal', and the check was
`if current_pattern == 'normal'`.

Today that is always the first case, so the message is true. The moment
MARKET_REGIME_FEATURES=1 is set it becomes the second, and the stage would
announce "has no MARKET_REGIME column" about data that had arrived -- a
warning firing on a legitimate state, which is how warnings get switched off.

This is the audit method's own invariant, in the code that enforces the
decision about it: a default must be accompanied by something saying it WAS a
default. The call now asks for None and decides from that.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage


def _value(frame: pd.DataFrame, default, timeframe="1d"):
    return ModelingStage._latest_context_value(
        frame, ("MARKET_REGIME", "market_regime", "regime"),
        default=default, timeframe=timeframe,
    )


def test_an_absent_column_returns_the_default_it_was_given():
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    assert _value(frame, None) is None
    assert _value(frame, "normal") == "normal"


def test_a_present_column_returns_its_own_last_value():
    frame = pd.DataFrame({
        "close": [1.0, 2.0, 3.0],
        "MARKET_REGIME_1d": ["high", "normal", "normal"],
    })
    assert _value(frame, None) == "normal"


def test_the_two_cases_are_distinguishable_when_the_default_is_none():
    """The whole point: with default='normal' these two return the same
    string and the caller cannot tell them apart."""
    absent = pd.DataFrame({"close": [1.0]})
    present = pd.DataFrame({"close": [1.0], "MARKET_REGIME_1d": ["normal"]})

    assert _value(absent, None) is None
    assert _value(present, None) == "normal"
    # And with the old default they collapse, which is the defect:
    assert _value(absent, "normal") == _value(present, "normal") == "normal"


def test_the_stage_asks_for_none_and_branches_on_that():
    source = inspect.getsource(ModelingStage)
    marker = source.index('("MARKET_REGIME", "market_regime", "regime")')
    window = source[marker - 400:marker + 600]
    assert "default=None" in window, (
        "the regime lookup asks for 'normal' again, so an absent column and a "
        "real 'normal' are one answer"
    )
    assert "if found_pattern is None:" in window, (
        "the warning branches on the VALUE rather than on whether anything was "
        "found, so it will fire on a legitimate 'normal' once the feature is "
        "enabled"
    )


def test_the_warning_still_names_the_cost_and_the_register_entry():
    """Option (a) was chosen with a condition attached; the message is where
    that condition lives."""
    source = inspect.getsource(ModelingStage)
    assert "MARKET_REGIME_FEATURES=1" in source, (
        "the warning no longer says how to turn the axis on"
    )
    assert "5.4h" in source, (
        "the cost is gone, so 'just enable it' reads as free"
    )
    assert "#182" in source, (
        "the message no longer points at the measurement behind the decision"
    )


@pytest.mark.parametrize("value", ["high", "low", "extreme"])
def test_a_real_regime_is_reported_not_warned_about(value):
    frame = pd.DataFrame({"close": [1.0], "MARKET_REGIME_1d": [value]})
    assert _value(frame, None) == value


def test_a_blank_label_is_not_mistaken_for_a_regime():
    """The helper stringifies whatever it finds, so an empty cell comes back
    as "" -- which is not None, and would key a champion by nothing."""
    frame = pd.DataFrame({"close": [1.0], "MARKET_REGIME_1d": ["   "]})
    assert _value(frame, None) == "   ", (
        "the helper changed shape; the guard in the stage was written for the "
        "value it actually returns"
    )

    source = inspect.getsource(ModelingStage)
    marker = source.index('("MARKET_REGIME", "market_regime", "regime")')
    window = source[marker:marker + 700]
    assert ".strip() or None" in window, (
        "a blank regime label slips past `is None` and becomes the champion "
        "key, silently, which is the same defect one character further on"
    )
