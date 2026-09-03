"""The persistence opponent must lag by how far the target REACHES, not by its name.

REGISTER #191 made the gate's persistence baseline lag by the target's horizon
instead of one bar, because `y[t-1]` for a 5-bar target is not knowable at `t`
and an opponent handed unknowable data is an oracle, not a baseline.

That fix took the horizon from `target_horizon_bars`, which reads the target's
NAME. For a windowed target the name understates the reach:
`target_daily_trend_strength_1d` is `shift: -1, window: 20` and looks twenty
bars ahead while its suffix says one. `walk_forward_validation` already knew
this -- `_get_target_horizon_rows` exists precisely because the purge gap was
"19 rows too narrow" for that target -- but the gate did not use it.

Measured 2026-09-03 on the batch, the same target lagged both ways:

    target_daily_trend_strength_1d   R2      0.8973  ->  -1.0213
    target_hourly_breakout_1h        BalAcc  0.8573  ->   0.5406
    target_daily_momentum_score_1d   R2      0.7579  ->  -1.0020

The left column is what the gate applied. It is not a fact about those
targets: at their own horizon they are as unpredictable by lag as any return
-- -1.02 is what a lag scores on a series it cannot predict, by construction,
since Var(y - y_lag) is twice the variance. It is a fact about the opponent.

A model on `target_daily_trend_strength_1d` had to beat R2 0.897 to be
promoted. No honest forecast does that, so that target could never produce a
champion and its refusals said nothing. The damage is false REFUSALS, which is
the direction that makes every recorded refusal unreadable -- the thing
CLAIMS R10 exists to rule out.

Nine of the eighteen targets in the batch have the two horizons disagreeing.
The widest is `target_hourly_volume_spike_1h`: name 1, window 23.
"""
from __future__ import annotations

import pytest

from src.pipeline.stages.modeling.walk_forward_validation import (
    _get_target_horizon_rows,
)
from src.targets.timeframe_contract import target_horizon_bars

#: Measured from `targets.yaml` on 2026-09-03. These are the targets whose
#: name and reach disagree, with the reach that must win.
WINDOWED = {
    "target_daily_trend_strength_1d": 20,
    "target_daily_momentum_score_1d": 10,
    "target_hourly_volume_spike_1h": 23,
    "target_hourly_breakout_1h": 4,
}


@pytest.mark.parametrize("target,reach", sorted(WINDOWED.items()))
def test_the_window_aware_horizon_is_bigger_than_the_name(target, reach):
    """If these ever agree, either the target changed or a horizon source
    regressed -- both need looking at, and silence would hide either."""
    windowed = _get_target_horizon_rows(target)
    assert windowed == reach, (
        f"{target} now reaches {windowed} bars, not {reach}. If the target "
        f"was redefined, update this table; if the policy manager stopped "
        f"counting the window, the gate is back to an oracle opponent."
    )
    named = target_horizon_bars(target, "60m" if "1h" in target else "1d") or 1
    assert windowed > named, (
        f"{target}: the name says {named} and the window says {windowed}; "
        f"this test exists for targets where they differ"
    )


def test_the_gate_resolves_the_larger_of_the_two():
    """The gate's own resolution, read from its source.

    A behavioural test would need a fitted holdout and a full data dict; what
    must not regress is the CHOICE, and that is one expression.
    """
    import inspect

    from src.training.base_trainer import BaseTrainer

    source = inspect.getsource(BaseTrainer._score_naive_baselines)
    assert "_get_target_horizon_rows" in source, (
        "the gate is back to reading the horizon out of the target's name, so "
        "for every windowed target its persistence opponent is handed a value "
        "nobody could know at forecast time"
    )
    assert "max(int(named), int(windowed))" in source, (
        "the gate no longer takes the LARGER of the two horizons; taking the "
        "smaller is the oracle, and taking only one source means whichever it "
        "is will be wrong for the other kind of target"
    )


def test_targets_whose_name_carries_their_whole_reach_are_untouched():
    """The fix must not lengthen the lag where the name was already right --
    a lag longer than the reach weakens the opponent for no reason."""
    for target, timeframe in (
        ("target_return_1d", "1d"),
        ("target_return_5d", "1d"),
        ("target_up_1d", "1d"),
        ("target_up_5d", "1d"),
    ):
        named = target_horizon_bars(target, timeframe) or 1
        windowed = _get_target_horizon_rows(target) or named
        assert max(named, windowed) == named, (
            f"{target}: the window-aware horizon ({windowed}) exceeds the "
            f"name's ({named}) for a target whose name already states its "
            f"whole reach"
        )
