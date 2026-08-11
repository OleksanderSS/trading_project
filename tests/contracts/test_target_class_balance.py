"""A classification target must be rare enough to be interesting and common
enough to be measured.

Stage 4 splits roughly 20% of each context into a holdout, which is ~103 rows
on a daily context, ~174 on 60m and ~220 on 15m. A target whose positive rate
is 1% therefore puts an EXPECTED two positives in the whole holdout, and often
zero. That is what produced

    Champion NOT promoted ...: holdout score 0.0000 does not beat the naive
    baseline 0.0000

on every volatility_spike context: F1 is zero for the model and zero for the
baseline because there is nothing to find. The promotion gate refuses
correctly, but the fault is in the target, not the model — no algorithm can be
evaluated against an event that does not occur in the evaluation window.

Measured on the 2026-08-11 batch:

    target_volatility_spike_1h      155 / 16,079   0.96%
    target_volatility_spike_15m     242 / 23,368   1.04%
    target_volatility_spike_1d      232 / 11,001   2.11%
    target_hourly_volume_spike_1h  1,487 / 35,812  4.15%
    target_hourly_breakout_1h      1,543 / 16,079  9.60%
    target_hourly_up_1h            7,710 / 36,602 21.06%
    ... up_* targets 21-38%

This is a RATCHET: the three known-degenerate targets are listed so the suite
is not permanently red, and anything NEW that falls below the floor fails.
Shrink the allowlist as those targets are redefined or retired.
"""
from pathlib import Path

import pandas as pd
import pytest

TARGETS = Path('data/colab/accumulated/main_database/targets.parquet')

#: A floor for "definitely unmeasurable", not a certificate of health. Below
#: 2% a holdout of the size this pipeline produces contains one or two
#: positives at best. Clearing 2% does NOT make a target comfortable —
#: volatility_spike_1d sits at 2.11%, which is still only ~2 expected
#: positives in a ~103-row daily holdout. Judging that properly needs expected
#: positives per timeframe rather than one global rate; this catches the
#: unambiguous cases without pretending to more precision than it has.
MIN_POSITIVE_RATE = 0.02

#: Known degenerate today. Each needs its definition changed (a lower spike
#: threshold) or the target retired — not a modelling fix.
#:
#: volatility_spike_1d is deliberately NOT listed: at 2.11% it clears the
#: floor, and the ratchet below caught it the moment it was allowlisted out of
#: habit. An allowlist that quietly covers a target which no longer needs it is
#: how the next real one gets hidden.
KNOWN_DEGENERATE = {
    'target_volatility_spike_1h',
    'target_volatility_spike_15m',
}


def _binary_targets():
    if not TARGETS.exists():
        pytest.skip(f"no targets artifact at {TARGETS}")
    frame = pd.read_parquet(TARGETS)
    out = {}
    for column in frame.columns:
        if not column.startswith('target_'):
            continue
        series = frame[column].dropna()
        if series.empty or not series.isin([0, 1]).all():
            continue
        out[column] = series
    return out


def test_no_new_target_is_too_rare_to_evaluate():
    targets = _binary_targets()
    if not targets:
        pytest.skip("no binary targets in the batch")

    too_rare = {}
    for name, series in targets.items():
        rate = float(series.mean())
        if rate < MIN_POSITIVE_RATE and name not in KNOWN_DEGENERATE:
            too_rare[name] = f"{rate:.2%} ({int(series.sum())} of {len(series)})"

    assert not too_rare, (
        "these targets cannot be measured on the holdout sizes this pipeline "
        f"produces: {too_rare}. Fewer than {MIN_POSITIVE_RATE:.0%} positives "
        "means an evaluation window with almost no positive examples, so both "
        "the model and the naive baseline score zero and the comparison says "
        "nothing. Redefine the target or retire it."
    )


def test_the_degenerate_allowlist_does_not_outlive_its_targets():
    """Shrink the allowlist when a target is fixed, so it cannot hide a fresh one."""
    targets = _binary_targets()
    if not targets:
        pytest.skip("no binary targets in the batch")

    stale = {
        name for name in KNOWN_DEGENERATE
        if name in targets and float(targets[name].mean()) >= MIN_POSITIVE_RATE
    }
    assert not stale, (
        f"{stale} now clear the {MIN_POSITIVE_RATE:.0%} floor — remove them "
        f"from KNOWN_DEGENERATE so the ratchet keeps its grip."
    )
