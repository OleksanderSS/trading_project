"""A gate meant for rare conditions must not fire on most bars.

Stage 6 applies context rules before its execution boundary: cut exposure when
`context_velocity` is high, block buys when it is critical. That is context
used as a GATE rather than as a feature — the cheap channel, asserted from
reasoning instead of learned from examples.

Measured on the real batch, 2026-08-20, with the thresholds hardcoded at 0.7
and 0.85:

    context_velocity_15m > 0.70   81.29% of bars
    context_velocity_15m > 0.85   64.02% of bars

So "CRITICAL ANXIETY, block the buy" was the default state, not an emergency.
And the column is entirely empty on 60m and 1d, so the gate cannot fire at all
on the timeframes the daily work uses.

Same defect as news_significance_level, whose 0.8/0.3 thresholds were set
against scores that spanned 0.058 to 0.102: a number calibrated to a scale the
feature never produced. These tests pin the behaviour AND the reporting that
makes a rotted threshold announce itself.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.stages.trading.orchestrator import TradingExecutionStage  # noqa: E402


@pytest.fixture
def stage():
    s = TradingExecutionStage.__new__(TradingExecutionStage)
    s.logger = logging.getLogger('test-gate')
    return s


def signal(velocity=None, confidence=1.0, prediction=0.02, ticker='AAA'):
    d = {'ticker': ticker, 'confidence': confidence, 'raw_forecast': prediction}
    if velocity is not None:
        d['context_velocity'] = velocity
    return d


class TestTheRulesStillWork:
    def test_calm_context_passes_untouched(self, stage):
        out = stage._apply_context_rules([signal(velocity=0.1)])
        assert out[0]['confidence'] == 1.0

    def test_high_velocity_halves_exposure(self, stage):
        out = stage._apply_context_rules([signal(velocity=0.75)])
        assert out[0]['confidence'] == pytest.approx(0.5)

    def test_critical_velocity_blocks_a_buy(self, stage):
        out = stage._apply_context_rules([signal(velocity=0.9, prediction=0.02)])
        assert out[0]['confidence'] == 0.0

    def test_critical_velocity_does_not_block_a_sell(self, stage):
        # The rule is about not buying into a panic, not about freezing exits.
        out = stage._apply_context_rules([signal(velocity=0.9, prediction=-0.02)])
        assert out[0]['confidence'] > 0.0


class TestAnAbsentReadingIsNotACalmOne:
    def test_a_signal_without_velocity_passes_through(self, stage):
        # The column does not exist on 60m or 1d. Treating absence as calm
        # would arm nothing; treating it as panic would mute everything. Both
        # are silent, so it passes through and the run says how often.
        out = stage._apply_context_rules([signal(velocity=None)])
        assert out[0]['confidence'] == 1.0

    def test_a_non_numeric_velocity_does_not_crash_or_gate(self, stage):
        s = signal(); s['context_velocity'] = 'unknown'
        assert stage._apply_context_rules([s])[0]['confidence'] == 1.0

    def test_it_warns_when_no_signal_has_a_reading(self, stage, caplog):
        with caplog.at_level(logging.WARNING):
            stage._apply_context_rules([signal(velocity=None) for _ in range(5)])
        assert any('NO velocity readings' in r.message for r in caplog.records)


class TestARottedThresholdAnnouncesItself:
    def test_it_warns_when_the_block_rule_fires_on_most_signals(self, stage, caplog):
        # The shipped 0.85 blocked 64% of real bars. That must be loud.
        with caplog.at_level(logging.WARNING):
            stage._apply_context_rules([signal(velocity=0.95) for _ in range(20)])
        assert any("rule 'block' fired" in r.message for r in caplog.records)

    def test_it_stays_quiet_when_the_gate_is_genuinely_rare(self, stage, caplog):
        signals = [signal(velocity=0.1) for _ in range(19)] + [signal(velocity=0.95)]
        with caplog.at_level(logging.WARNING):
            stage._apply_context_rules(signals)
        assert not any('fired on' in r.message for r in caplog.records)

    def test_thresholds_are_configurable_rather_than_hardcoded(self, stage):
        stage.context_gate_config = {'reduce_velocity': 0.95, 'block_velocity': 0.99}
        out = stage._apply_context_rules([signal(velocity=0.9)])
        assert out[0]['confidence'] == 1.0, 'a raised threshold must actually raise it'
