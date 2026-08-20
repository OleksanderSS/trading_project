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


class TestTheRankIsPreferredAndCannotRot:
    """An absolute threshold against a feature whose scale is a config choice.

    `context_velocity` is the share of recent bars whose fingerprint changed,
    so its scale is set by `fingerprint_columns` — a property of our config,
    not of the market. The shipped 0.85 was chosen when fingerprints were
    nearly unique. Measured 2026-08-20 it is exceeded on 64% of 15m bars, 65%
    of 60m and 82% of daily.

    A percentile of the ticker's own history cannot drift that way: 0.90 is the
    busiest decile by construction, whatever the fingerprint width becomes.
    """

    def rank_signal(self, rank, confidence=1.0, prediction=0.02):
        return {'ticker': 'AAA', 'confidence': confidence,
                'raw_forecast': prediction, 'context_velocity_rank': rank,
                'context_velocity': 0.99}          # absolute would block

    def test_the_rank_wins_over_absolute_velocity(self, stage):
        # Absolute velocity 0.99 would block; a calm rank must not.
        out = stage._apply_context_rules([self.rank_signal(0.10)])
        assert out[0]['confidence'] == 1.0

    def test_a_busy_rank_still_blocks(self, stage):
        out = stage._apply_context_rules([self.rank_signal(0.95)])
        assert out[0]['confidence'] == 0.0

    def test_rank_thresholds_are_fire_rates_by_construction(self, stage):
        """0.90 must block about a tenth of a uniform rank distribution."""
        signals = [self.rank_signal(i / 100) for i in range(100)]
        out = stage._apply_context_rules(signals)
        blocked = sum(1 for s in out if s['confidence'] == 0.0)
        assert 8 <= blocked <= 12, f'blocked {blocked} of 100, expected ~10'

    def test_absolute_velocity_is_still_used_when_no_rank_exists(self, stage):
        # Batches built before the rank column existed must keep working.
        out = stage._apply_context_rules([signal(velocity=0.95)])
        assert out[0]['confidence'] == 0.0
