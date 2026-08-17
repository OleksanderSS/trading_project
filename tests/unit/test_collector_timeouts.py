"""One timeout for every collector is one timeout for none of them.

On 2026-08-17 the daily history window went from two years to thirty. The
consequence was not that yahoo took longer. TEN collectors were cancelled at
the same instant -- exactly 300 seconds after collection began -- because they
share an event loop and one slow member starves the rest:

    yahoo_finance  cftc  fear_greed  fred  insider
    newsapi  reddit_sentiment  sdmx_macro  sec_filings  wikimedia_attention

None of them saved anything and the pipeline reported success. The log said
"Successfully downloaded 7541 rows for AAPL/1d" and market_data_raw did not
gain a single row, because YFCollector downloads everything, then filters,
then upserts once at the end -- so a cancellation two thirds of the way in
discards every row already on the machine.

This file pins the table that replaced the hardcoded 300. It cannot prove a
timeout is generous enough for a future network, and does not try; it proves
the number is DECLARED PER COLLECTOR rather than standing in for all of them,
which is the defect that actually recurred.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.stages.collection.orchestrator import CollectionStage  # noqa: E402


class FakeCollector:
    def __init__(self, configs: dict | None = None):
        self.configs = configs or {}


def timeout_for(name: str, configs: dict | None = None) -> int:
    return CollectionStage._collector_timeout(FakeCollector(configs), name)


class TestTheTableIsRealPerCollector:
    def test_the_slowest_collector_gets_the_longest_leash(self):
        # The 30-year download alone ran 456s before dedup began.
        assert timeout_for('yahoo_finance') >= 1800

    def test_yahoo_gets_more_than_the_default(self):
        assert timeout_for('yahoo_finance') > timeout_for('anything_unlisted')

    @pytest.mark.parametrize('name', [
        'yahoo_finance', 'huggingface', 'free_google_trends',
        'sec_filings', 'insider', 'wikimedia_attention',
    ])
    def test_each_named_collector_has_its_own_number(self, name):
        assert name in CollectionStage._COLLECTOR_TIMEOUT_SECONDS
        assert CollectionStage._COLLECTOR_TIMEOUT_SECONDS[name] > 0

    def test_an_unlisted_collector_still_gets_a_bound(self):
        # A missing entry must not mean "no timeout" — a collector that hangs
        # forever is worse than one that is cancelled.
        assert timeout_for('a_collector_written_next_year') > 0


class TestTheOldFailureCannotSilentlyReturn:
    def test_nothing_is_left_at_the_old_three_hundred(self):
        """300s was chosen once, for everything. That is the defect."""
        assert CollectionStage._DEFAULT_COLLECTOR_TIMEOUT_SECONDS > 300
        assert all(v > 300 for v in CollectionStage._COLLECTOR_TIMEOUT_SECONDS.values())

    def test_every_collector_cancelled_on_2026_08_17_now_has_more_room(self):
        cancelled = ['yahoo_finance', 'cftc', 'fear_greed', 'fred', 'insider',
                     'newsapi', 'reddit_sentiment', 'sdmx_macro', 'sec_filings',
                     'wikimedia_attention']
        for name in cancelled:
            assert timeout_for(name) > 300, f'{name} would still die at 300s'


class TestConfigCanOverride:
    def test_a_configured_value_wins(self):
        assert timeout_for('yahoo_finance', {'collector_timeout_seconds': 42}) == 42

    @pytest.mark.parametrize('bad', [0, -5, None, 'soon', ''])
    def test_a_nonsense_override_falls_back_rather_than_disabling_the_bound(self, bad):
        # A zero or negative timeout would cancel instantly; a string would
        # raise inside wait_for. Both must fall back to the declared number.
        assert timeout_for('yahoo_finance', {'collector_timeout_seconds': bad}) == 1800

    def test_a_collector_without_a_configs_attribute_does_not_crash(self):
        class Bare:
            pass
        assert CollectionStage._collector_timeout(Bare(), 'yahoo_finance') == 1800
