"""NewsAPI gives 100 requests per 24 hours. The collector used to ask for 1,656.

One run of 2026-08-21 built one query per ticker and per keyword -- 552 terms
-- fired all of them at once through `asyncio.gather`, retried each up to
three times, and brought back zero articles against 1,129 rate limits. The
first hundred requests spend the whole day's allowance; refused requests are
counted too, so the retries make the *next* day worse rather than rescuing
this one.

The key was fine the entire time. Asked directly, the service answers
`code: rateLimited` with the quota spelled out in the message.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.data.collectors.newsapi_collector import (
    NewsApiQuotaExhausted,
    NewsAPICollector,
)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    instance = NewsAPICollector.__new__(NewsAPICollector)
    instance.configs = {}
    instance.daily_budget = 10
    instance._budget_file = tmp_path / "budget.json"
    instance._api_key = "test-key"

    class _Logger:
        def __init__(self):
            self.lines = []

        def _record(self, msg, *args):
            self.lines.append(msg % args if args else msg)

        info = warning = error = _record

    instance.logger = _Logger()
    return instance


def test_a_fresh_day_has_the_whole_allowance(collector):
    assert collector._remaining_requests() == 10


def test_spending_is_remembered_across_runs(collector):
    """A budget that resets every run is how the quota got spent three times a day."""
    collector._write_budget(spent=7, exhausted=False)
    assert collector._remaining_requests() == 3


def test_yesterdays_spending_does_not_count(collector):
    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    collector._budget_file.write_text(
        json.dumps({"date": yesterday, "spent": 10, "exhausted": True}), encoding="utf-8"
    )
    assert collector._remaining_requests() == 10


def test_an_exhausted_day_offers_nothing_even_below_the_count(collector):
    """The service refused before the budget ran out -- believe the service."""
    collector._write_budget(spent=4, exhausted=True)
    assert collector._remaining_requests() == 0


def test_tickers_are_queried_before_generic_keywords():
    """Ninety requests against five hundred terms: the order decides the day."""
    ordered = NewsAPICollector._prioritise(
        tickers=["AAPL", "NVDA"], keywords=["inflation", "AAPL", "container"]
    )
    assert ordered[:2] == ["AAPL", "NVDA"]
    assert ordered.count("AAPL") == 1
    assert "container" in ordered


def test_ordering_is_stable_not_set_iteration():
    first = NewsAPICollector._prioritise(["MSFT", "KO"], ["gdp", "oil"])
    second = NewsAPICollector._prioritise(["MSFT", "KO"], ["gdp", "oil"])
    assert first == second == ["MSFT", "KO", "gdp", "oil"]


def test_an_unwritable_budget_is_reported_not_swallowed(collector, tmp_path):
    collector._budget_file = tmp_path / "no" / "such" / "dir" / "x.json"
    collector._budget_file.parent.mkdir(parents=True)
    collector._budget_file.parent.chmod(0o500)
    try:
        collector._write_budget(spent=3, exhausted=False)
    finally:
        collector._budget_file.parent.chmod(0o700)
    # On filesystems that ignore the mode this simply succeeds; the contract
    # under test is that failure is never silent.
    assert collector._budget_file.exists() or any(
        "Could not record request budget" in line for line in collector.logger.lines
    )


def test_the_quota_signal_is_its_own_exception():
    """It must not be mistaken for a transient error and retried."""
    assert issubclass(NewsApiQuotaExhausted, RuntimeError)
    assert not issubclass(NewsApiQuotaExhausted, (ValueError, TypeError, KeyError))


# --------------------------------------------------------------- the run loop


def _runnable(collector, monkeypatch, fail_after=None):
    """Wire up run() with the network and database replaced."""
    collector.configs = {"table_name": "newsapi_articles"}
    collector.hash_keys = ["url", "publishedAt"]
    collector.cache_manager = None
    collector.asked: list[str] = []

    async def _fetch(term, api_key):
        collector.asked.append(term)
        if fail_after is not None and len(collector.asked) > fail_after:
            raise NewsApiQuotaExhausted("rateLimited: allowance spent")
        return [{"url": f"http://x/{term}", "publishedAt": "2026-08-21", "title": term}]

    monkeypatch.setattr(collector, "_fetch_for_term", _fetch)
    monkeypatch.setattr(collector, "_check_newsapi_cache", lambda *a, **k: None)
    monkeypatch.setattr(collector, "_update_cache", lambda *a, **k: None)

    class _DB:
        def filter_new_records(self, table, df):
            return df

        def upsert(self, table, df, unique_on=None):
            pass

    collector.db_manager = _DB()
    return collector


@pytest.mark.asyncio
async def test_the_run_stops_asking_once_the_service_says_stop(collector, monkeypatch):
    """552 terms used to be in flight before the first refusal came back."""
    _runnable(collector, monkeypatch, fail_after=3)
    terms = [f"T{i}" for i in range(50)]

    await collector.run(tickers=terms)

    assert len(collector.asked) == 4          # three good, one refusal, then stop
    assert collector._remaining_requests() == 0


@pytest.mark.asyncio
async def test_a_run_never_asks_for_more_than_the_allowance(collector, monkeypatch):
    _runnable(collector, monkeypatch)
    await collector.run(tickers=[f"T{i}" for i in range(500)])

    assert len(collector.asked) == 10         # daily_budget in the fixture
    assert collector._remaining_requests() == 0


@pytest.mark.asyncio
async def test_the_next_run_that_day_issues_no_request_at_all(collector, monkeypatch):
    _runnable(collector, monkeypatch)
    collector._write_budget(spent=10, exhausted=False)

    result = await collector.run(tickers=["AAPL", "NVDA"])

    assert result is None
    assert collector.asked == []              # refusals count against the quota too
    assert any("already spent" in line for line in collector.logger.lines)


@pytest.mark.asyncio
async def test_a_partial_day_spends_only_what_is_left(collector, monkeypatch):
    _runnable(collector, monkeypatch)
    collector._write_budget(spent=7, exhausted=False)

    await collector.run(tickers=[f"T{i}" for i in range(20)])

    assert len(collector.asked) == 3


def test_the_configs_own_key_name_is_honoured(monkeypatch):
    """collectors.yaml writes `api_key_env`; the code read `api_key_name`.

    It worked only because the default happened to equal the configured value,
    so pointing the config at a different variable did nothing at all.
    """
    monkeypatch.setenv("SOME_OTHER_NEWS_KEY", "from-config")
    monkeypatch.delenv("NEWS_API_KEY", raising=False)

    built = NewsAPICollector(
        configs={"api_key_env": "SOME_OTHER_NEWS_KEY"},
        http_client_factory=None,
        db_manager=None,
    )
    assert built._get_api_key() == "from-config"


def test_a_missing_key_is_still_reported_as_missing(monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    built = NewsAPICollector(configs={}, http_client_factory=None, db_manager=None)
    assert built._get_api_key() is None
