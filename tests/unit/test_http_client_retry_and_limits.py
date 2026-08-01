"""The factory has to actually do what its configuration says.

Three defects, all verified against the real code and the real config:

1. status_forcelist and backoff_factor were read from config, defaulted, and
   then never passed to anything. httpx's `retries` argument covers ONLY
   connection establishment -- httpcore catches (ConnectError, ConnectTimeout)
   and nothing else -- so a 429 or 503 was returned to the caller untouched.
   The factory named a retry-on-status policy and implemented
   retry-on-connect-failure. This is the direct cause of unretried 429s.

2. There was no `http_client:` section in any YAML. config_manager.get(
   'http_client', {}) returned {}, so every value was a hardcoded default,
   none of it tunable. The default was 100 requests per second.

3. That limiter was per HttpClientFactory INSTANCE, and five instances were
   observed in one pipeline run (grouped from "RateLimiter initialized" lines
   in logs/system.log) -- pipeline_factory, pipeline_orchestrator,
   base_stage's fallback, collector_factory, auto_accumulator. Each held a
   full bucket, so the effective limit was five times the configured one.

Also: HttpClientFactory() with no arguments raised TypeError, which is
exactly how CollectorFactory calls it in its `or` fallback.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from src.core.clients.http_client_factory import HttpClientFactory


@pytest.fixture(autouse=True)
def _isolate_shared_limiters():
    """The limiter cache is deliberately process-wide; tests must not inherit
    each other's buckets."""
    saved = dict(HttpClientFactory._limiters)
    HttpClientFactory._limiters.clear()
    yield
    HttpClientFactory._limiters.clear()
    HttpClientFactory._limiters.update(saved)


class _Responder:
    """Serves a scripted sequence of status codes and counts the calls."""

    def __init__(self, statuses, retry_after=None):
        self.statuses = list(statuses)
        self.retry_after = retry_after
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        status = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        headers = {}
        if status == 429 and self.retry_after is not None:
            headers['Retry-After'] = self.retry_after
        return httpx.Response(status, headers=headers, text="body")


async def _client_with(responder, **kwargs):
    factory = HttpClientFactory()
    client = await factory.get_http_client(**kwargs)
    # Replace the transport underneath the wrapper, so the wrapper -- the
    # thing under test -- stays in place.
    client._transport = httpx.MockTransport(responder)
    return client


def test_factory_constructs_with_no_arguments():
    """CollectorFactory does `http_client_factory or HttpClientFactory()`."""
    assert HttpClientFactory() is not None


def test_the_config_section_exists_and_is_read():
    factory = HttpClientFactory()
    assert factory.client_config, "http_client section missing from every YAML"
    assert factory.client_config['status_forcelist']
    assert factory._per_host_limits, "per-host limits not configured"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_retryable_status_is_retried(status):
    responder = _Responder([status, status, 200])

    async def run():
        client = await _client_with(responder, retries=3, backoff_factor=0.0)
        return await client.get("https://example.test/data")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert responder.calls == 3, "the failing statuses were not retried"


def test_a_non_retryable_status_is_returned_immediately():
    responder = _Responder([404])

    async def run():
        client = await _client_with(responder, retries=3, backoff_factor=0.0)
        return await client.get("https://example.test/data")

    assert asyncio.run(run()).status_code == 404
    assert responder.calls == 1


def test_retries_are_bounded_and_the_last_response_is_returned():
    responder = _Responder([429])

    async def run():
        client = await _client_with(responder, retries=2, backoff_factor=0.0)
        return await client.get("https://example.test/data")

    response = asyncio.run(run())

    assert response.status_code == 429
    assert responder.calls == 3, "expected the original call plus 2 retries"


def test_retry_after_is_obeyed_when_the_server_sends_it():
    """A 429 usually says how long to wait; guessing shorter earns another."""
    responder = _Responder([429, 200], retry_after="0.4")

    async def run():
        client = await _client_with(responder, retries=2, backoff_factor=0.0)
        started = time.monotonic()
        response = await client.get("https://example.test/data")
        return response, time.monotonic() - started

    response, elapsed = asyncio.run(run())

    assert response.status_code == 200
    assert elapsed >= 0.4, f"Retry-After ignored (waited {elapsed:.2f}s)"


def test_an_unparseable_retry_after_falls_back_to_backoff():
    responder = _Responder([429, 200], retry_after="Wed, 21 Oct 2026 07:28:00 GMT")

    async def run():
        client = await _client_with(responder, retries=2, backoff_factor=0.0)
        return await client.get("https://example.test/data")

    assert asyncio.run(run()).status_code == 200


def test_limiters_are_shared_across_factory_instances():
    """The regression: five factories in one run meant five full buckets."""
    first = HttpClientFactory()._limiter_for("https://example.test/a")
    second = HttpClientFactory()._limiter_for("https://example.test/b")

    assert first is second


def test_each_host_gets_its_own_bucket():
    factory = HttpClientFactory()

    reddit = factory._limiter_for("https://www.reddit.com/r/all.json")
    other = factory._limiter_for("https://example.test/data")

    assert reddit is not other
    assert reddit.rate_limit / reddit.per_seconds < other.rate_limit / other.per_seconds, (
        "reddit, which returned 429s, must be limited more tightly than the default"
    )


def test_an_unparseable_url_still_gets_a_limiter():
    assert HttpClientFactory()._limiter_for("not a url") is not None
    assert HttpClientFactory()._limiter_for(None) is not None


def test_the_rate_limiter_actually_paces_requests():
    responder = _Responder([200])
    factory = HttpClientFactory()

    async def run():
        client = await factory.get_http_client(retries=0)
        client._transport = httpx.MockTransport(responder)
        # 2 per second: the third request cannot start before ~1s.
        HttpClientFactory._limiters['slow.test'] = type(
            factory._limiter_for("https://slow.test/")
        )(rate_limit=2, per_seconds=1.0)
        factory._per_host_limits['slow.test'] = {'rate_limit': 2, 'per_seconds': 1.0}

        started = time.monotonic()
        for _ in range(3):
            await client.get("https://slow.test/data")
        return time.monotonic() - started

    assert asyncio.run(run()) >= 0.4, "requests were not paced at all"
