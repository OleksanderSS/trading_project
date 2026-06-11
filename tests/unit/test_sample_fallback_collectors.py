import asyncio

import pytest

from src.data.collectors.cftc_collector import CFTCCollector
from src.data.collectors.put_call_ratio_collector import PutCallRatioCollector


class LoggerStub:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class ResponseStub:
    status_code = 200
    text = "exchange volume page without the expected ratio"


class HttpClientStub:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        return ResponseStub()


class HttpClientFactoryStub:
    def get_http_client(self, timeout):
        return HttpClientStub()


def _put_call_collector(allow_sample_fallback):
    collector = object.__new__(PutCallRatioCollector)
    collector.allow_sample_fallback = allow_sample_fallback
    collector.timeout = 1
    collector.http_client_factory = HttpClientFactoryStub()
    collector.logger = LoggerStub()
    return collector


def test_put_call_ratio_missing_ratio_does_not_create_sample_without_opt_in():
    collector = _put_call_collector(allow_sample_fallback=False)

    with pytest.raises(RuntimeError, match="sample fallback disabled"):
        asyncio.run(collector._fetch_put_call_data())


def test_put_call_ratio_sample_fallback_is_marked_not_trainable():
    collector = _put_call_collector(allow_sample_fallback=True)

    data = asyncio.run(collector._fetch_put_call_data())

    assert data
    assert all(row["is_synthetic"] for row in data)
    assert all(row["eligible_for_training"] is False for row in data)


def test_cftc_sample_fallback_is_marked_not_trainable():
    collector = object.__new__(CFTCCollector)

    data = collector._create_sample_cftc_data("GOLD")

    assert data
    assert all(row["is_synthetic"] for row in data)
    assert all(row["eligible_for_training"] is False for row in data)
