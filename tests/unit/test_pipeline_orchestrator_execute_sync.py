import asyncio

import pytest

from src.pipeline.pipeline_orchestrator import PipelineOrchestrator


class StubConfig:
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds

    def get(self, key, default=None):
        if key == "pipeline.sync_timeout_seconds":
            return self.timeout_seconds
        return default


def _orchestrator(timeout_seconds=300):
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.config_manager = StubConfig(timeout_seconds)
    return orchestrator


async def _returns(value):
    return value


async def _raises():
    raise ValueError("boom")


def test_execute_sync_returns_value_inside_running_event_loop():
    async def runner():
        return _orchestrator()._execute_sync(_returns("ok"))

    assert asyncio.run(runner()) == "ok"


def test_execute_sync_propagates_worker_exception():
    async def runner():
        return _orchestrator()._execute_sync(_raises())

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(runner())
