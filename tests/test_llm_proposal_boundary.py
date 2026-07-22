from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.agents.modular_pipeline.base_lens import LensAnalysisResult
from src.agents.modular_pipeline.orchestrator import get_default_orchestrator
from src.core.clients.llm_client import LLMClient


def _parsed_response():
    parsed = LensAnalysisResult.model_validate(
        {
            "insights": {
                "focus_area": "test",
                "observation": "conditional inference",
                "downstream_impact": [],
            },
            "scenario_nodes": [],
            "evidence_gaps": ["primary source missing"],
        }
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))]
    )


def test_missing_credentials_produce_no_mock_analysis(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = LLMClient()

    assert client.available is False
    assert client.availability()["may_create_evidence"] is False
    assert asyncio.run(
        client.generate_structured("text", LensAnalysisResult)
    ) is None


def test_non_transient_failure_is_not_retried() -> None:
    calls = 0

    class Completions:
        async def parse(self, **kwargs):
            nonlocal calls
            calls += 1
            raise ValueError("bad schema")

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    client = LLMClient(api_key="present", model="explicit-test-model", client=fake)

    assert asyncio.run(
        client.generate_structured("text", LensAnalysisResult, max_retries=3)
    ) is None
    assert calls == 1


def test_transient_failure_uses_bounded_retry(monkeypatch) -> None:
    calls = 0
    RateLimitError = type("RateLimitError", (Exception,), {})

    class Completions:
        async def parse(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RateLimitError()
            return _parsed_response()

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("src.core.clients.llm_client.asyncio.sleep", no_sleep)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client = LLMClient(api_key="present", model="explicit-test-model", client=fake)

    result = asyncio.run(
        client.generate_structured("text", LensAnalysisResult, max_retries=2)
    )
    assert result is not None
    assert calls == 2


def test_default_lenses_are_explicitly_unavailable_without_llm() -> None:
    packet = asyncio.run(
        get_default_orchestrator().analyze(
            "A headline that must not generate canned conclusions.",
            ["technology"],
        )
    )

    assert packet["insights"] == {}
    assert packet["scenario_nodes"] == []
    assert packet["authority"]["is_evidence"] is False
    assert packet["lens_status"]["TechnologySectorLens"][
        "analysis_status"
    ] == "unavailable"
