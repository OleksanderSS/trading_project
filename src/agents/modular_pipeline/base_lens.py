from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ScenarioNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    impact: str = Field(min_length=1)


class LensInsights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_area: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    downstream_impact: list[str] = Field(default_factory=list)


class LensAnalysisResult(BaseModel):
    """Schema for an untrusted LLM proposal, never an evidence record."""

    model_config = ConfigDict(extra="forbid")

    insights: LensInsights
    scenario_nodes: list[ScenarioNode]
    evidence_gaps: list[str]


class BaseLens(ABC):
    """Optional analytical lens with no evidence or execution authority."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    @property
    @abstractmethod
    def lens_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_tags(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def analyze(
        self, source_text: str, current_state: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def request_proposal(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        if self.llm_client is None:
            return self.unavailable_delta("llm_client_not_configured")
        result = await self.llm_client.generate_structured(
            prompt=prompt,
            response_model=LensAnalysisResult,
            system_prompt=system_prompt,
        )
        if result is None:
            return self.unavailable_delta("llm_call_unavailable_or_failed")
        return self.proposal_delta(result)

    def proposal_delta(self, result: LensAnalysisResult) -> dict[str, Any]:
        payload = result.model_dump()
        payload.update(_authority("llm_proposal_ready"))
        return payload

    def unavailable_delta(
        self,
        reason: Literal[
            "llm_client_not_configured", "llm_call_unavailable_or_failed"
        ],
    ) -> dict[str, Any]:
        return {
            "insights": {},
            "scenario_nodes": [],
            "evidence_gaps": [reason],
            **_authority("unavailable"),
        }


def _authority(status: str) -> dict[str, Any]:
    return {
        "analysis_status": status,
        "authority": {
            "proposal_only": True,
            "is_evidence": False,
            "may_confirm_hypothesis": False,
            "may_change_hypothesis_status": False,
            "may_write_learning_memory": False,
            "may_trade": False,
        },
    }


__all__ = [
    "BaseLens",
    "LensAnalysisResult",
    "LensInsights",
    "ScenarioNode",
]
