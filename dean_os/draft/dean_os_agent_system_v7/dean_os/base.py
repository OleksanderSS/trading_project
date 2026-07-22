from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dean_os.schemas import AgentCapabilities, BaseAgentReport, EvidenceItem, MarketContext
from dean_os.utils import sha256_json


class BaseAgent(ABC):
    name: str
    version = "0.1.0"
    branch = "pipeline"

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None):
        self.name = name or getattr(self, "name", self.__class__.__name__)
        self.config = config or {}
        self.capabilities = AgentCapabilities(
            can_veto=self.config.get("veto_level") == "hard",
            timeout_seconds=int(self.config.get("timeout_seconds", 10)),
            error_behavior=self.config.get("error_behavior", "skip"),
            proposal_only=bool(self.config.get("proposal_only", False)),
        )

    def check_prerequisites(self, context: MarketContext) -> bool:
        required_inputs = self.config.get("required_inputs", [])
        root = Path(self.config.get("project_root", "."))
        for relative_path in required_inputs:
            if not (root / relative_path).exists():
                return False
        return True

    def should_run_in_phase(self, context: MarketContext) -> bool:
        run_phases = self.config.get("run_phases")
        if not run_phases:
            return True
        return context.phase in {
            str(phase).strip()
            for phase in run_phases
            if str(phase).strip()
        }

    @abstractmethod
    async def run(self, context: MarketContext) -> BaseAgentReport:
        raise NotImplementedError

    def evidence(self, source_type: str, source: str, key: str, value: Any) -> EvidenceItem:
        return EvidenceItem(source_type=source_type, source=source, key=key, value=value)

    def context_hash(self, context: MarketContext) -> str:
        compact_context = {
            "phase": context.phase,
            "as_of": context.as_of,
            "tickers": context.tickers,
            "timeframes": context.timeframes,
            "timeframe": context.timeframe,
            "positions": context.positions,
            "metadata": context.metadata,
        }
        return sha256_json(compact_context)


class AnalyticalAgent(BaseAgent):
    branch = "analytical"
