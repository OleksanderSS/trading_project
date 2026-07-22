"""DEAN-OS agent library with lazy public exports.

Importing one lightweight agent must not import every optional research, database,
ML, or broker dependency. Registry class paths should import the concrete module
directly; this package-level API is retained for compatibility.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Guardians
    "RiskAgent": ("dean_os.agents.risk", "RiskAgent"),
    "DataQualityAgent": ("dean_os.agents.data_quality", "DataQualityAgent"),
    "PipelineAuditAgent": ("dean_os.agents.pipeline_audit", "PipelineAuditAgent"),
    "AgentEvaluationControllerAgent": (
        "dean_os.agents.agent_evaluation_controller",
        "AgentEvaluationControllerAgent",
    ),
    # Pipeline control / advisory
    "PipelineControlAgent": ("dean_os.agents.pipeline_control", "PipelineControlAgent"),
    "RegimeAgent": ("dean_os.agents.regime", "RegimeAgent"),
    "MarketDataFreshnessAgent": (
        "dean_os.agents.market_data_freshness",
        "MarketDataFreshnessAgent",
    ),
    "ModelPerformanceAgent": (
        "dean_os.agents.model_performance",
        "ModelPerformanceAgent",
    ),
    "TuningAgent": ("dean_os.agents.tuning", "TuningAgent"),
    "ContextSynthesisAgent": (
        "dean_os.agents.context_synthesis",
        "ContextSynthesisAgent",
    ),
    "PipelineReadinessAgent": (
        "dean_os.agents.pipeline_readiness",
        "PipelineReadinessAgent",
    ),
    # Domain analysts
    "DomainAnalystAgent": ("dean_os.agents.domain_analyst", "DomainAnalystAgent"),
    "DomainAnalyticalAgent": (
        "dean_os.agents.domain_analytical",
        "DomainAnalyticalAgent",
    ),
    "PipelineManagerAgent": (
        "dean_os.agents.pipeline_manager",
        "PipelineManagerAgent",
    ),
    "WorkingDomainAnalystAgent": (
        "dean_os.agents.working_domain_analyst",
        "WorkingDomainAnalystAgent",
    ),
    # Keyword research
    "MacroPolicyAgent": ("dean_os.agents.domain_research", "MacroPolicyAgent"),
    "GeoPoliticalAgent": ("dean_os.agents.domain_research", "GeoPoliticalAgent"),
    "NewsCatalystAgent": ("dean_os.agents.domain_research", "NewsCatalystAgent"),
    "SectorCycleAgent": ("dean_os.agents.domain_research", "SectorCycleAgent"),
    "IndustryMapAgent": ("dean_os.agents.domain_research", "IndustryMapAgent"),
    "ContrarianThesisAgent": (
        "dean_os.agents.domain_research",
        "ContrarianThesisAgent",
    ),
    "ValueScreeningAgent": (
        "dean_os.agents.domain_research",
        "ValueScreeningAgent",
    ),
    "HistoricalAnalogiesAgent": (
        "dean_os.agents.historical_analogies",
        "HistoricalAnalogiesAgent",
    ),
    # Research / ingestion
    "ResearchIngestionAgent": (
        "dean_os.agents.research_agents",
        "ResearchIngestionAgent",
    ),
    "SpecialistResearchAgent": (
        "dean_os.agents.research_agents",
        "SpecialistResearchAgent",
    ),
    "UnifiedResearchAgent": (
        "dean_os.agents.unified_research_agent",
        "UnifiedResearchAgent",
    ),
    "SourceRoutingAgent": (
        "dean_os.agents.source_routing",
        "SourceRoutingAgent",
    ),
    # NLP / events
    "FinancialNLPAgent": ("dean_os.agents.financial_nlp", "FinancialNLPAgent"),
    "NewsEventAnalyzerAgent": (
        "dean_os.agents.news_event_analyzer",
        "NewsEventAnalyzerAgent",
    ),
    # Audit / coherence
    "FreshnessAuditAgent": (
        "dean_os.agents.freshness_audit",
        "FreshnessAuditAgent",
    ),
    "CoherenceScanAgent": (
        "dean_os.agents.coherence_scan",
        "CoherenceScanAgent",
    ),
    # Operations
    "OperationsProposalAgent": (
        "dean_os.agents.operations",
        "OperationsProposalAgent",
    ),
    "ChiefReviewAgent": ("dean_os.agents.chief_review", "ChiefReviewAgent"),
    "DiaryBridgeAgent": ("dean_os.agents.diary_bridge", "DiaryBridgeAgent"),
    "PaperPortfolioAgent": (
        "dean_os.agents.paper_portfolio",
        "PaperPortfolioAgent",
    ),
    "EvidenceSynthesisAgent": (
        "dean_os.agents.synthesis",
        "EvidenceSynthesisAgent",
    ),
    "CollectorHealthAgent": (
        "dean_os.agents.collector_health",
        "CollectorHealthAgent",
    ),
    "CollectorInventoryAgent": (
        "dean_os.agents.collector_inventory",
        "CollectorInventoryAgent",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
