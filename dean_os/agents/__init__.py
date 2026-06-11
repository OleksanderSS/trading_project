"""Built-in DEAN-OS agents."""

from dean_os.agents.chief_review import ChiefReviewAgent
from dean_os.agents.data_quality import DataQualityAgent
from dean_os.agents.collector_inventory import CollectorInventoryAgent
from dean_os.agents.collector_health import CollectorHealthAgent
from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.agents.domain_research import (
    ContrarianThesisAgent,
    GeoPoliticalAgent,
    HistoricalAnalogiesAgent,
    IndustryMapAgent,
    MacroPolicyAgent,
    NewsCatalystAgent,
    SectorCycleAgent,
    ValueScreeningAgent,
)
from dean_os.agents.financial_nlp import FinancialNLPAgent
from dean_os.agents.market_data_freshness import MarketDataFreshnessAgent
from dean_os.agents.model_performance import ModelPerformanceAgent
from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.agents.pipeline_audit import PipelineAuditAgent
from dean_os.agents.regime import RegimeAgent
from dean_os.agents.operations import OperationsProposalAgent
from dean_os.agents.research_agents import ResearchIngestionAgent, SpecialistResearchAgent
from dean_os.agents.synthesis import EvidenceSynthesisAgent
from dean_os.agents.risk import RiskAgent
from dean_os.agents.source_routing import SourceRoutingAgent
from dean_os.agents.tuning import TuningAgent

__all__ = [
    "ContrarianThesisAgent",
    "ChiefReviewAgent",
    "CollectorInventoryAgent",
    "CollectorHealthAgent",
    "DataQualityAgent",
    "DiaryBridgeAgent",
    "FinancialNLPAgent",
    "EvidenceSynthesisAgent",
    "GeoPoliticalAgent",
    "HistoricalAnalogiesAgent",
    "IndustryMapAgent",
    "MacroPolicyAgent",
    "MarketDataFreshnessAgent",
    "ModelPerformanceAgent",
    "NewsCatalystAgent",
    "OperationsProposalAgent",
    "PaperPortfolioAgent",
    "PipelineAuditAgent",
    "RegimeAgent",
    "ResearchIngestionAgent",
    "RiskAgent",
    "SectorCycleAgent",
    "SpecialistResearchAgent",
    "SourceRoutingAgent",
    "TuningAgent",
    "ValueScreeningAgent",
]
