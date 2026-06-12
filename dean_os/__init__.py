"""DEAN-OS: multi-agent governance layer for the trading pipeline."""

from dean_os.branches import AnalyticalBranch, PipelineBranch
from dean_os.consensus import ConsensusEngine
from dean_os.context_performance import AgentPerformanceByContext
from dean_os.decision_logger import DecisionLogger
from dean_os.event_log import EventLog
from dean_os.execution_gateway import ExecutionGateway, ExecutionPolicy
from dean_os.factory import create_dean_orchestrator, create_hybrid_dean_orchestrator
from dean_os.agent_lab import AgentLabRunner
from dean_os.historical_replay import HistoricalReplayAnalyst, HistoricalReplayRunner
from dean_os.learning import LearningStore
from dean_os.material_loaders import ingest_research_path, load_research_directory, load_research_document
from dean_os.orchestrator import DEANOrchestrator
from dean_os.operation_queue import OperationQueue
from dean_os.outcome_evaluation import OutcomeEvaluationRunner
from dean_os.paper_autonomy import PaperAutonomyRunner
from dean_os.paper_portfolio import PaperPortfolioSimulator
from dean_os.paper_trading import PaperTradeEvaluationRunner, PaperTradeStore
from dean_os.pipeline_adapter import HybridPipelineAdapter
from dean_os.recommendation_memory import RecommendationMemoryStore
from dean_os.replay_price_normalizer import ReplayPriceNormalizer
from dean_os.regime_context import RegimeContextBuilder, normalize_context_tags
from dean_os.agents.chief_review import ChiefReviewAgent
from dean_os.agents.collector_inventory import CollectorInventoryAgent
from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.agents.market_data_freshness import MarketDataFreshnessAgent
from dean_os.agents.model_performance import ModelPerformanceAgent
from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.agents.regime import RegimeAgent
from dean_os.agents.source_routing import SourceRoutingAgent
from dean_os.agents.tuning import TuningAgent
from dean_os.research_corpus import ResearchCorpus
from dean_os.review import AgentReviewBuilder
from dean_os.review_actions import ReviewActionStore
from dean_os.sample_materials import agent_lab_sample_documents
from dean_os.synthesis import EvidenceBoundSynthesizer
from dean_os.registry import AgentRegistry
from dean_os.schemas import (
    AgentCapabilities,
    AgentLabRunReport,
    AnalyticalReport,
    BaseAgentReport,
    ConsensusDecision,
    EvidenceItem,
    ExecutionOutcome,
    AgentLearningRecord,
    FinancialNLPResult,
    MarketContext,
    MarketRegimeSnapshot,
    PaperTradeRecord,
    PipelineReport,
    PipelineActionProposal,
    RecommendationMemoryRecord,
    ResearchChunk,
    ResearchDocument,
    ResearchNote,
    ReviewActionRecord,
    SourceCitation,
)

__all__ = [
    "AgentCapabilities",
    "AgentPerformanceByContext",
    "AgentRegistry",
    "AnalyticalBranch",
    "AnalyticalReport",
    "BaseAgentReport",
    "ChiefReviewAgent",
    "ConsensusDecision",
    "ConsensusEngine",
    "CollectorInventoryAgent",
    "DEANOrchestrator",
    "DecisionLogger",
    "DiaryBridgeAgent",
    "EvidenceItem",
    "EventLog",
    "ExecutionGateway",
    "EvidenceBoundSynthesizer",
    "ExecutionOutcome",
    "ExecutionPolicy",
    "AgentLearningRecord",
    "FinancialNLPResult",
    "AgentLabRunReport",
    "AgentLabRunner",
    "AgentReviewBuilder",
    "LearningStore",
    "ReviewActionStore",
    "HybridPipelineAdapter",
    "HistoricalReplayAnalyst",
    "HistoricalReplayRunner",
    "MarketContext",
    "MarketDataFreshnessAgent",
    "ModelPerformanceAgent",
    "MarketRegimeSnapshot",
    "OperationQueue",
    "OutcomeEvaluationRunner",
    "PaperAutonomyRunner",
    "PaperPortfolioAgent",
    "PaperPortfolioSimulator",
    "PaperTradeEvaluationRunner",
    "PaperTradeRecord",
    "PaperTradeStore",
    "PipelineBranch",
    "PipelineReport",
    "PipelineActionProposal",
    "RecommendationMemoryRecord",
    "RecommendationMemoryStore",
    "ReplayPriceNormalizer",
    "RegimeContextBuilder",
    "RegimeAgent",
    "ResearchChunk",
    "ResearchCorpus",
    "ResearchDocument",
    "ResearchNote",
    "ReviewActionRecord",
    "SourceCitation",
    "SourceRoutingAgent",
    "TuningAgent",
    "create_dean_orchestrator",
    "create_hybrid_dean_orchestrator",
    "agent_lab_sample_documents",
    "ingest_research_path",
    "load_research_directory",
    "load_research_document",
    "normalize_context_tags",
]
