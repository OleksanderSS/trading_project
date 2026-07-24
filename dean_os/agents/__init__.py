"""DEAN-OS Agent Library.

Agents are organized by role and branch (pipeline / analytical).

GUARDIAN AGENTS  (pipeline branch, hard-veto authority)
────────────────────────────────────────────────────────
  RiskAgent               — drawdown / VaR / exposure limits
  DataQualityAgent        — data integrity, leakage, schema checks
  PipelineAuditAgent      — pipeline integrity and config verification

PIPELINE ANALYSTS  (pipeline branch, advisory — no veto)
─────────────────────────────────────────────────────────
  RegimeAgent             — market regime classification
  MarketDataFreshnessAgent — staleness / coverage checks
  PipelineReadinessAgent  — Stage 4/5 readiness gate evidence
  ModelPerformanceAgent   — model accuracy and calibration tracking
  TuningAgent             — hyperparameter tuning proposals (dry-run only)
  ContextSynthesisAgent   — Stage 5/7 context compatibility layer

DOMAIN ANALYST AGENTS  (pipeline or analytical branch)
───────────────────────────────────────────────────────
  DomainAnalystAgent      — generic domain analyst (cloned per sector via YAML)
  PipelineManagerAgent    — composite: pipeline + sector analyst in one agent

KEYWORD RESEARCH AGENTS  (analytical branch)
─────────────────────────────────────────────
  MacroPolicyAgent        — macro / central bank / fiscal policy signals
  GeoPoliticalAgent       — geopolitical risk signals
  NewsCatalystAgent       — news catalyst screening
  SectorCycleAgent        — sector cycle position
  IndustryMapAgent        — industry structure and competitive dynamics
  HistoricalAnalogiesAgent — historical analog matching
  ContrarianThesisAgent   — contrarian signal detection
  ValueScreeningAgent     — quantitative value screening

RESEARCH & INGESTION  (analytical branch)
──────────────────────────────────────────
  ResearchIngestionAgent  — ingests documents into research corpus
  SpecialistResearchAgent — deep-dive specialist research queries
  UnifiedResearchAgent    — unified corpus search + synthesis
  SourceRoutingAgent      — routes evidence to appropriate producers

NLP & EVENTS  (analytical branch)
───────────────────────────────────
  FinancialNLPAgent       — sentiment NLP on news / filings
  NewsEventAnalyzerAgent  — event classification + causal graph

AUDIT & COHERENCE  (analytical branch)
────────────────────────────────────────
  FreshnessAuditAgent     — data age vs as_of thresholds
  CoherenceScanAgent      — cross-agent overlap and coherence

META / OPERATIONS  (pipeline branch, dry-run / proposal only)
───────────────────────────────────────────────────────────────
  OperationsProposalAgent — pipeline operation proposals (never executes)
  ChiefReviewAgent        — cross-agent review orchestration
  DiaryBridgeAgent        — decision diary + memory bridge
  PaperPortfolioAgent     — paper trading simulation (no real execution)
  EvidenceSynthesisAgent  — evidence aggregation across sources

WORKING TOOLS  (standalone / not in orchestrator loop)
───────────────────────────────────────────────────────
  WorkingDomainAnalystAgent — full report writer for manual review sessions
                              (not in agent_registry.yaml, used via CLI)
"""

# ── Guardian Agents (hard-veto) ───────────────────────────────────────────────
from dean_os.agents.risk import RiskAgent
from dean_os.agents.data_quality import DataQualityAgent
from dean_os.agents.pipeline_audit import PipelineAuditAgent
from dean_os.agents.agent_evaluation_controller import AgentEvaluationControllerAgent

# ── Pipeline Analysts (advisory) ─────────────────────────────────────────────
from dean_os.agents.regime import RegimeAgent
from dean_os.agents.market_data_freshness import MarketDataFreshnessAgent
from dean_os.agents.model_performance import ModelPerformanceAgent
from dean_os.agents.tuning import TuningAgent
from dean_os.agents.context_synthesis import ContextSynthesisAgent
from dean_os.agents.pipeline_readiness import PipelineReadinessAgent

# ── Domain Analysts ───────────────────────────────────────────────────────────
from dean_os.agents.domain_analyst import DomainAnalystAgent
from dean_os.agents.pipeline_manager import PipelineManagerAgent

# ── Keyword Research Agents ───────────────────────────────────────────────────
from dean_os.agents.domain_research import (
    MacroPolicyAgent,
    GeoPoliticalAgent,
    NewsCatalystAgent,
    SectorCycleAgent,
    IndustryMapAgent,
    ContrarianThesisAgent,
    ValueScreeningAgent,
)
from dean_os.agents.historical_analogies import HistoricalAnalogiesAgent

# ── Research & Ingestion ──────────────────────────────────────────────────────
from dean_os.agents.research_agents import ResearchIngestionAgent, SpecialistResearchAgent
from dean_os.agents.unified_research_agent import UnifiedResearchAgent
from dean_os.agents.source_routing import SourceRoutingAgent

# ── NLP & Events ─────────────────────────────────────────────────────────────
from dean_os.agents.financial_nlp import FinancialNLPAgent
from dean_os.agents.news_event_analyzer import NewsEventAnalyzerAgent

# ── Audit & Coherence ─────────────────────────────────────────────────────────
from dean_os.agents.freshness_audit import FreshnessAuditAgent
from dean_os.agents.coherence_scan import CoherenceScanAgent

# ── Meta / Operations ─────────────────────────────────────────────────────────
from dean_os.agents.operations import OperationsProposalAgent
from dean_os.agents.chief_review import ChiefReviewAgent
from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.agents.synthesis import EvidenceSynthesisAgent

# ── Working Tools (standalone) ────────────────────────────────────────────────
from dean_os.agents.working_domain_analyst import WorkingDomainAnalystAgent

__all__ = [
    # Guardians
    "RiskAgent",
    "DataQualityAgent",
    "PipelineAuditAgent",
    "AgentEvaluationControllerAgent",
    # Pipeline analysts
    "RegimeAgent",
    "MarketDataFreshnessAgent",
    "ModelPerformanceAgent",
    "TuningAgent",
    "ContextSynthesisAgent",
    "PipelineReadinessAgent",
    # Domain analysts
    "DomainAnalystAgent",
    "PipelineManagerAgent",
    # Keyword research
    "MacroPolicyAgent",
    "GeoPoliticalAgent",
    "NewsCatalystAgent",
    "SectorCycleAgent",
    "IndustryMapAgent",
    "HistoricalAnalogiesAgent",
    "ContrarianThesisAgent",
    "ValueScreeningAgent",
    # Research & ingestion
    "ResearchIngestionAgent",
    "SpecialistResearchAgent",
    "UnifiedResearchAgent",
    "SourceRoutingAgent",
    # NLP & events
    "FinancialNLPAgent",
    "NewsEventAnalyzerAgent",
    # Audit & coherence
    "FreshnessAuditAgent",
    "CoherenceScanAgent",
    # Meta / operations
    "OperationsProposalAgent",
    "ChiefReviewAgent",
    "DiaryBridgeAgent",
    "PaperPortfolioAgent",
    "EvidenceSynthesisAgent",
    # Working tools
    "WorkingDomainAnalystAgent",
]
