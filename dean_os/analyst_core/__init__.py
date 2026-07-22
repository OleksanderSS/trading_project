"""Analyst core: schemas nucleus + modular lens contract + unified SectorAnalyst.

Implements the review-only, deterministic foundation for the modular analyst
described in the analyst design notes. See ``schemas.py`` and
``lens_contract.py`` for the contracts; ``lenses/`` holds concrete lenses;
``lens_orchestrator.py`` runs lenses sequentially; ``sector_analyst.py``
is the unified entry point for any sector.
"""
from dean_os.analyst_core.lens_contract import (
    AnalysisPacket,
    AnalystLens,
    LensRegistry,
    ModuleDelta,
)
from dean_os.analyst_core.lens_orchestrator import LensOrchestrator
from dean_os.analyst_core.schemas import (
    OUTCOME_HORIZONS,
    REGIME_DIMENSIONS,
    SCENARIO_EDGE_TYPES,
    SCENARIO_NODE_TYPES,
    Confidence,
    EvidenceGap,
    HistoricalOutcomeCheck,
    HorizonOutcome,
    HypothesisLedgerEntry,
    HypothesisStatus,
    Priority,
    RegimeContextVector,
    RegimeDimensionState,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeGraph,
    Trend,
)
from dean_os.analyst_core.sector_analyst import SectorAnalyst, SectorReport

__all__ = [
    # schemas
    "OUTCOME_HORIZONS",
    "REGIME_DIMENSIONS",
    "SCENARIO_NODE_TYPES",
    "SCENARIO_EDGE_TYPES",
    "Confidence",
    "EvidenceGap",
    "HistoricalOutcomeCheck",
    "HorizonOutcome",
    "HypothesisLedgerEntry",
    "HypothesisStatus",
    "Priority",
    "RegimeContextVector",
    "RegimeDimensionState",
    "ScenarioEdge",
    "ScenarioNode",
    "ScenarioOutcomeGraph",
    "Trend",
    # lens contract
    "AnalysisPacket",
    "AnalystLens",
    "LensRegistry",
    "ModuleDelta",
    # orchestrator
    "LensOrchestrator",
    # unified sector analyst
    "SectorAnalyst",
    "SectorReport",
]
