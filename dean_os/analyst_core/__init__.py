"""Analyst core: schemas nucleus + modular lens contract (Phase 1).

Implements the review-only, deterministic foundation for the modular analyst
described in the analyst design notes. See ``schemas.py`` and
``lens_contract.py`` for the contracts; ``lenses/`` holds concrete lenses.
"""
from dean_os.analyst_core.lens_contract import (
    AnalysisPacket,
    AnalystLens,
    LensRegistry,
    ModuleDelta,
)
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
]
