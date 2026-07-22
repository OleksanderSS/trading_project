"""Concrete analyst lenses. Each is a review-only, deterministic module that
reads an AnalysisPacket and returns a ModuleDelta."""
from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
from dean_os.analyst_core.lenses.evidence_gap_lens import EvidenceGapLens
from dean_os.analyst_core.lenses.expectation_gap_lens import ExpectationGapLens
from dean_os.analyst_core.lenses.historical_analog_lens import HistoricalAnalogLens
from dean_os.analyst_core.lenses.hypothesis_ledger_lens import HypothesisLedgerLens
from dean_os.analyst_core.lenses.regime_context_lens import RegimeContextLens
from dean_os.analyst_core.lenses.transmission_mapper_lens import TransmissionMapperLens

__all__ = [
    "EventClassifierLens",
    "EvidenceGapLens",
    "ExpectationGapLens",
    "HistoricalAnalogLens",
    "HypothesisLedgerLens",
    "RegimeContextLens",
    "TransmissionMapperLens",
]
