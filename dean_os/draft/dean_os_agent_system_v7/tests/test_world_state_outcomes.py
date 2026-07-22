from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dean_os.analyst_core import ScenarioNode, ScenarioOutcomeGraph
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids import (
    ContextDimensionState,
    ContextGrid,
    ContextGridNode,
    IndicatorObservation,
    IndicatorStateGrid,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_outcomes import (
    FalseAnalogyEvaluator,
    LearningPromotionGate,
    OutcomeCalibrationService,
    OutcomeEvidence,
    OutcomeReviewDecisionBuilder,
    OutcomeSnapshotBuilder,
    ReviewDecisionStatus,
    SQLiteOutcomeStore,
    ScenarioResolution,
    ScenarioResolutionStatus,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import (
    HistoricalAnalogMatch,
    WorldStateSnapshotBuilder,
)

DOMAIN = "semiconductor_ai_infrastructure"


def _world_state(
    as_of: str,
    *,
    base_probability: float = 0.7,
    economic_phase: str = "expansion",
):
    context = ContextGrid(
        domain_id=DOMAIN,
        as_of=as_of,
        status="ready",
        nodes=[
            ContextGridNode(
                node_id="global",
                level="global",
                label="Global",
                dimensions={
                    "economic_phase": ContextDimensionState(
                        dimension="economic_phase",
                        state=economic_phase,
                        direction="stable",
                        confidence=0.8,
                        as_of=as_of,
                        source="fixture",
                        evidence_ids=[f"context:{economic_phase}"],
                    )
                },
            )
        ],
        completeness={"status": "fixture"},
        point_in_time={"status": "ready"},
    )
    indicators = IndicatorStateGrid(
        domain_id=DOMAIN,
        as_of=as_of,
        status="ready",
        observations=[
            IndicatorObservation(
                indicator_id=f"indicator:{as_of}",
                family="sector",
                scope=DOMAIN,
                name="hbm_supply_growth",
                value=12.0,
                unit="percent",
                period="fixture",
                available_at=as_of,
                as_of=as_of,
                source="fixture",
                evidence_status="point_in_time",
                quality_score=1.0,
                provenance={"evidence_id": "indicator:hbm"},
            )
        ],
        family_counts={"sector": 1},
        point_in_time={"status": "ready"},
    )
    graph = ScenarioOutcomeGraph(
        as_of=as_of,
        nodes=[
            ScenarioNode(
                node_id="scenario:base",
                node_type="scenario",
                label="Supply catches demand",
                as_of=as_of,
                probability=base_probability,
                probability_kind="review_prior",
            ),
            ScenarioNode(
                node_id="scenario:stress",
                node_type="scenario",
                label="Bottleneck persists",
                as_of=as_of,
                probability=1.0 - base_probability,
                probability_kind="review_prior",
            ),
        ],
    )
    return WorldStateSnapshotBuilder().build(
        domain_id=DOMAIN,
        as_of=as_of,
        knowledge_cutoff=as_of,
        context_grid=context,
        indicator_state_grid=indicators,
        scenario_outcome_graph=graph,
        world_model_summary={"packet_status": "fixture"},
    )


def _evidence(evaluation_as_of: str, *, evidence_id: str = "outcome:1") -> OutcomeEvidence:
    return OutcomeEvidence(
        evidence_id=evidence_id,
        source="fixture",
        observed_at=evaluation_as_of,
        available_at=evaluation_as_of,
        metric_name="realized_supply_path",
        value="supply_catches_demand",
        quality_score=1.0,
    )


def _evaluated_outcome(
    world_state,
    *,
    horizon_days: int = 5,
    winner: str = "scenario:base",
    outcome_class: str = "normalization",
    evaluation_as_of: str = "2026-07-06T12:00:00+00:00",
):
    evidence = _evidence(evaluation_as_of)
    loser = "scenario:stress" if winner == "scenario:base" else "scenario:base"
    return OutcomeSnapshotBuilder().build(
        world_state=world_state,
        horizon_days=horizon_days,
        evaluation_as_of=evaluation_as_of,
        evidence=[evidence],
        scenario_resolutions=[
            ScenarioResolution(
                scenario_node_id=winner,
                status=ScenarioResolutionStatus.REALIZED,
                outcome_class=outcome_class,
                confidence=0.9,
                supporting_evidence_ids=[evidence.evidence_id],
                rationale="Observed path matched this scenario.",
            ),
            ScenarioResolution(
                scenario_node_id=loser,
                status=ScenarioResolutionStatus.REJECTED,
                confidence=0.8,
                contradicting_evidence_ids=[evidence.evidence_id],
                rationale="Observed path contradicted this scenario.",
            ),
        ],
    )


def test_outcome_builder_rejects_evaluation_before_fixed_horizon() -> None:
    state = _world_state("2026-07-01T12:00:00+00:00")

    with pytest.raises(ValueError, match="before its fixed horizon"):
        OutcomeSnapshotBuilder().build(
            world_state=state,
            horizon_days=5,
            evaluation_as_of="2026-07-05T11:59:59+00:00",
            evidence=[],
        )


def test_outcome_builder_rejects_future_evidence() -> None:
    state = _world_state("2026-07-01T12:00:00+00:00")

    with pytest.raises(ValueError, match="future leakage"):
        OutcomeSnapshotBuilder().build(
            world_state=state,
            horizon_days=5,
            evaluation_as_of="2026-07-06T12:00:00+00:00",
            evidence=[
                OutcomeEvidence(
                    evidence_id="future",
                    source="fixture",
                    observed_at="2026-07-06T12:00:00+00:00",
                    available_at="2026-07-06T12:01:00+00:00",
                    metric_name="future_metric",
                    value=1,
                )
            ],
        )


def test_scenario_probability_score_is_computed_without_mutating_graph() -> None:
    state = _world_state("2026-07-01T12:00:00+00:00", base_probability=0.7)
    outcome = _evaluated_outcome(state)

    assert outcome.status == "evaluated_pending_review"
    assert outcome.probability_score.scored is True
    assert outcome.probability_score.realized_scenario_node_id == "scenario:base"
    assert outcome.probability_score.top_scenario_correct is True
    assert outcome.probability_score.winner_probability == pytest.approx(0.7)
    assert outcome.probability_score.brier_score == pytest.approx(0.09)
    assert outcome.authority_boundary["can_write_learning_memory"] is False
    assert state.scenario_outcome_graph.nodes[0].probability == pytest.approx(0.7)


def test_outcome_store_and_reviews_are_append_only_and_idempotent(tmp_path: Path) -> None:
    store = SQLiteOutcomeStore(tmp_path / "world.sqlite3")
    state = _world_state("2026-07-01T12:00:00+00:00")
    outcome = _evaluated_outcome(state)

    first = store.append(outcome)
    second = store.append(outcome)
    review = OutcomeReviewDecisionBuilder().build(
        outcome_snapshot_id=outcome.outcome_snapshot_id,
        decision=ReviewDecisionStatus.APPROVED,
        reviewer="human-reviewer",
        rationale="Evidence and scenario resolution were checked.",
        decided_at="2026-07-06T13:00:00+00:00",
    )
    review_first = store.append_review(review)
    review_second = store.append_review(review)

    assert first.status == "stored"
    assert second.status == "already_exists"
    assert review_first.status == "stored"
    assert review_second.status == "already_exists"
    assert store.get(outcome.outcome_snapshot_id) == outcome
    assert store.latest_review(outcome.outcome_snapshot_id) == review

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE world_state_outcomes SET status = 'changed' "
                "WHERE outcome_snapshot_id = ?",
                (outcome.outcome_snapshot_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM world_state_outcome_reviews WHERE review_decision_id = ?",
                (review.review_decision_id,),
            )


def test_calibration_proposal_requires_human_approved_scored_sample(tmp_path: Path) -> None:
    store = SQLiteOutcomeStore(tmp_path / "world.sqlite3")
    state = _world_state("2026-07-01T12:00:00+00:00")
    outcome = _evaluated_outcome(state)
    store.append(outcome)

    blocked = OutcomeCalibrationService().propose(
        outcome_store=store,
        domain_id=DOMAIN,
        horizon_days=5,
        min_approved_samples=1,
    )
    assert blocked.status == "blocked"
    assert blocked.approved_scored_sample_count == 0
    assert blocked.can_change_probabilities is False

    review = OutcomeReviewDecisionBuilder().build(
        outcome_snapshot_id=outcome.outcome_snapshot_id,
        decision="approved",
        reviewer="reviewer",
        rationale="Approved for calibration sample only.",
        decided_at="2026-07-06T13:00:00+00:00",
    )
    store.append_review(review)
    ready = OutcomeCalibrationService().propose(
        outcome_store=store,
        domain_id=DOMAIN,
        horizon_days=5,
        min_approved_samples=1,
    )
    assert ready.status == "ready_for_human_calibration_review"
    assert ready.approved_scored_sample_count == 1
    assert ready.mean_brier_score == pytest.approx(0.09)
    assert ready.can_write_learning_memory is False


def test_false_analogy_score_flags_similar_state_with_different_outcome() -> None:
    target_state = _world_state("2026-07-01T12:00:00+00:00")
    analog_state = _world_state("2026-06-01T12:00:00+00:00")
    target_outcome = _evaluated_outcome(
        target_state,
        winner="scenario:base",
        outcome_class="normalization",
    )
    analog_outcome = _evaluated_outcome(
        analog_state,
        winner="scenario:stress",
        outcome_class="persistent_bottleneck",
        evaluation_as_of="2026-06-06T12:00:00+00:00",
    )
    match = HistoricalAnalogMatch(
        snapshot_id=analog_state.snapshot_id,
        domain_id=DOMAIN,
        as_of=analog_state.as_of,
        knowledge_cutoff=analog_state.knowledge_cutoff,
        similarity_score=0.9,
    )

    assessment = FalseAnalogyEvaluator().assess(
        target_outcome=target_outcome,
        analog_match=match,
        analog_outcome=analog_outcome,
    )

    assert assessment.outcome_path_match is False
    assert assessment.false_analogy_score == pytest.approx(0.9)
    assert assessment.risk_band == "high"
    assert assessment.eligible_for_learning is False


def test_learning_promotion_gate_never_writes_and_requires_explicit_approval(tmp_path: Path) -> None:
    store = SQLiteOutcomeStore(tmp_path / "world.sqlite3")
    state = _world_state("2026-07-01T12:00:00+00:00")
    outcome = _evaluated_outcome(state)
    store.append(outcome)
    store.append_review(
        OutcomeReviewDecisionBuilder().build(
            outcome_snapshot_id=outcome.outcome_snapshot_id,
            decision="approved",
            reviewer="reviewer",
            rationale="Approved for calibration sample only.",
            decided_at="2026-07-06T13:00:00+00:00",
        )
    )
    proposal = OutcomeCalibrationService().propose(
        outcome_store=store,
        domain_id=DOMAIN,
        horizon_days=5,
        min_approved_samples=1,
    )

    blocked = LearningPromotionGate().evaluate(
        calibration_proposal=proposal,
        reviewer_decision="pending",
    )
    approved = LearningPromotionGate().evaluate(
        calibration_proposal=proposal,
        reviewer_decision="approved",
    )

    assert blocked.status == "blocked"
    assert approved.status == "approved_for_separate_manual_shadow_implementation"
    assert approved.eligible_for_manual_implementation is True
    assert approved.learning_memory_write_performed is False
    assert approved.model_promotion_performed is False
    assert approved.config_write_performed is False


def test_agent_system_readiness_separates_structure_from_operation() -> None:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.agent_system_readiness import AgentSystemReadinessAssessor

    package_root = Path(__file__).resolve().parents[1]
    report = AgentSystemReadinessAssessor().assess(
        package_root=package_root,
        domain_id=DOMAIN,
        pipeline_deferred=True,
    )

    assert report.overall_status == "runnable_structural_mvp_with_major_operational_gaps"
    assert report.structural_readiness > report.operational_readiness
    assert report.structural_readiness < 0.9
    assert {branch.branch for branch in report.branches} == {
        "pipeline",
        "analytical",
        "world_model",
        "replay_learning",
    }
    pipeline = next(branch for branch in report.branches if branch.branch == "pipeline")
    assert pipeline.status == "prepared_boundary_deferred"


def test_agent_only_factory_disables_heavy_pipeline(tmp_path: Path) -> None:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import create_agent_only_system

    system = create_agent_only_system(
        project_root=tmp_path,
        domain_id=DOMAIN,
        save_world_state_snapshots=False,
    )

    assert system.orchestrator.pipeline_runner is None
    assert system.domain_id == DOMAIN
