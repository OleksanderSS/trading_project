from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from dean_os.analyst_core import ScenarioNode, ScenarioOutcomeGraph
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids import (
    ContextDimensionState,
    ContextGrid,
    ContextGridNode,
    IndicatorObservation,
    IndicatorStateGrid,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.world_state_store import (
    HistoricalWorldStateRetriever,
    InMemoryWorldStateStore,
    SQLiteWorldStateStore,
    WorldStateSnapshot,
    WorldStateSnapshotBuilder,
)

DOMAIN = "semiconductor_ai_infrastructure"


def _context_grid(
    as_of: str,
    *,
    economic_phase: str = "expansion",
    ai_cycle: str = "capex_boom",
) -> ContextGrid:
    return ContextGrid(
        domain_id=DOMAIN,
        as_of=as_of,
        status="ready",
        nodes=[
            ContextGridNode(
                node_id="global",
                level="global",
                label="Global context",
                dimensions={
                    "economic_phase": ContextDimensionState(
                        dimension="economic_phase",
                        state=economic_phase,
                        direction="stable",
                        confidence=0.7,
                        as_of=as_of,
                        source="fixture",
                        evidence_ids=[f"evidence:{economic_phase}"],
                    ),
                    "ai_cycle": ContextDimensionState(
                        dimension="ai_cycle",
                        state=ai_cycle,
                        direction="stable",
                        confidence=0.8,
                        as_of=as_of,
                        source="fixture",
                        evidence_ids=[f"evidence:{ai_cycle}"],
                    ),
                    # Unknown dimensions must not inflate analog similarity.
                    "credit_phase": ContextDimensionState(
                        dimension="credit_phase",
                        state="unknown",
                        as_of=as_of,
                        source="missing",
                    ),
                },
            )
        ],
        completeness={"status": "fixture"},
        point_in_time={"status": "ready"},
    )


def _indicator_grid(
    as_of: str,
    *,
    available_at: str | None = None,
    policy_rate: float = 5.25,
) -> IndicatorStateGrid:
    return IndicatorStateGrid(
        domain_id=DOMAIN,
        as_of=as_of,
        status="ready",
        observations=[
            IndicatorObservation(
                indicator_id=f"policy-rate:{as_of}:{policy_rate}",
                family="macro",
                scope="global",
                name="policy_rate",
                value=policy_rate,
                unit="percent",
                period="2026-07",
                available_at=available_at or as_of,
                as_of=as_of,
                source="fixture",
                evidence_status="point_in_time",
                quality_score=1.0,
                provenance={"evidence_id": f"macro:{policy_rate}"},
            )
        ],
        family_counts={"macro": 1},
        point_in_time={"status": "ready"},
    )


def _scenario_graph(as_of: str, *, base_probability: float = 0.6) -> ScenarioOutcomeGraph:
    return ScenarioOutcomeGraph(
        as_of=as_of,
        nodes=[
            ScenarioNode(
                node_id="scenario:base",
                node_type="scenario",
                label="Capacity expansion absorbs demand",
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


def _snapshot(
    as_of: str,
    *,
    knowledge_cutoff: str | None = None,
    economic_phase: str = "expansion",
    ai_cycle: str = "capex_boom",
    policy_rate: float = 5.25,
    available_at: str | None = None,
    scenario: bool = True,
    parent_snapshot_id: str | None = None,
):
    return WorldStateSnapshotBuilder().build(
        domain_id=DOMAIN,
        as_of=as_of,
        knowledge_cutoff=knowledge_cutoff or as_of,
        context_grid=_context_grid(
            as_of,
            economic_phase=economic_phase,
            ai_cycle=ai_cycle,
        ),
        indicator_state_grid=_indicator_grid(
            as_of,
            available_at=available_at,
            policy_rate=policy_rate,
        ),
        scenario_outcome_graph=_scenario_graph(as_of) if scenario else None,
        world_model_summary={"packet_status": "fixture"},
        parent_snapshot_id=parent_snapshot_id,
    )


def test_world_state_builder_rejects_future_indicator_leakage() -> None:
    as_of = "2026-07-12T12:00:00+00:00"

    with pytest.raises(ValueError, match="future_indicator_observation"):
        _snapshot(
            as_of,
            knowledge_cutoff=as_of,
            available_at="2026-07-12T12:01:00+00:00",
        )


def test_world_state_snapshot_hash_detects_payload_tampering() -> None:
    snapshot = _snapshot("2026-07-12T12:00:00+00:00")
    payload = snapshot.model_dump(mode="json")
    payload["context_grid"]["nodes"][0]["dimensions"]["economic_phase"][
        "state"
    ] = "recession"

    with pytest.raises(ValidationError, match="content hash"):
        WorldStateSnapshot.model_validate(payload)


def test_sqlite_world_state_store_is_atomic_append_only_and_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "world_states.sqlite3"
    store = SQLiteWorldStateStore(path)
    snapshot = _snapshot("2026-07-12T12:00:00+00:00")

    first = store.append(snapshot)
    second = store.append(snapshot)
    loaded = store.get(snapshot.snapshot_id)

    assert first.status == "stored"
    assert second.status == "already_exists"
    assert loaded == snapshot
    assert loaded is not None
    assert loaded.scenario_outcome_graph is not None
    assert set(loaded.integrity.atomic_components) == {
        "context_grid",
        "indicator_state_grid",
        "scenario_outcome_graph",
    }

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE world_state_snapshots SET domain_id = 'changed' "
                "WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM world_state_snapshots WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            )


def test_get_as_of_uses_real_time_not_lexicographic_offset_order(
    tmp_path: Path,
) -> None:
    store = SQLiteWorldStateStore(tmp_path / "world_states.sqlite3")
    # 13:00+02:00 == 11:00 UTC
    earlier = _snapshot("2026-07-12T13:00:00+02:00", policy_rate=5.0)
    # 12:00+00:00 == 12:00 UTC
    later = _snapshot("2026-07-12T12:00:00+00:00", policy_rate=5.5)
    store.append(earlier)
    store.append(later)

    selected = store.get_as_of(
        domain_id=DOMAIN,
        as_of="2026-07-12T11:30:00+00:00",
        knowledge_cutoff="2026-07-12T11:30:00+00:00",
    )

    assert selected is not None
    assert selected.snapshot_id == earlier.snapshot_id


def test_historical_analog_retrieval_prefers_similar_past_state_and_excludes_future() -> None:
    store = InMemoryWorldStateStore()
    close = _snapshot(
        "2026-06-01T12:00:00+00:00",
        economic_phase="expansion",
        ai_cycle="capex_boom",
        policy_rate=5.0,
    )
    distant = _snapshot(
        "2026-05-01T12:00:00+00:00",
        economic_phase="recession",
        ai_cycle="correction",
        policy_rate=1.0,
    )
    future = _snapshot(
        "2026-08-01T12:00:00+00:00",
        economic_phase="expansion",
        ai_cycle="capex_boom",
        policy_rate=5.25,
    )
    target = _snapshot(
        "2026-07-12T12:00:00+00:00",
        economic_phase="expansion",
        ai_cycle="capex_boom",
        policy_rate=5.25,
    )
    for item in (close, distant, future, target):
        store.append(item)

    matches = HistoricalWorldStateRetriever(store).find_analogs(target, limit=10)

    assert [item.snapshot_id for item in matches][:2] == [
        close.snapshot_id,
        distant.snapshot_id,
    ]
    assert future.snapshot_id not in {item.snapshot_id for item in matches}
    assert target.snapshot_id not in {item.snapshot_id for item in matches}
    assert matches[0].similarity_score > matches[1].similarity_score
    assert matches[0].review_only is True
    assert matches[0].eligible_for_learning is False
    assert not any("credit_phase" in key for key in matches[0].matched_context_dimensions)


def test_parent_snapshot_can_form_immutable_version_chain(tmp_path: Path) -> None:
    store = SQLiteWorldStateStore(tmp_path / "world_states.sqlite3")
    first = _snapshot("2026-07-10T12:00:00+00:00")
    store.append(first)
    second = _snapshot(
        "2026-07-11T12:00:00+00:00",
        parent_snapshot_id=first.snapshot_id,
        policy_rate=5.3,
    )
    store.append(second)

    loaded = store.get(second.snapshot_id)
    assert loaded is not None
    assert loaded.parent_snapshot_id == first.snapshot_id
    assert store.get(first.snapshot_id) == first


def test_minimal_system_retrieves_prior_world_state_before_scenario_build(
    tmp_path: Path,
) -> None:
    import asyncio

    from dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system import create_minimal_system
    from dean_os.schemas import MarketContext

    system = create_minimal_system(
        project_root=tmp_path,
        domain_id=DOMAIN,
        pipeline_enabled=False,
        save_world_model_artifacts=False,
        historical_analog_limit=3,
    )

    first_context = MarketContext(
        as_of="2026-07-10T12:00:00+00:00",
        metadata={
            "regime_dimensions": {
                "economic_phase": {"state": "expansion", "confidence": 0.7},
                "ai_cycle": {"state": "capex_boom", "confidence": 0.8},
            }
        },
    )
    first = asyncio.run(system.run(first_context))
    assert first.world_state_persistence["status"] == "stored"

    second_context = MarketContext(
        as_of="2026-07-12T12:00:00+00:00",
        news=[
            {
                "title": "TSMC expands advanced packaging capacity",
                "summary": "CoWoS capacity expands as AI accelerator demand grows.",
                "published_at": "2026-07-11T08:00:00+00:00",
                "url": "https://example.test/cowos",
                "_dean_semantic_evidence": {
                    "evidence_type": "supply",
                    "matched_terms": ["advanced packaging", "cowos"],
                    "required_lane_eligible": True,
                    "source_tier": "tier_2_strong_context",
                    "source_identity": "example_test",
                    "candidate_sha256": "cowos-fixture",
                },
            }
        ],
        metadata={
            "regime_dimensions": {
                "economic_phase": {"state": "expansion", "confidence": 0.7},
                "ai_cycle": {"state": "capex_boom", "confidence": 0.8},
            }
        },
    )
    second = asyncio.run(system.run(second_context))

    assert second.status == "completed"
    assert len(second.historical_world_state_analogs) == 1
    assert second.historical_world_state_analogs[0]["snapshot_id"] == (
        first.world_state_snapshot["snapshot_id"]
    )
    assert second.world_model_event_learning["summary"][
        "historical_world_state_analog_count"
    ] == 1
    assert second.world_model_event_learning["historical_world_state_analogs"][0][
        "snapshot_id"
    ] == first.world_state_snapshot["snapshot_id"]
    assert second.world_state_snapshot["parent_snapshot_id"] == (
        first.world_state_snapshot["snapshot_id"]
    )


def test_knowledge_cutoff_limits_sources_not_decision_time_derivations() -> None:
    as_of = "2026-07-12T12:00:00+00:00"
    cutoff = "2026-07-11T12:00:00+00:00"

    snapshot = _snapshot(
        as_of,
        knowledge_cutoff=cutoff,
        available_at="2026-07-11T11:59:00+00:00",
    )

    assert snapshot.as_of == as_of
    assert snapshot.knowledge_cutoff == cutoff
    assert snapshot.scenario_outcome_graph is not None
    assert snapshot.integrity.leakage_audit.point_in_time_valid is True


def test_world_model_filters_news_after_knowledge_cutoff() -> None:
    from dean_os.schemas import MarketContext
    from dean_os.world_model.world_model_event_learning import WorldModelEventLearningPacket

    context = MarketContext(
        as_of="2026-07-12T12:00:00+00:00",
        news=[
            {
                "title": "Packaging capacity update published after cutoff",
                "summary": "Advanced packaging capacity expands.",
                "published_at": "2026-07-11T13:00:00+00:00",
                "url": "https://example.test/after-cutoff",
                "_dean_semantic_evidence": {
                    "evidence_type": "supply",
                    "matched_terms": ["advanced packaging"],
                    "required_lane_eligible": True,
                    "source_tier": "tier_2_strong_context",
                    "source_identity": "example_test",
                    "candidate_sha256": "after-cutoff-fixture",
                },
            }
        ],
        metadata={"knowledge_cutoff": "2026-07-11T12:00:00+00:00"},
    )

    payload = WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN,
        save=False,
    )

    assert payload["inputs"]["knowledge_cutoff"] == "2026-07-11T12:00:00+00:00"
    assert payload["summary"]["accepted_evidence_count"] == 0
    assert payload["summary"]["classified_event_count"] == 0
    assert payload["summary"]["packet_status"] == "blocked_no_point_in_time_event_evidence"
