from __future__ import annotations

import pytest

from dean_os.learning import LearningStore
from dean_os.recommendation_memory import RecommendationMemoryStore
from dean_os.schemas import AgentLearningRecord


def test_draft_recommendation_memory_is_not_retrievable(tmp_path):
    store = RecommendationMemoryStore(tmp_path / "memory.sqlite")
    draft = store.add_case(
        source_type="manual_case",
        source_id="case_1",
        agent_name="analyst",
        topic="capex",
        thesis="Unreviewed capex conclusion",
        expected_direction="bullish",
        context_tags=["capex"],
    )

    assert draft.lifecycle_status == "draft"
    assert store.relevant_records(context_tags=["capex"]) == []

    validated = store.transition_lifecycle(
        draft.memory_id,
        "validated",
        actor="human:reviewer",
        reason="Source and reasoning reviewed.",
    )

    assert validated.lifecycle_status == "validated"
    assert [item.memory_id for item in store.relevant_records(context_tags=["capex"])] == [
        draft.memory_id
    ]


def test_rejected_and_superseded_memory_cannot_be_retrieved(tmp_path):
    store = RecommendationMemoryStore(tmp_path / "memory.sqlite")
    record = store.add_case(
        source_type="manual_case",
        source_id="case_2",
        agent_name="analyst",
        topic="supply",
        thesis="Supply conclusion",
        expected_direction="neutral",
        context_tags=["supply"],
        lifecycle_status="validated",
        lifecycle_actor="human:reviewer",
        lifecycle_reason="Initial review.",
    )
    store.transition_lifecycle(
        record.memory_id,
        "superseded",
        actor="human:reviewer",
        reason="Replaced by corrected case.",
    )

    assert store.relevant_records(context_tags=["supply"]) == []
    with pytest.raises(ValueError, match="Invalid memory lifecycle transition"):
        store.transition_lifecycle(
            record.memory_id,
            "validated",
            actor="human:reviewer",
            reason="Unsafe resurrection.",
        )


def test_learning_score_ignores_draft_records(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite")
    draft = AgentLearningRecord(
        agent_name="analyst",
        note_id="n1",
        expected_direction="bullish",
        horizon_days=30,
        outcome_label="hit",
    )
    store.add_record(draft)

    blocked_score = store.score_agent("analyst")
    assert blocked_score["total_record_count"] == 1
    assert blocked_score["eligible_record_count"] == 0
    assert blocked_score["lifecycle_excluded_count"] == 1

    store.transition_lifecycle(
        draft.record_id,
        "human-corrected",
        actor="human:reviewer",
        reason="Corrected and reviewed record.",
    )
    score = store.score_agent("analyst")
    assert score["total_record_count"] == 1
    assert score["hit_rate"] == 1.0
