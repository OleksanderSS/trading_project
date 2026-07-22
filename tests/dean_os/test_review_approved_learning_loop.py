from __future__ import annotations

import json

from dean_os.learning import LearningStore
from dean_os.review_actions import ReviewActionStore
from dean_os.review_approved_learning_loop import ReviewApprovedLearningLoop
from dean_os.schemas import AgentLabRunReport, ResearchNote


def _write_profile_run(tmp_path, run_id: str = "lab_run_1") -> tuple[str, str]:
    report_dir = tmp_path / "agent_lab"
    report_dir.mkdir()
    note = ResearchNote(
        note_id="note_1",
        agent_name="specialist_research",
        topic="semiconductor cycle",
        thesis="AI demand creates a constructive semiconductor setup.",
        patterns=["ai_compute_cycle"],
        tailwinds=["data center demand"],
        tickers=["NVDA"],
        sectors=["semiconductor"],
        horizon_days=180,
        confidence=0.72,
        data_quality="partial",
    )
    report = AgentLabRunReport(
        run_id=run_id,
        corpus_path=str(tmp_path / "corpus.sqlite"),
        document_count=1,
        chunk_count=1,
        note_count=1,
        research_notes=[note],
        summary={"context_tags": ["ai_cycle"], "regime_tags": ["rising_market"]},
    )
    report_path = report_dir / f"{run_id}.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "run_id": "profile_run_1",
                "mode": "analyst_profile_orchestrator",
                "evidence_pack": {"run_id": "pack_1", "path": "pack.json"},
                "profile_runs": [
                    {
                        "profile": "generalist_base_analyst",
                        "status": "completed",
                        "runner": "agent_lab",
                        "agent_lab_run_id": run_id,
                        "report_json": str(report_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(profile_path), run_id


def test_review_loop_preview_blocks_unreviewed_source(tmp_path):
    profile_path, _ = _write_profile_run(tmp_path)

    payload = ReviewApprovedLearningLoop(tmp_path / "loop").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        apply=False,
    )

    assert payload["loop_gate"]["status"] == "blocked"
    assert payload["review_actions"] == []
    assert payload["final_bridge"]["promotion_gate"]["promoted_count"] == 0
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []


def test_review_loop_marks_reviewed_without_learning_write(tmp_path):
    profile_path, run_id = _write_profile_run(tmp_path)

    payload = ReviewApprovedLearningLoop(tmp_path / "loop").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        mark_reviewed=True,
        review_notes="Reviewed citations and accepted for pending outcome tracking.",
        apply=False,
    )

    actions = ReviewActionStore(tmp_path / "review.sqlite", event_log_path=None).list_actions()
    assert payload["loop_gate"]["status"] == "reviewed_ready_to_apply"
    assert payload["final_bridge"]["promotion_gate"]["promotable_count"] == 1
    assert len(actions) == 1
    assert actions[0].source_id == run_id
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []


def test_review_loop_mark_reviewed_and_apply_promotes_learning(tmp_path):
    profile_path, run_id = _write_profile_run(tmp_path)

    payload = ReviewApprovedLearningLoop(tmp_path / "loop").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        mark_reviewed=True,
        review_notes="Reviewed citations and accepted for pending outcome tracking.",
        apply=True,
    )
    records = LearningStore(tmp_path / "learning.sqlite").list_records()

    assert payload["loop_gate"]["status"] == "applied"
    assert payload["final_bridge"]["promotion_gate"]["promoted_count"] == 1
    assert len(records) == 1
    assert records[0].note_id == "note_1"
    assert records[0].metadata["source_id"] == run_id
    assert records[0].metadata["evidence_pack_run_id"] == "pack_1"
    assert records[0].metadata["review_action_ids"]
    assert payload["context_performance"]["overall"]["record_count"] == 1


def test_review_loop_needs_more_data_blocks_apply(tmp_path):
    profile_path, _ = _write_profile_run(tmp_path)

    payload = ReviewApprovedLearningLoop(tmp_path / "loop").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        needs_more_data_request="Add filings or transcript evidence before learning promotion.",
        review_notes="Current source is too thin.",
        apply=True,
    )

    actions = ReviewActionStore(tmp_path / "review.sqlite", event_log_path=None).list_actions()
    assert payload["loop_gate"]["status"] == "needs_more_data_recorded"
    assert payload["final_bridge"]["promotion_gate"]["status"] == "blocked"
    assert actions[0].action_type == "needs_more_data"
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []
