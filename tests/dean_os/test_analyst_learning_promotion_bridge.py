from __future__ import annotations

import json

from dean_os.analyst_core.analyst_learning_promotion_bridge import AnalystLearningPromotionBridge
from dean_os.learning import LearningStore
from dean_os.review_actions import ReviewActionStore
from dean_os.schemas import AgentLabRunReport, ResearchNote


def _write_report(tmp_path, run_id: str = "lab_run_1") -> tuple[str, str]:
    report_dir = tmp_path / "agent_lab"
    report_dir.mkdir()
    note = ResearchNote(
        note_id="note_1",
        agent_name="specialist_research",
        topic="ai cycle",
        thesis="AMD AI compute cycle evidence is constructive.",
        patterns=["ai_compute_cycle"],
        tailwinds=["ai_compute_cycle"],
        tickers=["AMD"],
        sectors=["semiconductor"],
        horizon_days=365,
        confidence=0.7,
        data_quality="partial",
    )
    report = AgentLabRunReport(
        run_id=run_id,
        corpus_path=str(tmp_path / "corpus.sqlite"),
        document_count=1,
        chunk_count=1,
        note_count=1,
        research_notes=[note],
        summary={"context_tags": ["ai_cycle"], "regime_tags": []},
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


def test_learning_bridge_blocks_unreviewed_sources(tmp_path):
    profile_path, _ = _write_report(tmp_path)
    payload = AnalystLearningPromotionBridge(output_dir=tmp_path / "bridge").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        apply=True,
    )

    assert payload["promotion_gate"]["status"] == "blocked"
    assert payload["promotion_gate"]["promoted_count"] == 0
    assert payload["sources"][0]["candidates"][0]["blockers"] == ["source_agent_lab_report_not_marked_reviewed"]
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []


def test_learning_bridge_dry_run_requires_apply_for_writes(tmp_path):
    profile_path, run_id = _write_report(tmp_path)
    ReviewActionStore(tmp_path / "review.sqlite", event_log_path=None).mark_reviewed(
        source_type="agent_lab_report",
        source_id=run_id,
        notes="Reviewed",
    )

    payload = AnalystLearningPromotionBridge(output_dir=tmp_path / "bridge").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        apply=False,
    )

    assert payload["promotion_gate"]["status"] == "dry_run_ready"
    assert payload["promotion_gate"]["promotable_count"] == 1
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []


def test_learning_bridge_apply_writes_reviewed_learning_record(tmp_path):
    profile_path, run_id = _write_report(tmp_path)
    ReviewActionStore(tmp_path / "review.sqlite", event_log_path=None).mark_reviewed(
        source_type="agent_lab_report",
        source_id=run_id,
        notes="Reviewed",
    )

    payload = AnalystLearningPromotionBridge(output_dir=tmp_path / "bridge").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        apply=True,
    )
    records = LearningStore(tmp_path / "learning.sqlite").list_records()

    assert payload["promotion_gate"]["status"] == "applied"
    assert payload["promotion_gate"]["promoted_count"] == 1
    assert len(records) == 1
    assert records[0].note_id == "note_1"
    assert records[0].metadata["evidence_pack_run_id"] == "pack_1"
    assert records[0].metadata["profile"] == "generalist_base_analyst"
    assert records[0].metadata["reviewed"] is True

