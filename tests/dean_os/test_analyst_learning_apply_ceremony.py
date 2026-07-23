from __future__ import annotations

import json

from dean_os.analyst_core.analyst_learning_apply_ceremony import AnalystLearningApplyCeremony
from dean_os.analyst_core.analyst_learning_promotion_bridge import AnalystLearningPromotionBridge
from dean_os.learning import LearningStore
from dean_os.review_actions import ReviewActionStore
from dean_os.schemas import AgentLabRunReport, ResearchNote


def _write_profile_run(tmp_path, run_id: str = "lab_run_1") -> tuple[str, str]:
    report_dir = tmp_path / "agent_lab"
    report_dir.mkdir()
    notes = [
        ResearchNote(
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
        ),
        ResearchNote(
            note_id="note_2",
            agent_name="evidence_synthesis",
            topic="ai cycle synthesis",
            thesis="Cited evidence supports the AI compute cycle.",
            patterns=["ai_compute_cycle"],
            tailwinds=["ai_compute_cycle"],
            tickers=["AMD", "NVDA"],
            sectors=["semiconductor"],
            horizon_days=365,
            confidence=0.8,
            data_quality="strong",
        ),
    ]
    report = AgentLabRunReport(
        run_id=run_id,
        corpus_path=str(tmp_path / "corpus.sqlite"),
        document_count=2,
        chunk_count=2,
        note_count=2,
        research_notes=notes,
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


def _bridge_dry_run(tmp_path) -> str:
    profile_path, run_id = _write_profile_run(tmp_path)
    ReviewActionStore(tmp_path / "review.sqlite", event_log_path=None).mark_reviewed(
        source_type="agent_lab_report",
        source_id=run_id,
        notes="Reviewed",
    )
    payload = AnalystLearningPromotionBridge(tmp_path / "bridge").run(
        profile_run_path=profile_path,
        learning_path=tmp_path / "learning.sqlite",
        review_actions_path=tmp_path / "review.sqlite",
        operations_path=tmp_path / "operation_queue.sqlite",
        apply=False,
    )
    return payload["saved_paths"]["latest_json"]


def test_learning_apply_ceremony_requires_explicit_apply_flag(tmp_path):
    bridge_path = _bridge_dry_run(tmp_path)

    payload = AnalystLearningApplyCeremony(tmp_path / "ceremony").apply(
        bridge_dry_run_path=bridge_path,
        apply_learning=False,
        save=False,
    )

    assert payload["summary"]["apply_status"] == "blocked_apply_flag_required"
    assert payload["summary"]["learning_write_performed"] is False
    assert LearningStore(tmp_path / "learning.sqlite").list_records() == []


def test_learning_apply_ceremony_writes_promotable_records(tmp_path):
    bridge_path = _bridge_dry_run(tmp_path)

    payload = AnalystLearningApplyCeremony(tmp_path / "ceremony").apply(
        bridge_dry_run_path=bridge_path,
        apply_learning=True,
        save=False,
    )
    records = LearningStore(tmp_path / "learning.sqlite").list_records()

    assert payload["summary"]["apply_status"] == "applied"
    assert payload["summary"]["learning_write_performed"] is True
    assert payload["summary"]["promoted_count"] == 2
    assert len(records) == 2
    assert {record.note_id for record in records} == {"note_1", "note_2"}


def test_learning_apply_ceremony_blocks_duplicate_notes(tmp_path):
    bridge_path = _bridge_dry_run(tmp_path)
    AnalystLearningApplyCeremony(tmp_path / "ceremony").apply(
        bridge_dry_run_path=bridge_path,
        apply_learning=True,
        save=False,
    )

    payload = AnalystLearningApplyCeremony(tmp_path / "ceremony2").apply(
        bridge_dry_run_path=bridge_path,
        apply_learning=True,
        save=False,
    )

    assert payload["summary"]["apply_status"] == "blocked_duplicate_learning_records"
    assert payload["summary"]["learning_write_performed"] is False
    assert len(LearningStore(tmp_path / "learning.sqlite").list_records()) == 2


def test_learning_apply_ceremony_blocks_non_ready_bridge(tmp_path):
    bridge_path = _bridge_dry_run(tmp_path)
    payload = json.loads(open(bridge_path, encoding="utf-8").read())
    payload["promotion_gate"]["status"] = "blocked"
    blocked_path = tmp_path / "blocked_bridge.json"
    blocked_path.write_text(json.dumps(payload), encoding="utf-8")

    result = AnalystLearningApplyCeremony(tmp_path / "ceremony").apply(
        bridge_dry_run_path=blocked_path,
        apply_learning=True,
        save=False,
    )

    assert result["summary"]["apply_status"] == "blocked_bridge_not_ready"
    assert result["summary"]["learning_write_performed"] is False
