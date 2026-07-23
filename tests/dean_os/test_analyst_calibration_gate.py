from __future__ import annotations

import json

from dean_os.analyst_core.analyst_calibration_gate import AnalystCalibrationGate
from dean_os.learning import LearningStore
from dean_os.schemas import AgentLearningRecord


def _write_scorecard(tmp_path, status: str = "ready_to_activate") -> str:
    path = tmp_path / "scorecard.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "scorecard_1",
                "mode": "analyst_profile_scorecard",
                "summary": {"profile_count": 1},
                "profiles": {
                    "generalist_base_analyst": {
                        "profile": "generalist_base_analyst",
                        "completed_count": 3,
                        "skipped_count": 0,
                        "avg_confidence": 0.65,
                        "avg_citations": 2.0,
                        "activation_status": status,
                        "blockers": [] if status == "ready_to_activate" else ["Needs more runs."],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _add_record(tmp_path, record_id: str, label: str, realized_return: float) -> None:
    store = LearningStore(tmp_path / "learning.sqlite")
    store.add_record(
        AgentLearningRecord(
            record_id=record_id,
            agent_name="generalist_base_analyst",
            note_id=f"note_{record_id}",
            expected_direction="bullish",
            horizon_days=30,
            created_at="2026-01-01T00:00:00+00:00",
            outcome_at="2026-02-01T00:00:00+00:00",
            realized_return=realized_return,
            outcome_label=label,
            metadata={
                "analyst_learning_bridge": True,
                "profile": "generalist_base_analyst",
                "topic": "ai cycle",
                "tickers": ["AMD"],
                "context_tags": ["ai_cycle"],
            },
        )
    )


def test_analyst_calibration_gate_marks_profile_ready_for_review(tmp_path):
    scorecard = _write_scorecard(tmp_path)
    _add_record(tmp_path, "r1", "hit", 0.12)
    _add_record(tmp_path, "r2", "hit", 0.08)
    _add_record(tmp_path, "r3", "miss", -0.03)

    payload = AnalystCalibrationGate(tmp_path / "gate").run(
        profile_scorecard_path=scorecard,
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        save=False,
    )

    card = payload["profiles"]["generalist_base_analyst"]
    assert card["calibration_status"] == "ready_for_review"
    assert card["suggested_weight_delta"] == 0.05
    assert payload["summary"]["ready_for_review_profiles"] == ["generalist_base_analyst"]


def test_analyst_calibration_gate_blocks_without_completed_outcomes(tmp_path):
    scorecard = _write_scorecard(tmp_path)

    payload = AnalystCalibrationGate(tmp_path / "gate").run(
        profile_scorecard_path=scorecard,
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        save=False,
    )

    card = payload["profiles"]["generalist_base_analyst"]
    assert card["calibration_status"] == "blocked"
    assert any("completed outcomes" in blocker for blocker in card["blockers"])


def test_analyst_calibration_gate_keeps_scorecard_candidate(tmp_path):
    scorecard = _write_scorecard(tmp_path, status="keep_candidate")
    _add_record(tmp_path, "r1", "hit", 0.12)
    _add_record(tmp_path, "r2", "hit", 0.08)
    _add_record(tmp_path, "r3", "hit", 0.04)

    payload = AnalystCalibrationGate(tmp_path / "gate").run(
        profile_scorecard_path=scorecard,
        learning_path=tmp_path / "learning.sqlite",
        memory_path=tmp_path / "memory.sqlite",
        save=False,
    )

    card = payload["profiles"]["generalist_base_analyst"]
    assert card["calibration_status"] == "keep_candidate"
    assert "Profile scorecard is not ready_to_activate." in card["blockers"]
