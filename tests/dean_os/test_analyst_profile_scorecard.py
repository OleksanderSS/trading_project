from __future__ import annotations

import json

from dean_os.analyst_core.analyst_profile_scorecard import AnalystProfileScorecard


def test_analyst_profile_scorecard_scores_saved_profile_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "run1.json").write_text(
        json.dumps(
            {
                "run_id": "run1",
                "mode": "analyst_profile_orchestrator",
                "profile_plan": {"skipped_profiles": [{"profile": "news_catalyst", "reason": "Needs permission."}]},
                "profile_runs": [
                    {
                        "profile": "generalist_base_analyst",
                        "status": "completed",
                        "runner": "agent_lab",
                        "note_count": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = AnalystProfileScorecard(output_dir=tmp_path / "scorecard").build(
        profile_runs_dir=runs_dir,
        min_completed_runs=1,
        min_avg_confidence=0.3,
        min_avg_citations=1.0,
    )

    base = payload["profiles"]["generalist_base_analyst"]
    skipped = payload["profiles"]["news_catalyst"]
    assert base["activation_status"] == "ready_to_activate"
    assert base["completed_count"] == 1
    assert skipped["activation_status"] == "blocked"
    assert skipped["skipped_count"] == 1
    assert (tmp_path / "scorecard" / "latest.json").exists()


def test_analyst_profile_scorecard_handles_empty_directory(tmp_path):
    payload = AnalystProfileScorecard(output_dir=tmp_path / "scorecard").build(
        profile_runs_dir=tmp_path / "missing",
    )

    assert payload["summary"]["orchestrator_run_count"] == 0
    assert payload["profiles"] == {}
    assert payload["recommendations"] == ["Run AnalystProfileOrchestrator before building scorecards."]

