from __future__ import annotations

import json
from pathlib import Path

from dean_os.prospective_accumulation_schedule import ProspectiveAccumulationScheduleBuilder


def test_schedule_marks_stale_lanes_but_does_not_execute(tmp_path: Path) -> None:
    runbook = tmp_path / "runbook.json"
    runbook.write_text(
        json.dumps(
            {
                "contract": "dean_prospective_accumulation_runbook_v1",
                "collection_lanes": [
                    {
                        "lane_id": "clean_market_15m_60m_1d",
                        "artifact_created_at": "2026-07-10T00:00:00+00:00",
                        "runner_exists": True,
                        "command_executable": True,
                        "command": "market command",
                    },
                    {
                        "lane_id": "sector_market_evidence",
                        "artifact_created_at": "2026-07-10T00:00:00+00:00",
                        "runner_exists": True,
                        "command_executable": True,
                        "command": "sector command",
                    },
                    {
                        "lane_id": "semiconductor_news",
                        "artifact_created_at": "2026-07-11T18:00:00+00:00",
                        "runner_exists": True,
                        "command_executable": True,
                        "command": "news command",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = ProspectiveAccumulationScheduleBuilder(tmp_path / "out").build(
        runbook, as_of="2026-07-12T12:00:00+00:00", save=False
    )
    assert payload["summary"]["due_lane_count"] == 2
    assert payload["summary"]["authorization_request_count"] == 1
    assert payload["authorization_requests"][0]["lane_id"] == "clean_market_15m_60m_1d"
    sector = next(item for item in payload["lanes"] if item["lane_id"] == "sector_market_evidence")
    assert sector["dependency_blocked_by"] == ["clean_market_15m_60m_1d"]
    assert payload["safety"]["command_execution_performed"] is False


def test_missing_artifact_requests_review_not_execution(tmp_path: Path) -> None:
    runbook = tmp_path / "runbook.json"
    runbook.write_text(
        json.dumps(
            {
                "contract": "dean_prospective_accumulation_runbook_v1",
                "collection_lanes": [
                    {
                        "lane_id": "macro_context",
                        "artifact_created_at": None,
                        "runner_exists": True,
                        "command_executable": True,
                        "command": "macro command",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = ProspectiveAccumulationScheduleBuilder(tmp_path / "out").build(
        runbook, as_of="2026-07-12T12:00:00+00:00", save=False
    )
    assert payload["authorization_requests"][0]["reason"] == "artifact_missing"
    assert payload["authorization_requests"][0]["approved"] is False
    assert payload["summary"]["automatic_execution_allowed"] is False
