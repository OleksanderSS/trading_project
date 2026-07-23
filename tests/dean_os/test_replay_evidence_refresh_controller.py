from __future__ import annotations

import asyncio
import json

from dean_os.replays.replay_evidence_refresh_controller import (
    ReplayEvidenceRefreshController,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifacts(tmp_path):
    lifecycle = tmp_path / "lifecycle.json"
    registration = tmp_path / "registration.json"
    gate = tmp_path / "gate.json"
    task_id = "task_60"
    _write(
        lifecycle,
        {
            "contract": "dean_replay_outcome_lifecycle_v1",
            "system_recommendations": [
                {
                    "action_type": "refresh_verified_checkpoint_evidence",
                    "task_id": task_id,
                }
            ],
        },
    )
    _write(
        registration,
        {
            "registration_plan": [
                {
                    "task_id": task_id,
                    "hypothesis_id": "h1",
                    "due_at": "2026-07-13T20:13:54+00:00",
                }
            ]
        },
    )
    _write(
        gate,
        {
            "hypothesis_review": [
                {
                    "hypothesis_id": "h1",
                    "measurement_spec": {
                        "target_metrics": [
                            "amat_or_equipment_basket_relative_total_return"
                        ]
                    },
                }
            ]
        },
    )
    return lifecycle, registration, gate


def test_refresh_controller_builds_allowlisted_job_without_network(tmp_path):
    lifecycle, registration, gate = _artifacts(tmp_path)
    payload = asyncio.run(
        ReplayEvidenceRefreshController(tmp_path / "out").build(
            lifecycle_json=lifecycle,
            registration_json=registration,
            review_gate_json=gate,
            as_of="2026-07-14T10:00:00+00:00",
            pipeline_paths=[],
            prior_outcome_json_paths=[],
            apply_refresh=False,
            save=False,
        )
    )

    assert payload["summary"]["status"] == "not_requested"
    assert payload["summary"]["refresh_job_count"] == 1
    assert payload["refresh_jobs"][0]["tickers"] == ["AMAT"]
    assert payload["summary"]["refresh_executed"] is False
    assert payload["safety"]["network_access_performed"] is False


def test_apply_executes_one_snapshot_and_one_lifecycle_rerun(tmp_path, monkeypatch):
    lifecycle, registration, gate = _artifacts(tmp_path)
    snapshot_path = tmp_path / "snapshot.parquet"
    snapshot_path.write_bytes(b"snapshot")
    calls = {"snapshot": 0, "lifecycle": 0}

    async def fake_snapshot(self, **kwargs):
        calls["snapshot"] += 1
        assert kwargs["tickers"] == ["AMAT"]
        assert kwargs["timeframes"] == ["1d"]
        return {
            "run_id": "snapshot_1",
            "contract": "dean_clean_yahoo_market_snapshot_v1",
            "summary": {"status": "clean_market_snapshot_validated"},
            "snapshot": {"path": str(snapshot_path)},
        }

    def fake_lifecycle(self, **kwargs):
        calls["lifecycle"] += 1
        assert kwargs["verified_price_paths"] == [str(snapshot_path)]
        return {
            "run_id": "life_2",
            "contract": "dean_replay_outcome_lifecycle_v1",
            "summary": {"status": "intermediate_checkpoint_packet_recorded"},
        }

    monkeypatch.setattr(
        "dean_os.replays.replay_evidence_refresh_controller.CleanYahooMarketSnapshot.build",
        fake_snapshot,
    )
    monkeypatch.setattr(
        "dean_os.replays.replay_evidence_refresh_controller.ReplayOutcomeLifecycleOrchestrator.build",
        fake_lifecycle,
    )
    payload = asyncio.run(
        ReplayEvidenceRefreshController(tmp_path / "out").build(
            lifecycle_json=lifecycle,
            registration_json=registration,
            review_gate_json=gate,
            as_of="2026-07-14T10:00:00+00:00",
            pipeline_paths=[],
            prior_outcome_json_paths=[],
            apply_refresh=True,
            save=False,
        )
    )

    assert calls == {"snapshot": 1, "lifecycle": 1}
    assert payload["summary"]["status"] == "single_refresh_pass_completed"
    assert payload["summary"]["post_refresh_lifecycle_status"] == (
        "intermediate_checkpoint_packet_recorded"
    )
    assert payload["summary"]["automatic_looping_allowed"] is False
