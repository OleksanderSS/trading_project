from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dean_os.verified_local_snapshot_ingestion import (
    VerifiedLocalSnapshotIngestion,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _router(path):
    _write(
        path,
        {
            "contract": "dean_verified_market_source_router_v1",
            "routes": [
                {
                    "task_id": "task_60",
                    "hypothesis_id": "h1",
                    "due_at": "2026-07-13T20:13:54+00:00",
                    "required_tickers": ["AMAT"],
                    "selected_provider": {
                        "provider_id": "local_validated_snapshot"
                    },
                }
            ],
        },
    )


def _candidate(path):
    pd.DataFrame(
        {
            "datetime": ["2026-07-14T00:00:00+00:00"],
            "ticker": ["AMAT"],
            "close": [200.0],
        }
    ).to_csv(path, index=False)


def _builder(tmp_path):
    return VerifiedLocalSnapshotIngestion(
        output_dir=tmp_path / "report",
        artifact_dir=tmp_path / "artifacts",
    )


def _kwargs(tmp_path, router, candidate=None):
    return {
        "source_router_json": router,
        "candidate_path": candidate,
        "registration_json": tmp_path / "registration.json",
        "review_gate_json": tmp_path / "gate.json",
        "as_of": "2026-07-14T21:00:00+00:00",
        "pipeline_paths": [],
        "prior_outcome_json_paths": [],
        "save": False,
    }


def test_ingestion_waits_without_candidate_and_does_not_poll(tmp_path):
    router = tmp_path / "router.json"
    _router(router)
    payload = _builder(tmp_path).build(**_kwargs(tmp_path, router))

    assert payload["summary"]["status"] == "awaiting_candidate"
    assert payload["summary"]["automatic_source_polling_allowed"] is False
    assert payload["summary"]["snapshot_ingested"] is False


def test_valid_candidate_preview_does_not_write_snapshot(tmp_path):
    router = tmp_path / "router.json"
    candidate = tmp_path / "candidate.csv"
    _router(router)
    _candidate(candidate)
    source_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()

    payload = _builder(tmp_path).build(
        **_kwargs(tmp_path, router, candidate), apply_ingestion=False
    )

    assert payload["summary"]["status"] == "candidate_valid_ready_for_ingestion"
    assert payload["validation"]["sha256"] == source_hash
    assert payload["summary"]["snapshot_ingested"] is False
    assert not (tmp_path / "artifacts").exists()


def test_apply_writes_immutable_snapshot_and_runs_lifecycle_once(tmp_path, monkeypatch):
    router = tmp_path / "router.json"
    candidate = tmp_path / "candidate.csv"
    _router(router)
    _candidate(candidate)
    source_before = candidate.read_bytes()
    calls = {"lifecycle": 0}

    def fake_lifecycle(self, **kwargs):
        calls["lifecycle"] += 1
        snapshot = kwargs["verified_price_paths"][0]
        frame = pd.read_parquet(snapshot)
        assert frame["source_provider"].unique().tolist() == [
            "local_validated_snapshot"
        ]
        return {
            "run_id": "life_2",
            "contract": "dean_replay_outcome_lifecycle_v1",
            "summary": {"status": "intermediate_checkpoint_packet_recorded"},
        }

    monkeypatch.setattr(
        "dean_os.verified_local_snapshot_ingestion.ReplayOutcomeLifecycleOrchestrator.build",
        fake_lifecycle,
    )
    payload = _builder(tmp_path).build(
        **_kwargs(tmp_path, router, candidate), apply_ingestion=True
    )

    assert calls["lifecycle"] == 1
    assert payload["summary"]["status"] == "snapshot_ingested_lifecycle_completed"
    assert payload["summary"]["snapshot_ingested"] is True
    assert Path(payload["snapshot"]["path"]).is_file()
    assert candidate.read_bytes() == source_before
