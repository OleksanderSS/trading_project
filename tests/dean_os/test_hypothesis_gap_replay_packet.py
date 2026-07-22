from __future__ import annotations

import hashlib
import json

from dean_os.hypothesis_gap_replay_packet import HypothesisGapReplayPacketBridge
from dean_os.world_model_replay_review_gate import WorldModelReplayReviewGate


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gap_replay_packet_enters_manual_gate_without_approval(tmp_path):
    hypothesis = {
        "hypothesis_id": "h1",
        "hypothesis": "Capex cycle will persist",
        "confidence": 0.4,
        "horizons_to_check": [30, 90],
        "expected_observations": ["orders rise"],
        "invalidation_signals": ["capex cut"],
    }
    analyst = _write(
        tmp_path / "analyst.json",
        {
            "contract": "dean_domain_analyst_review_run_v1",
            "domain_id": "semiconductor_ai_infrastructure",
            "inputs": {"as_of": "2026-07-01T00:00:00+00:00"},
            "agent_report": {
                "metrics_snapshot": {
                    "evidence_count": 10,
                    "hypotheses": [hypothesis],
                }
            },
            "safety": {"review_only": True},
        },
    )
    gap = _write(
        tmp_path / "gap.json",
        {
            "contract": "dean_hypothesis_evidence_gap_review_v1",
            "status": "hypothesis_gaps_reviewed_manual_action_required",
            "inputs": {
                "analyst_review": {"path": str(analyst), "sha256": _sha(analyst)}
            },
            "gap_reviews": [{"gap_id": "g1", "resolution_status": "missing"}],
            "replay_task_candidates": [
                {"hypothesis_id": "h1", "linked_gap_ids": ["g1"]}
            ],
            "safety": {
                "review_only": True,
                "replay_task_registration_performed": False,
            },
        },
    )

    packet = HypothesisGapReplayPacketBridge(tmp_path / "packet").build(
        gap, save=False
    )
    gate = WorldModelReplayReviewGate().build(packet, save=False)

    assert packet["summary"]["replay_task_count"] == 2
    assert all(task["manual_review_gate_required"] for task in packet["replay_tasks"])
    assert gate["summary"]["gate_status"] == (
        "manual_review_required_for_replay_registration"
    )
    assert gate["summary"]["can_register_replay_tasks"] is False
    assert gate["registration_bundle"] is None


def test_gap_replay_packet_uses_hash_bound_pipeline_and_regime_context(tmp_path):
    pipeline = _write(
        tmp_path / "pipeline.json",
        {
            "contract": "dean_world_model_pipeline_context_v1",
            "pipeline_context": {
                "status": "pipeline_context_bundle_ready",
                "timeframe_lane_status": {
                    "15m": "exact",
                    "60m": "exact",
                    "1d": "exact",
                },
            },
            "indicator_state_grid": {
                "status": "indicator_state_grid_ready_with_gaps",
                "metrics": {"stage3_shard_count": 12, "stage4_exact_context_count": 3},
                "context_tags": ["pipeline_lane_15m_exact_context"],
            },
        },
    )
    hypothesis = {
        "hypothesis_id": "h1",
        "horizons_to_check": [30],
        "expected_observations": [],
        "invalidation_signals": [],
    }
    analyst = _write(
        tmp_path / "analyst.json",
        {
            "contract": "dean_domain_analyst_review_run_v1",
            "domain_id": "semiconductor_ai_infrastructure",
            "inputs": {
                "as_of": "2026-07-01T00:00:00+00:00",
                "pipeline_context_artifact": {
                    "path": str(pipeline),
                    "sha256": _sha(pipeline),
                },
            },
            "agent_report": {
                "metrics_snapshot": {
                    "hypotheses": [hypothesis],
                    "regime_context": {
                        "confidence": "medium",
                        "dimensions": {
                            "market_state": {"state": "sector_rotation_signal"}
                        },
                    },
                }
            },
            "safety": {"review_only": True},
        },
    )
    gap = _write(
        tmp_path / "gap.json",
        {
            "contract": "dean_hypothesis_evidence_gap_review_v1",
            "status": "hypothesis_gaps_reviewed_manual_action_required",
            "inputs": {"analyst_review": {"path": str(analyst), "sha256": _sha(analyst)}},
            "gap_reviews": [],
            "replay_task_candidates": [{"hypothesis_id": "h1", "linked_gap_ids": []}],
            "safety": {"review_only": True, "replay_task_registration_performed": False},
        },
    )

    packet = HypothesisGapReplayPacketBridge(tmp_path / "packet").build(gap, save=False)
    snapshot = packet["replay_tasks"][0]["pipeline_context_snapshot"]

    assert snapshot["indicator_metric_count"] == 2
    assert snapshot["pipeline_metrics"]["stage3_shard_count"] == 12
    assert snapshot["regime_label"] == "sector_rotation_signal"
    assert snapshot["timeframe_lane_status"]["1d"] == "exact"
    assert packet["summary"]["indicator_metric_count"] == 2
