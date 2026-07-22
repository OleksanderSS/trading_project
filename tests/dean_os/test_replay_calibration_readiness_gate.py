from __future__ import annotations

import json

from dean_os.replay_calibration_readiness_gate import ReplayCalibrationReadinessGate


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _repair_report(path):
    return _write_json(
        path,
        {
            "mode": "replay_price_artifact_repair",
            "summary": {"repair_status": "candidate_artifact_written", "candidate_quality_warning_count": 0},
            "artifact": {"path": "data/repaired.parquet"},
        },
    )


def _price_quality_report(path, status="clear", warnings=0):
    return _write_json(
        path,
        {
            "mode": "replay_price_quality_investigation",
            "summary": {
                "investigation_status": status,
                "warning_record_count": warnings,
                "extreme_benchmark_warning_count": 0,
            },
            "artifact_diagnostics": [{"path": "data/repaired.parquet", "warnings": []}],
        },
    )


def _replay_batch(path, clear_runs=12, quality_blocked=0):
    return _write_json(
        path,
        {
            "mode": "historical_replay_batch",
            "summary": {
                "total_runs": clear_runs + quality_blocked,
                "evaluated_runs": clear_runs + quality_blocked,
                "clear_evaluated_runs": clear_runs,
                "quality_blocked_runs": quality_blocked,
                "clear_hit_rate": 0.58,
                "clear_average_realized_return": 0.04,
            },
        },
    )


def _research_batch(path, clear_runs=12, weak=0, directional=6):
    runs = []
    for index in range(clear_runs):
        expected = "bullish" if index < directional else "neutral"
        runs.append(
            {
                "research_expected_direction": expected,
                "research_price_agreement": "confirmed" if index < directional else "research_inconclusive",
            }
        )
    return _write_json(
        path,
        {
            "mode": "historical_research_replay_batch",
            "summary": {
                "total_runs": clear_runs,
                "evaluated_runs": clear_runs,
                "clear_evaluated_runs": clear_runs,
                "quality_blocked_runs": 0,
                "weak_evidence_runs": weak,
                "research_inconclusive_runs": clear_runs - directional,
                "evidence_quality_counts": {"strong": clear_runs - weak, "partial": weak},
                "research_stance_counts": {"bullish": directional, "mixed": clear_runs - directional},
                "clear_hit_rate": 0.58,
            },
            "runs": runs,
        },
    )


def _gate(tmp_path, **overrides):
    paths = {
        "repair_report_path": _repair_report(tmp_path / "repair.json"),
        "price_quality_report_path": _price_quality_report(tmp_path / "quality.json"),
        "replay_batch_path": _replay_batch(tmp_path / "replay.json"),
        "research_batch_path": _research_batch(tmp_path / "research.json"),
    }
    paths.update(overrides)
    return ReplayCalibrationReadinessGate(tmp_path / "reports").build(save=False, **paths)


def test_replay_calibration_readiness_blocks_dirty_price_quality(tmp_path):
    payload = _gate(
        tmp_path,
        price_quality_report_path=_price_quality_report(tmp_path / "quality_dirty.json", status="blocked_price_quality", warnings=1),
    )

    assert payload["summary"]["readiness_status"] == "price_quality_blocked"
    assert payload["summary"]["can_create_calibration_review_packet"] is False
    assert payload["gate"]["blockers"][0]["check"] == "price_quality"


def test_replay_calibration_readiness_requires_more_clean_replay_samples(tmp_path):
    payload = _gate(tmp_path, replay_batch_path=_replay_batch(tmp_path / "replay_small.json", clear_runs=2))

    assert payload["summary"]["readiness_status"] == "need_more_replay_samples"
    assert payload["checks"]["price_quality"]["status"] == "pass"
    assert payload["checks"]["replay_sample"]["status"] == "blocked"


def test_replay_calibration_readiness_blocks_weak_evidence_after_sample_is_large(tmp_path):
    payload = _gate(tmp_path, research_batch_path=_research_batch(tmp_path / "research_weak.json", clear_runs=12, weak=1))

    assert payload["summary"]["readiness_status"] == "need_evidence_backfill"
    assert payload["checks"]["research_sample"]["status"] == "pass"
    assert payload["checks"]["evidence_coverage"]["status"] == "blocked"


def test_replay_calibration_readiness_allows_manual_review_with_directionality_caution(tmp_path):
    payload = _gate(tmp_path, research_batch_path=_research_batch(tmp_path / "research_neutral.json", clear_runs=12, weak=0, directional=0))

    assert payload["summary"]["readiness_status"] == "ready_for_manual_review_with_caution"
    assert payload["summary"]["can_create_calibration_review_packet"] is True
    assert payload["checks"]["research_directionality"]["status"] == "caution"
    assert payload["summary"]["can_change_analyst_weights"] is False


def test_replay_calibration_readiness_ready_for_manual_review(tmp_path):
    payload = _gate(tmp_path)

    assert payload["summary"]["readiness_status"] == "ready_for_manual_review"
    assert payload["summary"]["can_create_calibration_review_packet"] is True
    assert payload["gate"]["blockers"] == []
    assert payload["safety"]["learning_write_performed"] is False
