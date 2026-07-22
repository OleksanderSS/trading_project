from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.full_system_cycle_world_model_bridge import (
    _hypothesis_alignment,
    _verified_fragment,
    verify_cycle_bindings,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verifies_cycle_manager_inputs_and_timeframe_binding(tmp_path: Path) -> None:
    manager = tmp_path / "manager.json"
    manager.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    cycle = {
        "contract": "dean_full_system_review_cycle_v1",
        "run_id": "cycle_1",
        "manager_report": {"path": str(manager), "sha256": _sha(manager)},
        "inputs": {
            "artifacts": {"news": {"path": str(source), "sha256": _sha(source)}},
            "timeframe_lane_readiness": {"path": str(readiness), "sha256": _sha(readiness)},
        },
    }
    cycle_path = tmp_path / "cycle.json"
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
    result = verify_cycle_bindings(cycle_path, cycle)
    assert result["manager_report"]["sha256"] == _sha(manager)
    assert result["timeframe_lane_readiness"]["sha256"] == _sha(readiness)


def test_rejects_changed_manager_report(tmp_path: Path) -> None:
    manager = tmp_path / "manager.json"
    manager.write_text("{}", encoding="utf-8")
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    cycle = {
        "contract": "dean_full_system_review_cycle_v1",
        "manager_report": {"path": str(manager), "sha256": _sha(manager)},
        "inputs": {
            "artifacts": {},
            "timeframe_lane_readiness": {"path": str(readiness), "sha256": _sha(readiness)},
        },
    }
    cycle_path = tmp_path / "cycle.json"
    cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
    manager.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="manager report SHA-256 mismatch"):
        verify_cycle_bindings(cycle_path, cycle)


def test_alignment_keeps_event_and_sector_horizon_families_separate() -> None:
    payload = {
        "hypotheses": [
            {
                "hypothesis_id": "world_capex",
                "hypothesis": "Capex cycle will sustain industry growth through 20 days",
                "horizon_family": "event_response_fixed_v1",
                "horizons_to_check": [1, 5, 20, 60, 120],
            }
        ],
        "upstream_domain_analysis": {
            "hypotheses": [
                {
                    "hypothesis_id": "sector_capex",
                    "hypothesis": "Capex cycle will sustain industry growth through 180 days",
                    "horizon_family": "sector_thesis_monitoring_v1",
                    "horizons_to_check": [30, 90, 180],
                },
                {
                    "hypothesis_id": "sector_supply",
                    "hypothesis": "Supply constraints will persist for 180 days",
                    "horizon_family": "sector_thesis_monitoring_v1",
                    "horizons_to_check": [30, 90, 180],
                },
            ]
        },
    }

    alignment = _hypothesis_alignment(payload)

    assert alignment["summary"]["aligned_upstream_hypothesis_count"] == 1
    assert alignment["summary"]["unaligned_upstream_hypothesis_count"] == 1
    assert alignment["summary"]["horizon_substitution_allowed"] is False
    capex = alignment["alignments"][0]
    assert capex["upstream_horizons_days"] == [30, 90, 180]
    assert capex["world_horizons_days"] == [1, 5, 20, 60, 120]


def test_fragment_may_precede_cycle_but_cannot_be_from_future(tmp_path: Path) -> None:
    path = tmp_path / "fragment.json"
    path.write_text("{}", encoding="utf-8")

    def old_loader(_: Path):
        return {"as_of": "2026-07-01T00:00:00+00:00"}

    accepted = _verified_fragment(
        old_loader,
        path,
        "2026-07-13T00:00:00+00:00",
        name="test",
    )
    assert accepted["as_of"] == "2026-07-01T00:00:00+00:00"

    def future_loader(_: Path):
        return {"as_of": "2026-07-14T00:00:00+00:00"}

    with pytest.raises(ValueError, match="future relative"):
        _verified_fragment(
            future_loader,
            path,
            "2026-07-13T00:00:00+00:00",
            name="test",
        )
