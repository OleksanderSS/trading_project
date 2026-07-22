from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.shadow_calibration_case_index import (
    validate_shadow_calibration_case,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.shadow_component_case_producer import (
    validate_component_case,
)
from dean_os.utils import json_ready, sha256_json

DEFAULT_SOURCES = {
    "prediction_review": (
        "reports/dean_os/"
        "pipeline_prediction_review_packet_current/latest.json"
    ),
    "specialist_context": (
        "reports/dean_os/"
        "specialist_context_review_amd_15m_current/latest.json"
    ),
    "capability_matrix": (
        "reports/dean_os/agent_capability_matrix_current/latest.json"
    ),
    "historical_case_index": (
        "reports/dean_os/"
        "shadow_calibration_case_index_current/latest.json"
    ),
}


class ShadowCalibrationReadinessPacket:
    """Review whether shadow reports have enough causal outcome cases."""

    def __init__(
        self,
        *,
        policy_path: str | Path = (
            "dean_os/config/shadow_calibration_policy.yaml"
        ),
        sources: dict[str, str | Path] | None = None,
        output_dir: str | Path = (
            "reports/dean_os/shadow_calibration_readiness_current"
        ),
    ):
        self.policy_path = Path(policy_path)
        self.sources = {
            name: Path(path)
            for name, path in (sources or DEFAULT_SOURCES).items()
        }
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        policy = yaml.safe_load(
            self.policy_path.read_text(encoding="utf-8")
        ) or {}
        artifacts = {
            name: _artifact_state(path)
            for name, path in self.sources.items()
        }
        records, invalid_case_record_count = _calibration_records(
            artifacts["historical_case_index"]
        )
        component_counts = {
            component: sum(
                record.get("component") == component
                for record in records
            )
            for component in (
                "prediction",
                "regime",
                "specialist",
                "context_synthesis",
            )
        }
        context_case_counts = _context_case_counts(records)
        case_requirements = _mapping(
            policy.get("case_requirements")
        )
        min_diagnostic = int(
            case_requirements.get(
                "diagnostic_min_cases_per_context",
                30,
            )
        )
        prediction = _prediction_readiness(
            artifacts["prediction_review"],
            component_counts["prediction"],
            _max_context_count(
                context_case_counts["prediction"]
            ),
            min_diagnostic,
        )
        specialist = _specialist_readiness(
            artifacts["specialist_context"],
            component_counts["specialist"],
            _max_context_count(
                context_case_counts["specialist"]
            ),
            min_diagnostic,
        )
        regime = _component_readiness(
            component="regime",
            case_count=component_counts["regime"],
            max_context_case_count=_max_context_count(
                context_case_counts["regime"]
            ),
            min_cases=min_diagnostic,
            additional_blockers=(
                ["no_saved_outcome_bound_regime_case_index"]
                if component_counts["regime"] == 0
                else []
            ),
        )
        synthesis = _component_readiness(
            component="context_synthesis",
            case_count=component_counts["context_synthesis"],
            max_context_case_count=_max_context_count(
                context_case_counts["context_synthesis"]
            ),
            min_cases=min_diagnostic,
            additional_blockers=(
                ["no_saved_outcome_bound_synthesis_case_index"]
                if component_counts["context_synthesis"] == 0
                else []
            ),
        )
        components = {
            "prediction": prediction,
            "regime": regime,
            "specialist": specialist,
            "context_synthesis": synthesis,
        }
        blockers = sorted(
            {
                blocker
                for component in components.values()
                for blocker in component["blockers"]
            }
        )
        if invalid_case_record_count:
            blockers.append(
                "historical_case_index_contains_invalid_records"
            )
            blockers = sorted(set(blockers))
        common_episode_counts = _common_episode_counts(records)
        diagnostic_ready_contexts = sorted(
            context_key
            for context_key, count in common_episode_counts.items()
            if count >= min_diagnostic
        )
        if (
            all(
                _max_context_count(context_case_counts[name])
                >= min_diagnostic
                for name in context_case_counts
            )
            and not diagnostic_ready_contexts
        ):
            blockers.append(
                "no_common_exact_context_meets_diagnostic_policy"
            )
            blockers = sorted(set(blockers))
        safety_counters = {
            "unsafe_output_count": 0,
            "time_leakage_count": 0,
            "sector_to_ticker_leakage_count": 0,
            "future_evidence_use_count": 0,
            "context_mismatch_accepted_count": 0,
        }
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "shadow_calibration_readiness",
            "schema_version": (
                "dean_shadow_calibration_readiness_v1"
            ),
            "status": (
                "shadow_calibration_ready_for_diagnostic_review"
                if not blockers
                else "shadow_calibration_blocked"
            ),
            "policy": {
                "path": str(self.policy_path),
                "sha256": _sha256(self.policy_path),
                "schema_version": policy.get("schema_version"),
                "case_requirements": case_requirements,
                "component_metrics": policy.get(
                    "component_metrics",
                    {},
                ),
                "safety_thresholds": policy.get(
                    "safety_thresholds",
                    {},
                ),
            },
            "source_inventory": artifacts,
            "accepted_case_record_count": len(records),
            "invalid_case_record_count": invalid_case_record_count,
            "component_readiness": components,
            "context_case_counts": context_case_counts,
            "common_episode_counts": common_episode_counts,
            "diagnostic_ready_contexts": (
                diagnostic_ready_contexts
            ),
            "blocking_gaps": blockers,
            "safety_counters": safety_counters,
            "quality_template_alignment": {
                "numeric_unit_period_completeness": (
                    "adapted_to_prediction_target_semantics"
                ),
                "time_leakage_rate": "required_zero",
                "unsafe_output_rate": "required_zero",
                "human_review_disagreement_rate": (
                    "required_before_weight_review"
                ),
                "historical_outcome_circularity": (
                    "single_stock_price_only_validation_disallowed"
                ),
            },
            "next_steps": [
                (
                    "Persist a trustworthy Stage5 prediction review "
                    "from a saved real pipeline result."
                ),
                (
                    "Capture the explicit Stage5 model-output contract "
                    "inside a trustworthy saved prediction review."
                ),
                (
                    "After each target horizon matures, bind immutable "
                    "outcome-source hashes without future leakage."
                ),
                (
                    "Chain separate regime, specialist, and synthesis "
                    "case producers on the accepted prediction case."
                ),
                (
                    "Require all component thresholds to intersect on "
                    "one exact ticker/timeframe/target/context."
                ),
                (
                    "Evaluate diagnostic metrics before any proposal "
                    "for consensus weight."
                ),
            ],
            "safety": {
                "review_only": True,
                "calibration_executed": False,
                "automatic_weight_change_allowed": False,
                "decision_influence": False,
                "can_write_learning_memory": False,
                "can_write_production_config": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
        }
        payload["readiness_fingerprint"] = sha256_json(
            {
                "policy": payload["policy"],
                "source_inventory": artifacts,
                "components": components,
                "blocking_gaps": blockers,
                "invalid_case_record_count": (
                    invalid_case_record_count
                ),
                "context_case_counts": context_case_counts,
                "common_episode_counts": common_episode_counts,
                "diagnostic_ready_contexts": (
                    diagnostic_ready_contexts
                ),
                "safety_counters": safety_counters,
            }
        )
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_shadow_calibration_readiness_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _prediction_readiness(
    artifact: dict[str, Any],
    case_count: int,
    max_context_case_count: int,
    min_cases: int,
) -> dict[str, Any]:
    blockers = []
    payload = _mapping(artifact.get("payload"))
    if not artifact.get("available"):
        blockers.append("saved_stage5_prediction_review_missing")
    elif payload.get("status") != "stage5_prediction_review_ready":
        blockers.append("stage5_prediction_review_not_ready")
    contexts = payload.get("contexts", [])
    if not any(
        _mapping(item.get("target_semantics")).get("status")
        == "target_semantics_ready"
        for item in contexts
        if isinstance(item, dict)
    ):
        blockers.append("ready_target_semantics_case_missing")
    if not any(
        _mapping(
            _mapping(item.get("target_semantics")).get(
                "calibration"
            )
        ).get("model_output_scale_known")
        for item in contexts
        if isinstance(item, dict)
    ):
        blockers.append("model_output_scale_contract_missing")
    if max_context_case_count < min_cases:
        blockers.append("prediction_outcome_case_count_below_policy")
    return {
        "status": "blocked" if blockers else "ready",
        "historical_case_count": case_count,
        "maximum_exact_context_case_count": max_context_case_count,
        "diagnostic_min_cases": min_cases,
        "blockers": sorted(set(blockers)),
        "consensus_weight_eligible": False,
    }


def _specialist_readiness(
    artifact: dict[str, Any],
    case_count: int,
    max_context_case_count: int,
    min_cases: int,
) -> dict[str, Any]:
    blockers = []
    payload = _mapping(artifact.get("payload"))
    if not artifact.get("available"):
        blockers.append("specialist_context_artifact_missing")
    else:
        safety = _mapping(payload.get("safety"))
        if not safety.get("eligible_for_exact_pipeline_context"):
            blockers.append(
                "specialist_context_not_exact_pipeline_eligible"
            )
    if max_context_case_count < min_cases:
        blockers.append("specialist_outcome_case_count_below_policy")
    return {
        "status": "blocked" if blockers else "ready",
        "historical_case_count": case_count,
        "maximum_exact_context_case_count": max_context_case_count,
        "diagnostic_min_cases": min_cases,
        "blockers": sorted(set(blockers)),
        "consensus_weight_eligible": False,
    }


def _component_readiness(
    *,
    component: str,
    case_count: int,
    max_context_case_count: int,
    min_cases: int,
    additional_blockers: list[str],
) -> dict[str, Any]:
    blockers = list(additional_blockers)
    if max_context_case_count < min_cases:
        blockers.append(
            f"{component}_outcome_case_count_below_policy"
        )
    return {
        "status": "blocked" if blockers else "ready",
        "historical_case_count": case_count,
        "maximum_exact_context_case_count": max_context_case_count,
        "diagnostic_min_cases": min_cases,
        "blockers": sorted(set(blockers)),
        "consensus_weight_eligible": False,
    }


def _context_case_counts(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts = {
        component: {}
        for component in (
            "prediction",
            "regime",
            "specialist",
            "context_synthesis",
        )
    }
    for record in records:
        component = record.get("component")
        if component not in counts:
            continue
        identity = _mapping(record.get("identity"))
        context_key = "|".join(
            str(identity.get(field) or "")
            for field in (
                "ticker",
                "timeframe",
                "target_name",
                "context_fingerprint",
            )
        )
        counts[component][context_key] = (
            counts[component].get(context_key, 0) + 1
        )
    return {
        component: dict(sorted(component_counts.items()))
        for component, component_counts in counts.items()
    }


def _max_context_count(counts: dict[str, int]) -> int:
    return max(counts.values(), default=0)


def _common_episode_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    episodes: dict[str, dict[str, set[str]]] = {}
    for record in records:
        component = str(record.get("component") or "")
        if component not in {
            "prediction",
            "regime",
            "specialist",
            "context_synthesis",
        }:
            continue
        identity = _mapping(record.get("identity"))
        context_key = "|".join(
            str(identity.get(field) or "")
            for field in (
                "ticker",
                "timeframe",
                "target_name",
                "context_fingerprint",
            )
        )
        episode_id = (
            record.get("case_id")
            if component == "prediction"
            else record.get("base_prediction_case_id")
        )
        if not episode_id:
            continue
        component_sets = episodes.setdefault(
            context_key,
            {
                name: set()
                for name in (
                    "prediction",
                    "regime",
                    "specialist",
                    "context_synthesis",
                )
            },
        )
        component_sets[component].add(str(episode_id))
    return {
        context_key: len(set.intersection(*component_sets.values()))
        for context_key, component_sets in sorted(episodes.items())
    }


def _artifact_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "available": False,
            "sha256": None,
            "mode": None,
            "status": "missing",
            "payload": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "available": False,
            "sha256": _sha256(path),
            "mode": None,
            "status": "invalid_json",
            "errors": [type(exc).__name__],
            "payload": {},
        }
    return {
        "path": str(path),
        "available": isinstance(payload, dict),
        "sha256": _sha256(path),
        "mode": payload.get("mode")
        if isinstance(payload, dict)
        else None,
        "status": payload.get("status")
        if isinstance(payload, dict)
        else "invalid_payload",
        "payload": payload if isinstance(payload, dict) else {},
    }


def _calibration_records(
    artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    payload = _mapping(artifact.get("payload"))
    if (
        payload.get("mode") != "shadow_calibration_case_index"
        or payload.get("schema_version")
        != "dean_shadow_calibration_case_index_v1"
    ):
        raw = payload.get("records", [])
        invalid_count = len(raw) if isinstance(raw, list) else 0
        return [], invalid_count
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list):
        return [], 1
    accepted = []
    for item in raw_records:
        issues = validate_shadow_calibration_case(item)
        if (
            isinstance(item, dict)
            and item.get("component") != "prediction"
        ):
            issues.extend(validate_component_case(item))
        if not issues:
            accepted.append(item)
    return accepted, len(raw_records) - len(accepted)


def render_shadow_calibration_readiness_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# DEAN-OS Shadow Calibration Readiness",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Blocking gaps: {len(payload.get('blocking_gaps', []))}",
        "- Calibration executed: False",
        "- Consensus weight eligible: False",
        "",
        "| Component | Total cases | Best exact context | Minimum | Status | Blockers |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, item in payload.get(
        "component_readiness",
        {},
    ).items():
        lines.append(
            "| {name} | {cases} | {context_cases} | {minimum} | {status} | "
            "{blockers} |".format(
                name=name,
                cases=item.get("historical_case_count"),
                context_cases=item.get(
                    "maximum_exact_context_case_count"
                ),
                minimum=item.get("diagnostic_min_cases"),
                status=item.get("status"),
                blockers=", ".join(item.get("blockers", [])),
            )
        )
    lines.extend(
        [
            "",
            "Readiness is review-only. Missing historical cases or "
            "output semantics cannot be replaced with fixtures or "
            "assumptions.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"shadow_calibration_readiness_{stamp}"
