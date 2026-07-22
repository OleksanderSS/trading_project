from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.shadow_calibration_case_index import (
    validate_shadow_calibration_case,
)
from dean_os.utils import json_ready, sha256_json

SUPPORTED_COMPONENTS = {
    "regime",
    "specialist",
    "context_synthesis",
}


class ShadowComponentCaseProducer:
    """Join one shadow assessment family to accepted prediction outcomes."""

    def __init__(
        self,
        *,
        base_case_index_path: str | Path,
        component: str,
        component_artifact_path: str | Path,
        output_dir: str | Path = (
            "reports/dean_os/shadow_calibration_case_index_current"
        ),
    ):
        if component not in SUPPORTED_COMPONENTS:
            raise ValueError(
                f"Unsupported shadow calibration component: {component}"
            )
        self.base_case_index_path = Path(base_case_index_path)
        self.component = component
        self.component_artifact_path = Path(
            component_artifact_path
        )
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        base_state = _json_state(self.base_case_index_path)
        component_state = _json_state(
            self.component_artifact_path
        )
        source_issues = []
        if not base_state["available"]:
            source_issues.append("base_case_index_unavailable")
        if not component_state["available"]:
            source_issues.append("component_artifact_unavailable")

        existing_records: list[dict[str, Any]] = []
        prediction_cases: list[dict[str, Any]] = []
        if not source_issues:
            base_payload = base_state["payload"]
            if (
                base_payload.get("mode")
                != "shadow_calibration_case_index"
                or base_payload.get("schema_version")
                != "dean_shadow_calibration_case_index_v1"
            ):
                source_issues.append("base_case_index_schema_mismatch")
            else:
                for record in base_payload.get("records", []):
                    record_issues = validate_shadow_calibration_case(
                        record
                    )
                    if (
                        isinstance(record, dict)
                        and record.get("component") != "prediction"
                    ):
                        record_issues.extend(
                            validate_component_case(record)
                        )
                    if record_issues:
                        source_issues.append(
                            "base_case_index_contains_invalid_records"
                        )
                        continue
                    existing_records.append(record)
                    if record.get("component") == "prediction":
                        prediction_cases.append(record)
                if not prediction_cases:
                    source_issues.append(
                        "accepted_prediction_base_cases_missing"
                    )

        component_payload = {}
        if not source_issues:
            component_payload = _extract_component_payload(
                component_state["payload"],
                self.component,
            )
            if not component_payload:
                source_issues.append(
                    "component_contract_not_found"
                )

        produced = []
        rejected = []
        if not source_issues:
            for prediction_case in prediction_cases:
                assessment, issues = _assessment_for_case(
                    component=self.component,
                    payload=component_payload,
                    prediction_case=prediction_case,
                )
                if issues:
                    rejected.append({
                        "base_prediction_case_id": (
                            prediction_case.get("case_id")
                        ),
                        "issues": issues,
                    })
                    continue
                produced.append(
                    _component_case(
                        component=self.component,
                        prediction_case=prediction_case,
                        assessment=assessment,
                        component_state=component_state,
                    )
                )

        records = _merge_records(existing_records, produced)
        component_counts = {
            name: sum(
                item.get("component") == name for item in records
            )
            for name in (
                "prediction",
                "regime",
                "specialist",
                "context_synthesis",
            )
        }
        payload = {
            "run_id": _run_id(self.component),
            "created_at": utc_now_iso(),
            "mode": "shadow_calibration_case_index",
            "schema_version": "dean_shadow_calibration_case_index_v1",
            "status": (
                "shadow_component_cases_added"
                if produced and not source_issues
                else "shadow_component_case_production_blocked"
            ),
            "producer": {
                "component": self.component,
                "schema_version": (
                    "dean_shadow_component_case_producer_v1"
                ),
            },
            "source_inventory": {
                "base_case_index": _public_state(base_state),
                "component_artifact": _public_state(
                    component_state
                ),
            },
            "source_issues": sorted(set(source_issues)),
            "record_count": len(records),
            "new_record_count": len(produced),
            "rejected_case_count": len(rejected),
            "records": records,
            "rejected_cases": rejected,
            "component_counts": component_counts,
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
        payload["index_fingerprint"] = sha256_json({
            "base_case_index_sha256": base_state.get("sha256"),
            "component_artifact_sha256": component_state.get(
                "sha256"
            ),
            "component": self.component,
            "records": records,
            "rejected_cases": rejected,
        })
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_shadow_component_case_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def _assessment_for_case(
    *,
    component: str,
    payload: dict[str, Any],
    prediction_case: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if component == "regime":
        return _regime_assessment(payload, prediction_case)
    if component == "specialist":
        return _specialist_assessment(payload, prediction_case)
    return _synthesis_assessment(payload, prediction_case)


def _regime_assessment(
    payload: dict[str, Any],
    prediction_case: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    issues = []
    if payload.get("schema_version") != (
        "dean_stage7_regime_review_v1"
    ):
        issues.append("regime_schema_mismatch")
    if payload.get("status") != "stage7_regime_contexts_recorded":
        issues.append("regime_review_not_ready")
    if payload.get("context_partitioned") is not True:
        issues.append("regime_context_partition_not_proven")
    identity = _mapping(prediction_case.get("identity"))
    matches = [
        item
        for item in payload.get("contexts", [])
        if isinstance(item, dict)
        and _same_ticker_timeframe(item, identity)
    ]
    if len(matches) != 1:
        issues.append("regime_exact_context_not_unique")
        return {}, sorted(set(issues))
    item = matches[0]
    if item.get("identity_status") != "exact_context_key":
        issues.append("regime_identity_was_inferred")
    regime = str(item.get("regime") or "").upper()
    if not regime or regime == "UNKNOWN":
        issues.append("regime_label_missing")
    as_of = _timestamp(item.get("as_of"))
    prediction_as_of = _prediction_as_of(prediction_case)
    issues.extend(
        _assessment_time_issues(
            as_of,
            prediction_as_of,
            prefix="regime",
        )
    )
    if item.get("decision_influence") is not False:
        issues.append("regime_decision_influence_not_false")
    return {
        "schema_version": "dean_shadow_regime_assessment_v1",
        "status": "accepted" if not issues else "rejected",
        "ticker": identity.get("ticker"),
        "timeframe": identity.get("timeframe"),
        "as_of": as_of.isoformat() if as_of else None,
        "regime": regime or None,
        "confidence": item.get("confidence"),
        "metrics": _mapping(item.get("metrics")),
        "identity_status": item.get("identity_status"),
        "decision_influence": False,
    }, sorted(set(issues))


def _specialist_assessment(
    payload: dict[str, Any],
    prediction_case: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    issues = []
    if payload.get("schema_version") != (
        "dean_specialist_context_review_v1"
    ):
        issues.append("specialist_schema_mismatch")
    identity = _mapping(prediction_case.get("identity"))
    requested = _mapping(payload.get("requested_context"))
    if not _same_ticker_timeframe(requested, identity):
        issues.append("specialist_exact_context_mismatch")
    context_as_of = _timestamp(requested.get("as_of"))
    prediction_as_of = _prediction_as_of(prediction_case)
    if (
        context_as_of is None
        or prediction_as_of is None
        or context_as_of != prediction_as_of
    ):
        issues.append("specialist_context_as_of_not_exact")
    if _mapping(payload.get("point_in_time")).get("status") != (
        "point_in_time_compatible"
    ):
        issues.append("specialist_point_in_time_not_compatible")
    if _mapping(payload.get("timeframe_alignment")).get(
        "status"
    ) != "aligned":
        issues.append("specialist_timeframe_not_aligned")
    safety = _mapping(payload.get("safety"))
    if safety.get("eligible_for_exact_pipeline_context") is not True:
        issues.append("specialist_not_exact_pipeline_eligible")
    if safety.get("manual_review_required") is not False:
        issues.append("specialist_manual_review_not_complete")
    if safety.get("decision_influence") is not False:
        issues.append("specialist_decision_influence_not_false")
    ticker_scope = _mapping(payload.get("ticker_scope"))
    if ticker_scope.get("evidence_scope") != (
        "direct_ticker_review_candidate"
    ):
        issues.append("specialist_not_direct_ticker_scope")
    return {
        "schema_version": "dean_shadow_specialist_assessment_v1",
        "status": "accepted" if not issues else "rejected",
        "ticker": identity.get("ticker"),
        "timeframe": identity.get("timeframe"),
        "as_of": (
            context_as_of.isoformat() if context_as_of else None
        ),
        "evidence_scope": ticker_scope.get("evidence_scope"),
        "domain_id": _mapping(payload.get("domain_scope")).get(
            "domain_id"
        ),
        "point_in_time_status": _mapping(
            payload.get("point_in_time")
        ).get("status"),
        "timeframe_alignment_status": _mapping(
            payload.get("timeframe_alignment")
        ).get("status"),
        "manual_review_required": safety.get(
            "manual_review_required"
        ),
        "eligible_for_exact_pipeline_context": safety.get(
            "eligible_for_exact_pipeline_context"
        ),
        "decision_influence": False,
    }, sorted(set(issues))


def _synthesis_assessment(
    payload: dict[str, Any],
    prediction_case: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    issues = []
    if payload.get("schema_version") != (
        "dean_pipeline_context_synthesis_v1"
    ):
        issues.append("synthesis_schema_mismatch")
    if payload.get("status") not in {
        "context_synthesis_ready",
        "context_synthesis_caution",
    }:
        issues.append("synthesis_not_compatible")
    identity = _mapping(prediction_case.get("identity"))
    if not _same_ticker_timeframe(payload, identity):
        issues.append("synthesis_exact_context_mismatch")
    matches = [
        item
        for item in payload.get("prediction_assessments", [])
        if isinstance(item, dict)
        and item.get("model_context_id")
        == identity.get("model_context_id")
        and item.get("target_name") == identity.get("target_name")
        and item.get("context_fingerprint")
        == identity.get("context_fingerprint")
    ]
    if len(matches) != 1:
        issues.append("synthesis_prediction_assessment_not_unique")
        return {}, sorted(set(issues))
    item = matches[0]
    prediction_as_of = _prediction_as_of(prediction_case)
    synthesis_prediction_as_of = _timestamp(
        item.get("prediction_as_of")
    )
    if (
        prediction_as_of is None
        or synthesis_prediction_as_of != prediction_as_of
    ):
        issues.append("synthesis_prediction_as_of_not_exact")
    regime_as_of = _timestamp(item.get("regime_as_of"))
    issues.extend(
        _assessment_time_issues(
            regime_as_of,
            prediction_as_of,
            prefix="synthesis_regime",
        )
    )
    if item.get("freshness_status") != "compatible":
        issues.append("synthesis_freshness_not_compatible")
    if payload.get("directional_synthesis_performed") is not False:
        issues.append("synthesis_directional_use_not_false")
    if payload.get("decision_influence") is not False:
        issues.append("synthesis_decision_influence_not_false")
    return {
        "schema_version": (
            "dean_shadow_context_synthesis_assessment_v1"
        ),
        "status": "accepted" if not issues else "rejected",
        "ticker": identity.get("ticker"),
        "timeframe": identity.get("timeframe"),
        "as_of": (
            synthesis_prediction_as_of.isoformat()
            if synthesis_prediction_as_of
            else None
        ),
        "regime": _mapping(payload.get("regime")).get("regime"),
        "regime_as_of": (
            regime_as_of.isoformat() if regime_as_of else None
        ),
        "freshness_status": item.get("freshness_status"),
        "as_of_skew_minutes": item.get("as_of_skew_minutes"),
        "conflict_codes": [
            conflict.get("code")
            for conflict in payload.get("conflicts", [])
            if isinstance(conflict, dict)
        ],
        "review_confidence": payload.get("review_confidence"),
        "directional_synthesis_performed": False,
        "decision_influence": False,
    }, sorted(set(issues))


def _component_case(
    *,
    component: str,
    prediction_case: dict[str, Any],
    assessment: dict[str, Any],
    component_state: dict[str, Any],
) -> dict[str, Any]:
    source_provenance = dict(
        _mapping(prediction_case.get("source_provenance"))
    )
    source_provenance["component_assessment"] = {
        "path": component_state["path"],
        "sha256": component_state["sha256"],
    }
    identity = dict(_mapping(prediction_case.get("identity")))
    prediction = dict(
        _mapping(prediction_case.get("prediction"))
    )
    realization = dict(
        _mapping(prediction_case.get("realization"))
    )
    market_regime = (
        assessment.get("regime")
        or prediction_case.get("market_regime")
        or "unknown"
    )
    case_id = "shadow_case:" + sha256_json({
        "component": component,
        "base_prediction_case_id": prediction_case.get("case_id"),
        "component_artifact_sha256": component_state["sha256"],
        "assessment": assessment,
    })[:24]
    return {
        "schema_version": "dean_shadow_calibration_case_v1",
        "case_id": case_id,
        "case_validation_status": "accepted",
        "component": component,
        "base_prediction_case_id": prediction_case.get("case_id"),
        "identity": identity,
        "market_regime": market_regime,
        "prediction": prediction,
        "realization": realization,
        "assessment": assessment,
        "source_provenance": source_provenance,
        "safety": {
            "exact_context_match": True,
            "exact_realization_timestamp_match": True,
            "time_leakage_detected": False,
            "future_evidence_used": False,
            "sector_to_ticker_leakage_detected": False,
            "unsafe_output_detected": False,
            "decision_influence": False,
            "can_trade": False,
        },
    }


def validate_component_case(record: Any) -> list[str]:
    issues = validate_shadow_calibration_case(record)
    if not isinstance(record, dict):
        return issues
    component = record.get("component")
    if component not in SUPPORTED_COMPONENTS:
        return issues
    if not _text(record.get("base_prediction_case_id")):
        issues.append("component_base_prediction_case_id_missing")
    assessment = _mapping(record.get("assessment"))
    if assessment.get("status") != "accepted":
        issues.append("component_assessment_not_accepted")
    assessment_source = _mapping(
        _mapping(record.get("source_provenance")).get(
            "component_assessment"
        )
    )
    if not _text(assessment_source.get("path")):
        issues.append("component_assessment_path_missing")
    if not _sha_text(assessment_source.get("sha256")):
        issues.append("component_assessment_sha256_invalid")
    if component == "regime":
        if not _text(assessment.get("regime")):
            issues.append("component_regime_label_missing")
        if _timestamp(assessment.get("as_of")) is None:
            issues.append("component_regime_as_of_invalid")
    elif component == "specialist":
        if assessment.get(
            "eligible_for_exact_pipeline_context"
        ) is not True:
            issues.append("component_specialist_not_exact")
        if assessment.get("manual_review_required") is not False:
            issues.append("component_specialist_manual_pending")
    elif component == "context_synthesis":
        if assessment.get("freshness_status") != "compatible":
            issues.append("component_synthesis_freshness_invalid")
        if assessment.get(
            "directional_synthesis_performed"
        ) is not False:
            issues.append("component_synthesis_directional")
    return sorted(set(issues))


def _extract_component_payload(
    raw: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    expected_schema = {
        "regime": "dean_stage7_regime_review_v1",
        "specialist": "dean_specialist_context_review_v1",
        "context_synthesis": "dean_pipeline_context_synthesis_v1",
    }[component]
    candidates = [
        raw,
        _mapping(raw.get("payload")),
        _mapping(raw.get("metrics_snapshot")),
        _mapping(raw.get("context_synthesis")),
        _mapping(raw.get("specialist_context_review")),
        _mapping(raw.get("stage7_regime_review")),
        _mapping(
            _mapping(raw.get("dean_os_review_contract")).get(
                "stage7_regime_review"
            )
        ),
    ]
    for item in candidates:
        if item.get("schema_version") == expected_schema:
            return item
    return {}


def _assessment_time_issues(
    assessment_as_of: datetime | None,
    prediction_as_of: datetime | None,
    *,
    prefix: str,
) -> list[str]:
    if assessment_as_of is None or prediction_as_of is None:
        return [f"{prefix}_as_of_missing"]
    if assessment_as_of > prediction_as_of:
        return [f"{prefix}_uses_post_prediction_evidence"]
    return []


def _prediction_as_of(
    prediction_case: dict[str, Any],
) -> datetime | None:
    return _timestamp(
        _mapping(prediction_case.get("prediction")).get("as_of")
    )


def _same_ticker_timeframe(
    item: dict[str, Any],
    identity: dict[str, Any],
) -> bool:
    return (
        str(item.get("ticker") or "").strip().upper()
        == str(identity.get("ticker") or "").strip().upper()
        and str(item.get("timeframe") or "").strip().lower()
        == str(identity.get("timeframe") or "").strip().lower()
    )


def _merge_records(
    existing: list[dict[str, Any]],
    produced: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("case_id")): item
        for item in existing
        if item.get("case_id")
    }
    for item in produced:
        by_id[str(item["case_id"])] = item
    return [by_id[key] for key in sorted(by_id)]


def render_shadow_component_case_markdown(
    payload: dict[str, Any],
) -> str:
    producer = _mapping(payload.get("producer"))
    lines = [
        "# DEAN-OS Shadow Component Case Production",
        "",
        f"- Component: `{producer.get('component')}`",
        f"- Status: `{payload.get('status')}`",
        f"- New cases: {payload.get('new_record_count')}",
        f"- Total records: {payload.get('record_count')}",
        f"- Rejected base cases: {payload.get('rejected_case_count')}",
        "- Calibration executed: False",
        "- Can trade: False",
    ]
    if payload.get("source_issues"):
        lines.extend(["", "## Source Blockers", ""])
        lines.extend(
            f"- `{item}`" for item in payload["source_issues"]
        )
    if payload.get("rejected_cases"):
        lines.extend(["", "## Rejected Cases", ""])
        for item in payload["rejected_cases"]:
            lines.append(
                f"- `{item.get('base_prediction_case_id')}`: "
                + ", ".join(item.get("issues", []))
            )
    lines.extend([
        "",
        "Produced cases are review-only. They do not calibrate agents, "
        "change weights, write memory/config, recommend, or trade.",
    ])
    return "\n".join(lines) + "\n"


def _json_state(path: Path) -> dict[str, Any]:
    state = {
        "path": str(path),
        "available": False,
        "sha256": None,
        "payload": {},
    }
    if not path.is_file():
        return state
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state
    if not isinstance(raw, dict):
        return state
    state.update({
        "available": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload": raw,
    })
    return state


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key != "payload"
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha_text(value: Any) -> bool:
    text = _text(value)
    return bool(
        text
        and len(text) == 64
        and all(char in "0123456789abcdef" for char in text.lower())
    )


def _run_id(component: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"shadow_{component}_case_producer_{stamp}"
