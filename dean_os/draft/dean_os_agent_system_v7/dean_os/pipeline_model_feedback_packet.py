from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.review_feedback_taxonomy import (
    applicable_labels,
    build_review_feedback_taxonomy,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_PIPELINE_MODEL_CASE_JSON = (
    "reports/dean_os/pipeline_model_case_packet_current/latest.json"
)

ACCEPTED_CASE_STATUSES = {
    "evaluation_block_case_ready",
    "evaluation_caution_case_ready",
    "evaluation_clear_case_ready",
}
INPUT_BINDINGS = (
    (
        "real_metric_evidence_json",
        "real_metric_evidence_sha256",
    ),
    ("model_evaluation_json", "model_evaluation_sha256"),
    ("feature_stability_json", "feature_stability_sha256"),
    ("metric_readiness_json", "metric_readiness_sha256"),
)
UNSAFE_REQUEST_FLAGS = (
    "apply_learning",
    "write_learning_memory",
    "write_config",
    "modify_thresholds",
    "same_fold_retry",
    "launch_model_variant",
    "requests_execution",
    "create_recommendation",
    "trade",
)


class PipelineModelFeedbackPacket:
    """Normalizes human feedback for a pipeline model case.

    Output candidates are review artifacts only. This packet cannot call the
    analyst learning apply loop because model-evaluation feedback has no market
    direction or realized forecast outcome.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_model_feedback_packet_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        pipeline_model_case_json: str | Path = (
            DEFAULT_PIPELINE_MODEL_CASE_JSON
        ),
        manual_feedback_json: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        case_packet = _load_json(pipeline_model_case_json)
        case_summary = _mapping(case_packet.get("summary"))
        case = _mapping(case_packet.get("case"))
        taxonomy = build_review_feedback_taxonomy(
            case_family="pipeline_model"
        )
        binding_checks = _case_binding_checks(case_packet)
        target = _feedback_target(case_packet)
        feedback_records = _normalize_feedback_records(
            _load_feedback_records(manual_feedback_json),
            taxonomy=taxonomy,
            target=target,
            case_packet=case_packet,
        )
        learning_candidates = _learning_candidates(
            feedback_records,
            case_packet,
        )
        checks = _review_checks(
            case_packet=case_packet,
            binding_checks=binding_checks,
            feedback_records=feedback_records,
        )
        status = _packet_status(checks, feedback_records)
        payload = {
            "run_id": _run_id("pipeline_model_feedback_packet"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_model_feedback_packet",
            "inputs": {
                "pipeline_model_case_json": str(
                    pipeline_model_case_json
                ),
                "pipeline_model_case_sha256": _file_sha256(
                    pipeline_model_case_json
                ),
                "pipeline_model_case_id": case_summary.get("case_id"),
                "manual_feedback_json": str(manual_feedback_json)
                if manual_feedback_json
                else None,
                "manual_feedback_sha256": _file_sha256(
                    manual_feedback_json
                ),
            },
            "summary": _summary(
                status=status,
                case_summary=case_summary,
                feedback_records=feedback_records,
                learning_candidates=learning_candidates,
                checks=checks,
            ),
            "feedback_target": target,
            "review_label_taxonomy": taxonomy,
            "manual_feedback_records": feedback_records,
            "learning_candidate_proposals": learning_candidates,
            "feedback_to_learning_contract": _feedback_contract(),
            "existing_learning_loop_compatibility": (
                _existing_learning_loop_compatibility(case)
            ),
            "review_checks": checks,
            "template_alignment": _template_alignment(),
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(
                status,
                feedback_records,
            ),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_model_feedback_packet_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_model_feedback_packet_markdown(
    payload: dict[str, Any],
) -> str:
    summary = _mapping(payload.get("summary"))
    target = _mapping(payload.get("feedback_target"))
    compatibility = _mapping(
        payload.get("existing_learning_loop_compatibility")
    )
    lines = [
        "# DEAN-OS Pipeline Model Feedback Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Packet status: `{summary.get('packet_status')}`",
        f"- Case ID: `{summary.get('case_id')}`",
        f"- Case status: `{summary.get('case_status')}`",
        f"- Feedback records: {summary.get('manual_feedback_record_count')}",
        f"- Learning candidates: {summary.get('learning_candidate_proposal_count')}",
        f"- Can apply learning: {summary.get('can_apply_learning')}",
        f"- Can launch model variant: {summary.get('can_launch_model_variant_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Feedback Target",
        "",
        f"- Target: `{target.get('target_id')}`",
        f"- Classification: `{target.get('case_classification')}`",
        f"- Root causes: {', '.join(target.get('root_cause_categories', [])) or 'none'}",
        "",
        "## Manual Feedback",
        "",
    ]
    if not payload.get("manual_feedback_records"):
        lines.append("- none")
    for record in payload.get("manual_feedback_records", []):
        lines.append(
            f"- `{record.get('feedback_id')}` "
            f"valid={record.get('can_be_learning_candidate')} "
            f"labels={', '.join(record.get('labels', []))}"
        )
        for blocker in record.get("blockers", []):
            lines.append(f"  - blocker: {blocker}")
    lines.extend(["", "## Proposal-only Learning Candidates", ""])
    if not payload.get("learning_candidate_proposals"):
        lines.append("- none")
    for candidate in payload.get("learning_candidate_proposals", []):
        lines.append(
            f"- `{candidate.get('candidate_id')}` "
            f"action=`{candidate.get('proposed_action')}` "
            f"status=`{candidate.get('promotion_status')}`"
        )
    lines.extend(["", "## Existing Learning Loop", ""])
    lines.append(
        f"- Compatible: {compatibility.get('compatible')}"
    )
    lines.append(f"- Reason: {compatibility.get('reason')}")
    lines.extend(["", "## Review Checks", ""])
    for check in payload.get("review_checks", []):
        lines.append(
            f"- {str(check.get('status')).upper()}: "
            f"`{check.get('code')}` - {check.get('message')}"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(
        f"- {item}" for item in payload.get("explicit_non_actions", [])
    )
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(
        f"- {item}" for item in payload.get("operator_next_steps", [])
    )
    return "\n".join(lines).strip() + "\n"


def _feedback_target(case_packet: dict[str, Any]) -> dict[str, Any]:
    summary = _mapping(case_packet.get("summary"))
    case = _mapping(case_packet.get("case"))
    return {
        "target_id": summary.get("case_id"),
        "target_type": "pipeline_model_evaluation_case",
        "case_status": summary.get("case_status"),
        "case_classification": summary.get("case_classification"),
        "result_label": summary.get("result_label"),
        "lineage": case.get("lineage", {}),
        "evaluated_at": case.get("evaluated_at"),
        "blocked_metric_planes": summary.get(
            "blocked_metric_planes",
            [],
        ),
        "root_cause_categories": summary.get(
            "root_cause_categories",
            [],
        ),
        "allowed_review_outputs": [
            "operator_rationale",
            "evidence_request",
            "evaluation_test_candidate",
            "incident_candidate",
            "pipeline_fix_candidate",
            "future_model_iteration_candidate_after_new_data",
            "no_learning_update",
        ],
    }


def _normalize_feedback_records(
    records: list[dict[str, Any]],
    *,
    taxonomy: dict[str, Any],
    target: dict[str, Any],
    case_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    known_labels = applicable_labels(taxonomy)
    known_actions = set(
        taxonomy.get("labels", {}).get("learning_action", [])
    )
    root_causes = set(target.get("root_cause_categories", []))
    case_status = target.get("case_status")
    new_data_status = _mapping(
        case_packet.get("new_data_requirement")
    ).get("status")
    normalized = []
    for index, record in enumerate(records):
        feedback_id = str(
            record.get("feedback_id")
            or f"pipeline_model_feedback_{index + 1}"
        )
        labels = _string_list(record.get("labels"))
        actions = _string_list(
            record.get("proposed_learning_actions")
        )
        blockers = []
        target_id = str(record.get("target_id") or "")
        if target_id != str(target.get("target_id") or ""):
            blockers.append("unknown_or_mismatched_target_id")
        if not str(record.get("reviewer") or "").strip():
            blockers.append("missing_reviewer")
        if not str(record.get("notes") or "").strip():
            blockers.append("missing_review_notes")
        if not labels:
            blockers.append("missing_labels")
        unknown_labels = sorted(
            set(labels).difference(known_labels)
        )
        if unknown_labels:
            blockers.append(
                "unknown_or_wrong_family_labels:"
                + ",".join(unknown_labels)
            )
        unknown_actions = sorted(
            set(actions).difference(known_actions)
        )
        if unknown_actions:
            blockers.append(
                "unknown_learning_actions:"
                + ",".join(unknown_actions)
            )
        for flag in UNSAFE_REQUEST_FLAGS:
            if record.get(flag) is True:
                blockers.append(f"unsafe_request:{flag}")
        if (
            "evaluation_block_valid" in labels
            and case_status != "evaluation_block_case_ready"
        ):
            blockers.append("evaluation_block_label_mismatches_case")
        if (
            "generalization_gap_confirmed" in labels
            and "generalization_gap" not in root_causes
        ):
            blockers.append(
                "generalization_gap_not_present_in_case"
            )
        if (
            "feature_instability_confirmed" in labels
            and "feature_instability" not in root_causes
        ):
            blockers.append(
                "feature_instability_not_present_in_case"
            )
        if "create_incident_candidate" in actions and not {
            "implementation_issue_suspected",
            "data_issue_suspected",
            "evidence_binding_issue",
        }.intersection(labels):
            blockers.append(
                "incident_candidate_requires_issue_label"
            )
        if (
            "propose_model_iteration_after_new_data" in actions
            and (
                "needs_new_forward_data" not in labels
                or new_data_status
                != "wait_for_new_forward_development_data"
            )
        ):
            blockers.append(
                "model_iteration_requires_new_forward_data_label_and_contract"
            )
        if "no_learning_update" in actions and len(actions) > 1:
            blockers.append(
                "no_learning_update_conflicts_with_other_actions"
            )
        normalized.append(
            {
                "feedback_id": feedback_id,
                "reviewer": str(
                    record.get("reviewer") or "human"
                ),
                "target_id": target_id,
                "review_type": str(
                    record.get("review_type")
                    or "model_evaluation_review"
                ),
                "severity": str(
                    record.get("severity") or "medium"
                ),
                "labels": labels,
                "proposed_learning_actions": actions,
                "notes": str(record.get("notes") or ""),
                "can_be_learning_candidate": bool(actions)
                and "no_learning_update" not in actions
                and not blockers,
                "no_learning_update_requested": actions
                == ["no_learning_update"],
                "blockers": _unique(blockers),
            }
        )
    return normalized


def _learning_candidates(
    feedback_records: list[dict[str, Any]],
    case_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    new_data = _mapping(case_packet.get("new_data_requirement"))
    for record in feedback_records:
        if not record.get("can_be_learning_candidate"):
            continue
        for action in record.get("proposed_learning_actions", []):
            candidates.append(
                {
                    "candidate_id": (
                        "pipeline_model_learning_candidate:"
                        f"{record.get('feedback_id')}:{action}"
                    ),
                    "source_feedback_id": record.get("feedback_id"),
                    "target_id": record.get("target_id"),
                    "case_family": "pipeline_model",
                    "review_type": record.get("review_type"),
                    "severity": record.get("severity"),
                    "labels": record.get("labels", []),
                    "proposed_action": action,
                    "promotion_status": (
                        "proposal_only_pending_separate_approval"
                    ),
                    "requires_new_forward_data": action
                    == "propose_model_iteration_after_new_data",
                    "data_after": new_data.get("data_after")
                    if action
                    == "propose_model_iteration_after_new_data"
                    else None,
                    "can_apply_now": False,
                    "can_write_learning_memory": False,
                    "can_write_config": False,
                    "can_launch_model_variant_now": False,
                    "can_trade": False,
                }
            )
    return candidates


def _case_binding_checks(
    case_packet: dict[str, Any],
) -> list[dict[str, str]]:
    inputs = _mapping(case_packet.get("inputs"))
    checks = []
    for path_key, hash_key in INPUT_BINDINGS:
        path = inputs.get(path_key)
        expected_hash = inputs.get(hash_key)
        current_hash = _file_sha256(path)
        matches = bool(
            path
            and expected_hash
            and current_hash
            and expected_hash == current_hash
        )
        checks.append(
            _check(
                "pass" if matches else "fail",
                f"{path_key}_binding_current",
                (
                    "Path exists and SHA-256 matches the model case."
                    if matches
                    else (
                        f"path={path!r}, "
                        f"expected_sha={expected_hash!r}, "
                        f"current_sha={current_hash!r}"
                    )
                ),
            )
        )
    return checks


def _review_checks(
    *,
    case_packet: dict[str, Any],
    binding_checks: list[dict[str, str]],
    feedback_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    summary = _mapping(case_packet.get("summary"))
    checks = [
        _check(
            "pass"
            if case_packet.get("mode")
            == "pipeline_model_case_packet"
            else "fail",
            "pipeline_model_case_artifact_type",
            str(case_packet.get("mode")),
        ),
        _check(
            "pass"
            if summary.get("case_status") in ACCEPTED_CASE_STATUSES
            else "fail",
            "pipeline_model_case_status_accepted",
            str(summary.get("case_status")),
        ),
        _check(
            "pass"
            if not summary.get("failed_review_checks")
            else "fail",
            "pipeline_model_case_has_no_failed_checks",
            str(summary.get("failed_review_checks", [])),
        ),
        *binding_checks,
    ]
    if not feedback_records:
        checks.append(
            _check(
                "warn",
                "manual_feedback_not_supplied",
                "Packet is ready to receive human model-case feedback.",
            )
        )
    else:
        invalid = [
            record["feedback_id"]
            for record in feedback_records
            if record.get("blockers")
        ]
        checks.append(
            _check(
                "pass" if not invalid else "fail",
                "manual_feedback_records_valid",
                (
                    "All supplied feedback records are valid."
                    if not invalid
                    else ", ".join(invalid)
                ),
            )
        )
    return checks


def _packet_status(
    checks: list[dict[str, str]],
    feedback_records: list[dict[str, Any]],
) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "pipeline_model_feedback_blocked"
    if feedback_records:
        return "pipeline_model_feedback_ready_with_candidates"
    return "pipeline_model_feedback_ready_pending_manual_feedback"


def _summary(
    *,
    status: str,
    case_summary: dict[str, Any],
    feedback_records: list[dict[str, Any]],
    learning_candidates: list[dict[str, Any]],
    checks: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "packet_status": status,
        "case_id": case_summary.get("case_id"),
        "case_status": case_summary.get("case_status"),
        "case_classification": case_summary.get(
            "case_classification"
        ),
        "manual_feedback_record_count": len(feedback_records),
        "learning_candidate_proposal_count": len(
            learning_candidates
        ),
        "failed_review_checks": [
            check["code"]
            for check in checks
            if check["status"] == "fail"
        ],
        "can_capture_manual_feedback": True,
        "can_create_learning_candidate_proposals": True,
        "can_route_to_existing_analyst_learning_apply_loop": False,
        "can_apply_learning": False,
        "can_write_learning_memory": False,
        "can_change_agent_weights": False,
        "can_modify_thresholds": False,
        "can_launch_model_variant_now": False,
        "can_write_production_config": False,
        "can_promote_model": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _feedback_contract() -> dict[str, Any]:
    return {
        "contract_id": "pipeline_model_feedback_to_learning_v1",
        "steps": [
            {
                "step_id": "capture_human_feedback",
                "output": "manual_feedback_record",
            },
            {
                "step_id": "validate_case_family_labels",
                "output": "classified_model_evaluation_feedback",
            },
            {
                "step_id": "create_candidates",
                "output": "proposal_only_learning_candidates",
            },
            {
                "step_id": "separate_approval",
                "output": "not_implemented_for_model_cases",
            },
            {
                "step_id": "apply",
                "output": "forbidden_in_this_packet",
            },
        ],
        "allowed_candidate_outputs": [
            "evaluation_test_candidate",
            "evidence_request",
            "incident_candidate",
            "pipeline_fix_candidate",
            "future_model_iteration_candidate_after_new_data",
        ],
        "rule": (
            "Human feedback can propose future evaluation work; it is not "
            "automatic truth and cannot mutate current model state."
        ),
    }


def _existing_learning_loop_compatibility(
    case: dict[str, Any],
) -> dict[str, Any]:
    lineage = _mapping(case.get("lineage"))
    return {
        "compatible": False,
        "loop": "ReviewApprovedLearningLoop",
        "status": "analyst_learning_loop_not_valid_for_model_case",
        "reason": (
            "The existing loop promotes directional Agent Lab theses. A "
            "pipeline evaluation case has model/target/timeframe lineage "
            "and no expected market direction or realized hit/miss outcome."
        ),
        "model_case_lineage": {
            "model": lineage.get("model"),
            "target_name": lineage.get("target_name"),
            "timeframe": lineage.get("timeframe"),
        },
        "required_before_any_future_apply": [
            "specialized model-feedback promotion schema",
            "human approval of one named candidate",
            "regression-test validation",
            "new-data requirement enforcement where applicable",
            "atomic audit event with no production mutation by default",
        ],
    }


def _template_alignment() -> dict[str, Any]:
    return {
        "harvested_contracts": [
            "REVIEW_LABEL_TAXONOMY",
            "FEEDBACK_TO_LEARNING_PIPELINE_TEMPLATE",
            "OUTCOME_REVIEW_TEMPLATE",
            "PATTERN_MEMORY_UPDATE_POLICY",
        ],
        "decisions": [
            "Use shared labels but keep domain outcomes and model evaluations in separate families.",
            "Create proposal-only candidates from reviewed feedback.",
            "Do not route model feedback into the directional analyst learning loop.",
            "Do not create a production incident merely because a metric constraint failed.",
        ],
    }


def _explicit_non_actions() -> list[str]:
    return [
        "No review action database, learning store, recommendation memory, pattern memory, prompt, agent weight, threshold, or config is written.",
        "No collector, pipeline, replay, evaluation, training, tuning, or model variant is run.",
        "No evaluation block is relabeled as a hit/miss market outcome.",
        "No model promotion, recommendation, allocation, order, broker call, paper trade, or live trade is created.",
    ]


def _operator_next_steps(
    status: str,
    feedback_records: list[dict[str, Any]],
) -> list[str]:
    if status == "pipeline_model_feedback_blocked":
        return [
            "Repair stale case bindings or invalid feedback labels before creating any candidate.",
            "Do not use the existing analyst learning apply loop for this case.",
        ]
    if feedback_records:
        return [
            "Review each proposal-only candidate separately; none is approved or applied.",
            "Use incident candidates only for an evidenced data or implementation defect, not for ordinary model underperformance.",
            "A future model-iteration candidate still waits for accepted new forward data.",
        ]
    return [
        "A human may provide a feedback JSON using the pipeline_model label family.",
        "Use no_learning_update when the negative case is valid and no reusable lesson is justified.",
        "Keep the current candidate blocked while unrelated system work continues.",
    ]


def _load_feedback_records(
    path: str | Path | None,
) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [
            record for record in payload if isinstance(record, dict)
        ]
    if not isinstance(payload, dict):
        return []
    for key in ("feedback_records", "records"):
        if isinstance(payload.get(key), list):
            return [
                record
                for record in payload[key]
                if isinstance(record, dict)
            ]
    return []


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )
