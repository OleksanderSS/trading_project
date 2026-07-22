from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.research_corpus.hypothesis_quality_assessment import (
    assess_hypothesis_quality,
    assessment_policy,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT

WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT = "dean_world_model_replay_review_gate_v1"
HYPOTHESIS_DISPOSITIONS = {
    "accept_for_replay",
    "reformulate",
    "defer",
    "reject",
}


class WorldModelReplayReviewGate:
    """Manual review gate for registering world-model replay tasks.

    The gate can produce an approved registration bundle, but it intentionally
    does not write a replay queue, learning memory, model config, or trading
    state. A separate operator-controlled step must consume the bundle.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_replay_review_gate",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_json: str | Path | dict[str, Any],
        *,
        approve: bool = False,
        reviewer: str | None = None,
        review_notes: str | None = None,
        hypothesis_dispositions: dict[str, Any] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        packet, packet_source_path = _load_packet(packet_json)
        packet_sha256 = (
            _sha256(packet_source_path) if packet_source_path is not None else None
        )
        replay_tasks = list(packet.get("replay_tasks", []) or [])
        decisions = _normalize_hypothesis_decisions(hypothesis_dispositions or {})
        hypothesis_review = _hypothesis_review(packet, decisions)
        issues = _gate_issues(
            packet,
            replay_tasks,
            approve=approve,
            reviewer=reviewer,
            review_notes=review_notes,
            hypothesis_review=hypothesis_review,
        )
        status = _gate_status(
            issues,
            approve=approve,
            hypothesis_review=hypothesis_review,
        )
        can_register = status == "replay_tasks_approved_for_registration"
        approved_hypothesis_ids = {
            item["hypothesis_id"]
            for item in hypothesis_review
            if item.get("disposition") == "accept_for_replay"
        }
        approved_tasks = (
            [
                task
                for task in replay_tasks
                if str(task.get("hypothesis_id")) in approved_hypothesis_ids
            ]
            if packet.get("cycle_binding_contract")
            else replay_tasks
        )
        registration_bundle = (
            _registration_bundle(
                packet,
                approved_tasks,
                reviewer=str(reviewer).strip(),
                review_notes=review_notes,
                source_packet_sha256=packet_sha256,
            )
            if can_register
            else None
        )
        payload = {
            "run_id": _run_id("world_model_replay_review_gate"),
            "created_at": utc_now_iso(),
            "mode": "world_model_replay_review_gate",
            "contract": WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
            "source_packet": {
                "path": str(packet_source_path) if packet_source_path else None,
                "sha256": packet_sha256,
                "run_id": packet.get("run_id"),
                "contract": packet.get("contract"),
                "packet_status": packet.get("summary", {}).get("packet_status"),
                "domain_id": packet.get("summary", {}).get("domain_id"),
                "as_of": packet.get("summary", {}).get("as_of"),
            },
            "summary": {
                "gate_status": status,
                "issue_count": len(issues),
                "issues": issues,
                "replay_task_count": len(replay_tasks),
                "approved_replay_task_count": (
                    len(approved_tasks) if can_register else 0
                ),
                "content_accepted_replay_task_count": len(approved_tasks),
                "hypothesis_count": len(hypothesis_review),
                "content_accepted_hypothesis_count": sum(
                    item.get("disposition") == "accept_for_replay"
                    for item in hypothesis_review
                ),
                "deferred_hypothesis_count": sum(
                    item.get("disposition") == "defer"
                    for item in hypothesis_review
                ),
                "pending_hypothesis_disposition_count": sum(
                    item.get("disposition") is None
                    for item in hypothesis_review
                ),
                "approved": can_register,
                "manual_hypothesis_review_complete": bool(hypothesis_review)
                and all(item.get("disposition") for item in hypothesis_review),
                "registration_blocked_hypothesis_count": sum(
                    bool(item.get("registration_blockers"))
                    for item in hypothesis_review
                ),
                "quality_replay_eligible_hypothesis_count": sum(
                    bool((item.get("quality_assessment") or {}).get("replay_eligible"))
                    for item in hypothesis_review
                ),
                "quality_band_counts": _quality_band_counts(hypothesis_review),
                "uncalibrated_confidence_hypothesis_count": sum(
                    (item.get("quality_assessment") or {}).get(
                        "confidence_probability"
                    )
                    is None
                    for item in hypothesis_review
                ),
                "matured_replay_checkpoint_count": sum(
                    task.get("checkpoint_state_at_packet") == "matured"
                    for task in replay_tasks
                ),
                "scheduled_replay_checkpoint_count": sum(
                    task.get("checkpoint_state_at_packet") == "scheduled"
                    for task in replay_tasks
                ),
                "manual_review_required_before_registration": not can_register,
                "can_register_replay_tasks": can_register,
                "registration_bundle_created": registration_bundle is not None,
                "cycle_hypothesis_alignment_status": (
                    (packet.get("hypothesis_alignment_review") or {})
                    .get("summary", {})
                    .get("status")
                ),
                "replay_task_registration_performed": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "operator_decision": {
                "requested_approval": approve,
                "reviewer": str(reviewer).strip() if reviewer else None,
                "review_notes": review_notes,
                "hypothesis_dispositions": {
                    hypothesis_id: decision.get("disposition")
                    for hypothesis_id, decision in decisions.items()
                },
                "hypothesis_decisions": decisions,
            },
            "hypothesis_alignment_review": packet.get(
                "hypothesis_alignment_review"
            ),
            "hypothesis_evaluation_policy": assessment_policy(),
            "hypothesis_review": hypothesis_review,
            "registration_bundle": registration_bundle,
            "source_replay_tasks_preview": replay_tasks[:20],
            "operator_next_steps": _operator_next_steps(
                status,
                can_register=can_register,
            ),
            "safety": _safety(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_replay_review_gate_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_world_model_replay_review_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    source = payload.get("source_packet", {})
    lines = [
        "# DEAN-OS World Model Replay Review Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Gate status: `{summary.get('gate_status')}`",
        f"- Source packet: `{source.get('run_id')}`",
        f"- Source status: `{source.get('packet_status')}`",
        f"- Domain: `{source.get('domain_id')}`",
        f"- As-of: `{source.get('as_of')}`",
        f"- Replay tasks: {summary.get('replay_task_count')}",
        f"- Content-ready hypotheses/tasks: {summary.get('content_accepted_hypothesis_count')}/{summary.get('content_accepted_replay_task_count')}",
        f"- Deferred hypotheses: {summary.get('deferred_hypothesis_count')}",
        f"- Can register replay tasks: {summary.get('can_register_replay_tasks')}",
        f"- Registration performed now: {summary.get('replay_task_registration_performed')}",
        f"- Can write learning memory: {summary.get('can_write_learning_memory')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Issues",
        "",
    ]
    issues = summary.get("issues") or []
    lines.extend(f"- {issue}" for issue in issues)
    if not issues:
        lines.append("- none")
    lines.extend(["", "## Hypothesis Review", ""])
    for item in payload.get("hypothesis_review", []):
        trigger = item.get("trigger_event") or {}
        lines.extend(
            [
                f"- `{item.get('hypothesis_id')}`: {item.get('hypothesis')}",
                f"  - evidence relationship: `{item.get('evidence_relationship_status')}`",
                f"  - trigger: {trigger.get('title') or 'missing'}",
                f"  - source: `{trigger.get('source_id') or 'missing'}`",
                f"  - source tier: `{trigger.get('source_tier') or 'unknown'}`",
                f"  - trigger published/available at: `{trigger.get('published_at') or 'missing'}`",
                f"  - sector horizons: {item.get('sector_thesis_horizons_days')}",
                f"  - event horizons: {item.get('event_response_horizons_days')}",
                f"  - event anchor: `{item.get('event_anchor_at') or 'missing'}`",
                f"  - matured/scheduled checkpoints: {item.get('matured_checkpoint_count')}/{item.get('scheduled_checkpoint_count')}",
                f"  - expectation context available: {item.get('expectation_context_available')}",
                f"  - claim version/action: `{item.get('claim_version') or 1}` / `{item.get('resolution_action') or 'original'}`",
                f"  - original hypothesis: `{item.get('original_hypothesis_id') or item.get('hypothesis_id')}`",
                f"  - resolution status: `{item.get('resolution_status') or 'not_versioned'}`",
                f"  - registration blockers: {item.get('registration_blockers') or []}",
                f"  - disposition: `{item.get('disposition') or 'pending'}`",
                f"  - next action: {item.get('recommended_next_action')}",
            ]
        )
        quality = item.get("quality_assessment") or {}
        if quality:
            lines.extend(
                [
                    f"  - hypothesis quality: `{quality.get('quality_band')}` ({quality.get('hypothesis_quality_score')}/100)",
                    f"  - replay quality floor met: {quality.get('replay_eligible')}",
                    f"  - maximum allowed use: `{quality.get('max_allowed_use')}`",
                    "  - confidence probability: `not calibrated`",
                    f"  - critical weaknesses: {quality.get('critical_weaknesses') or []}",
                    f"  - missing evidence: {quality.get('missing_evidence') or []}",
                    f"  - score caps: {quality.get('score_caps_applied') or []}",
                ]
            )
            for name, dimension in (quality.get("dimensions") or {}).items():
                lines.append(
                    f"    - {name}: {dimension.get('score')}/4 (`{dimension.get('level')}`)"
                )
        measurement = item.get("measurement_spec") or {}
        if measurement:
            lines.extend(
                [
                    f"  - target metrics: {measurement.get('target_metrics') or []}",
                    f"  - assessment rule: {measurement.get('assessment_rule')}",
                ]
            )
        if item.get("rationale"):
            lines.append(f"  - rationale: {item.get('rationale')}")
        if item.get("proposed_hypothesis"):
            lines.append(
                f"  - proposed reformulation: {item.get('proposed_hypothesis')}"
            )
    if not payload.get("hypothesis_review"):
        lines.append("- none")
    bundle = payload.get("registration_bundle")
    lines.extend(["", "## Registration Bundle", ""])
    if bundle:
        lines.extend(
            [
                f"- Bundle ID: `{bundle.get('bundle_id')}`",
                f"- Approved by: `{bundle.get('approved_by')}`",
                f"- Task count: {bundle.get('task_count')}",
                "- Status: approved candidate bundle; not yet written to a replay queue.",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _load_packet(
    packet_json: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(packet_json, dict):
        return dict(packet_json), None
    path = Path(packet_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"World-model packet must be a JSON object: {path}")
    return payload, path


def _gate_issues(
    packet: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
    *,
    approve: bool,
    reviewer: str | None,
    review_notes: str | None,
    hypothesis_review: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    if packet.get("contract") != WORLD_MODEL_EVENT_LEARNING_CONTRACT:
        issues.append("source_packet_contract_mismatch")
    if not packet.get("run_id"):
        issues.append("source_packet_run_id_missing")
    if not replay_tasks:
        issues.append("source_packet_has_no_replay_tasks")
    if packet.get("summary", {}).get("can_trade") is not False:
        issues.append("source_packet_trade_boundary_not_false")
    if packet.get("summary", {}).get("can_write_learning_memory") is not False:
        issues.append("source_packet_learning_write_boundary_not_false")
    if approve and not str(reviewer or "").strip():
        issues.append("reviewer_required_for_replay_registration_approval")
    if approve:
        invalid_tasks = [
            str(task.get("task_id") or index)
            for index, task in enumerate(replay_tasks)
            if task.get("manual_review_gate_required") is not True
        ]
        if invalid_tasks:
            issues.append(
                "replay_tasks_missing_manual_review_gate_required:"
                + ",".join(invalid_tasks[:10])
            )
    if packet.get("cycle_binding_contract"):
        issues.extend(_event_anchor_issues(packet, replay_tasks))
        alignment = packet.get("hypothesis_alignment_review") or {}
        alignment_summary = alignment.get("summary") or {}
        if alignment.get("contract") != "dean_cycle_hypothesis_alignment_review_v1":
            issues.append("cycle_bound_hypothesis_alignment_contract_missing")
        if alignment_summary.get("horizon_substitution_allowed") is not False:
            issues.append("cycle_bound_horizon_substitution_boundary_invalid")
        if approve and alignment_summary.get("unaligned_upstream_hypothesis_count") != 0:
            issues.append("cycle_bound_upstream_hypothesis_alignment_incomplete")
        if approve and not str(review_notes or "").strip():
            issues.append("cycle_bound_review_notes_required")
        if approve:
            pending = [
                str(item.get("hypothesis_id"))
                for item in hypothesis_review
                if item.get("disposition") is None
            ]
            invalid = [
                str(item.get("hypothesis_id"))
                for item in hypothesis_review
                if item.get("disposition") is not None
                and item.get("disposition") not in HYPOTHESIS_DISPOSITIONS
            ]
            if pending:
                issues.append(
                    "cycle_bound_hypothesis_dispositions_missing:"
                    + ",".join(pending[:10])
                )
            if invalid:
                issues.append(
                    "cycle_bound_hypothesis_dispositions_invalid:"
                    + ",".join(invalid[:10])
                )
            if hypothesis_review and not any(
                item.get("disposition") == "accept_for_replay"
                for item in hypothesis_review
            ):
                issues.append("cycle_bound_no_hypothesis_accepted_for_replay")
            quality_blocked_accepts = [
                str(item.get("hypothesis_id"))
                for item in hypothesis_review
                if item.get("disposition") == "accept_for_replay"
                and not bool(
                    (item.get("quality_assessment") or {}).get("replay_eligible")
                )
            ]
            if quality_blocked_accepts:
                issues.append(
                    "cycle_bound_accepted_hypotheses_fail_quality_floor:"
                    + ",".join(quality_blocked_accepts[:10])
                )
            blocked_accepts = [
                str(item.get("hypothesis_id"))
                for item in hypothesis_review
                if item.get("disposition") == "accept_for_replay"
                and item.get("registration_blockers")
            ]
            if blocked_accepts:
                issues.append(
                    "cycle_bound_accepted_hypotheses_have_registration_blockers:"
                    + ",".join(blocked_accepts[:10])
                )
        invalid_scope = [
            str(task.get("task_id") or index)
            for index, task in enumerate(replay_tasks)
            if task.get("replay_scope") != "event_response"
            or task.get("horizon_family") != "event_response_fixed_v1"
        ]
        if invalid_scope:
            issues.append(
                "cycle_bound_replay_horizon_scope_invalid:"
                + ",".join(invalid_scope[:10])
            )
    return issues


def _hypothesis_review(
    packet: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events = {
        str(event.get("evidence_id") or event.get("event_id")): event
        for event in packet.get("classified_events", []) or []
        if isinstance(event, dict)
    }
    alignment_by_world: dict[str, dict[str, Any]] = {}
    for row in (
        (packet.get("hypothesis_alignment_review") or {}).get("alignments", [])
        or []
    ):
        for world_id in row.get("world_hypothesis_ids", []) or []:
            alignment_by_world[str(world_id)] = row
    rows: list[dict[str, Any]] = []
    tasks_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
    for task in packet.get("replay_tasks", []) or []:
        if isinstance(task, dict):
            tasks_by_hypothesis.setdefault(
                str(task.get("hypothesis_id") or ""), []
            ).append(task)
    for hypothesis in packet.get("hypotheses", []) or []:
        if not isinstance(hypothesis, dict):
            continue
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        trigger_ids = list(hypothesis.get("trigger_evidence_ids") or [])
        if not trigger_ids:
            trigger_ids = list(hypothesis.get("supporting_evidence_ids") or [])
        trigger = events.get(str(trigger_ids[0])) if trigger_ids else None
        alignment = alignment_by_world.get(hypothesis_id, {})
        provenance = dict((trigger or {}).get("provenance") or {})
        decision = decisions.get(hypothesis_id, {})
        hypothesis_tasks = tasks_by_hypothesis.get(hypothesis_id, [])
        registration_blockers = list(hypothesis.get("registration_blockers") or [])
        event_anchor_at = next(
            (
                str(task.get("trigger_event_at"))
                for task in hypothesis_tasks
                if task.get("trigger_event_at")
            ),
            None,
        )
        quality_assessment = assess_hypothesis_quality(
            hypothesis,
            trigger_event=trigger,
            evidence_events=list(events.values()),
            packet_summary=dict(packet.get("summary") or {}),
            alignment=alignment,
            replay_tasks=hypothesis_tasks,
        )
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis": hypothesis.get("hypothesis"),
                "evidence_relationship_status": hypothesis.get(
                    "evidence_relationship_status",
                    "legacy_trigger_relation_requires_review",
                ),
                "trigger_evidence_ids": trigger_ids,
                "supporting_evidence_ids": list(
                    hypothesis.get("supporting_evidence_ids") or []
                ),
                "trigger_event": (
                    {
                        "event_class": trigger.get("event_class"),
                        "sentiment": trigger.get("sentiment"),
                        "title": trigger.get("title"),
                        "source_id": trigger.get("source_id"),
                        "source_type": trigger.get("source_type"),
                        "source_tier": provenance.get("source_tier"),
                        "source_identity": provenance.get("source_identity"),
                        "published_at": provenance.get("published_at"),
                        "record_sha256": provenance.get("record_sha256"),
                    }
                    if trigger
                    else None
                ),
                "related_upstream_hypothesis_id": alignment.get(
                    "upstream_hypothesis_id"
                ),
                "sector_thesis_horizons_days": list(
                    alignment.get("upstream_horizons_days") or []
                ),
                "event_response_horizons_days": list(
                    hypothesis.get("horizons_to_check") or []
                ),
                "event_anchor_at": event_anchor_at,
                "matured_checkpoint_count": sum(
                    task.get("checkpoint_state_at_packet") == "matured"
                    for task in hypothesis_tasks
                ),
                "scheduled_checkpoint_count": sum(
                    task.get("checkpoint_state_at_packet") == "scheduled"
                    for task in hypothesis_tasks
                ),
                "expectation_context_available": packet.get("summary", {}).get(
                    "expectation_context_available"
                ),
                "claim_version": hypothesis.get("claim_version"),
                "resolution_action": hypothesis.get("resolution_action"),
                "resolution_status": hypothesis.get("resolution_status"),
                "original_hypothesis_id": hypothesis.get(
                    "original_hypothesis_id"
                ),
                "measurement_spec": hypothesis.get("measurement_spec"),
                "expected_observations": list(
                    hypothesis.get("expected_observations") or []
                ),
                "invalidation_signals": list(
                    hypothesis.get("invalidation_signals") or []
                ),
                "registration_blockers": registration_blockers,
                "quality_assessment": quality_assessment,
                "recommended_next_action": _hypothesis_next_action(
                    decision.get("disposition"),
                    registration_blockers,
                    hypothesis_tasks,
                ),
                "allowed_dispositions": sorted(HYPOTHESIS_DISPOSITIONS),
                "disposition": decision.get("disposition"),
                "rationale": decision.get("rationale"),
                "proposed_hypothesis": decision.get("proposed_hypothesis"),
                "source_assessment": decision.get("source_assessment"),
                "manual_review_required": True,
            }
        )
    return rows


def _quality_band_counts(
    hypothesis_review: list[dict[str, Any]],
) -> dict[str, int]:
    counts = {"weak": 0, "limited": 0, "moderate": 0, "strong": 0}
    for item in hypothesis_review:
        band = str((item.get("quality_assessment") or {}).get("quality_band") or "")
        if band in counts:
            counts[band] += 1
    return counts


def _normalize_hypothesis_decisions(
    raw: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for key, value in dict(raw).items():
        hypothesis_id = str(key)
        if isinstance(value, dict):
            decision = {
                str(field): item
                for field, item in value.items()
                if field
                in {
                    "disposition",
                    "rationale",
                    "proposed_hypothesis",
                    "source_assessment",
                }
            }
            if decision.get("disposition") is not None:
                decision["disposition"] = str(decision["disposition"])
        else:
            decision = {"disposition": str(value)}
        decisions[hypothesis_id] = decision
    return decisions


def _hypothesis_next_action(
    disposition: Any,
    blockers: list[str],
    tasks: list[dict[str, Any]],
) -> str:
    if disposition == "accept_for_replay":
        matured = sum(
            task.get("checkpoint_state_at_packet") == "matured" for task in tasks
        )
        scheduled = sum(
            task.get("checkpoint_state_at_packet") == "scheduled" for task in tasks
        )
        return (
            "content_ready; collect and verify point-in-time checkpoint outcomes "
            f"(matured={matured}, scheduled={scheduled}); operator approval is still required before registration"
        )
    if disposition == "defer":
        if blockers:
            return "collect required context: " + ", ".join(blockers)
        return "record the reason for deferral and define a dated re-review condition"
    if disposition == "reformulate":
        return "create a new versioned claim with preserved lineage, then perform a fresh review"
    if disposition == "reject":
        return "archive as a negative reviewed case and send any diagnosed pattern to the learning proposal queue"
    return "complete manual content review"


def _event_anchor_issues(
    packet: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    packet_as_of = parse_timezone_aware(
        str((packet.get("summary") or {}).get("as_of") or "")
    )
    if packet_as_of is None:
        return ["cycle_bound_packet_as_of_missing_or_not_timezone_aware"]
    for index, task in enumerate(replay_tasks):
        task_id = str(task.get("task_id") or index)
        event_anchor = parse_timezone_aware(str(task.get("trigger_event_at") or ""))
        task_as_of = parse_timezone_aware(str(task.get("as_of") or ""))
        task_packet_as_of = parse_timezone_aware(
            str(task.get("packet_as_of") or "")
        )
        due_at = parse_timezone_aware(str(task.get("due_at") or ""))
        try:
            horizon = int(task.get("horizon_days"))
        except (TypeError, ValueError):
            horizon = 0
        if event_anchor is None:
            issues.append(f"cycle_bound_event_anchor_missing:{task_id}")
            continue
        if event_anchor > packet_as_of:
            issues.append(f"cycle_bound_event_anchor_after_packet_as_of:{task_id}")
        if task_as_of != event_anchor:
            issues.append(f"cycle_bound_task_as_of_not_event_anchor:{task_id}")
        if task_packet_as_of != packet_as_of:
            issues.append(f"cycle_bound_task_packet_as_of_mismatch:{task_id}")
        if horizon <= 0 or due_at != event_anchor + timedelta(days=horizon):
            issues.append(f"cycle_bound_due_at_not_event_anchored:{task_id}")
        expected_state = "matured" if due_at and due_at <= packet_as_of else "scheduled"
        if task.get("checkpoint_state_at_packet") != expected_state:
            issues.append(f"cycle_bound_checkpoint_state_invalid:{task_id}")
    return issues


def _gate_status(
    issues: list[str],
    *,
    approve: bool,
    hypothesis_review: list[dict[str, Any]],
) -> str:
    if issues:
        if "reviewer_required_for_replay_registration_approval" in issues:
            return "blocked_missing_reviewer_for_replay_approval"
        return "blocked_replay_registration_gate"
    if approve:
        return "replay_tasks_approved_for_registration"
    dispositions = [item.get("disposition") for item in hypothesis_review]
    if hypothesis_review and all(dispositions):
        if "reformulate" in dispositions:
            return "hypothesis_review_complete_reformulation_required"
        if "defer" in dispositions:
            return "hypothesis_review_complete_deferred"
        if all(item == "reject" for item in dispositions):
            return "hypothesis_review_complete_no_replay"
        return "hypothesis_review_complete_registration_not_requested"
    return "manual_review_required_for_replay_registration"


def _registration_bundle(
    packet: dict[str, Any],
    replay_tasks: list[dict[str, Any]],
    *,
    reviewer: str,
    review_notes: str | None,
    source_packet_sha256: str | None,
) -> dict[str, Any]:
    return {
        "bundle_id": _run_id("world_model_replay_registration_candidate"),
        "source_packet_id": packet.get("run_id"),
        "source_packet_contract": packet.get("contract"),
        "source_packet_sha256": source_packet_sha256,
        "cycle_binding_contract": packet.get("cycle_binding_contract"),
        "hypothesis_alignment_contract": (
            (packet.get("hypothesis_alignment_review") or {}).get("contract")
        ),
        "horizon_contract": (
            (packet.get("hypothesis_alignment_review") or {}).get(
                "horizon_contract"
            )
        ),
        "approved_by": reviewer,
        "approved_at": utc_now_iso(),
        "review_notes": review_notes,
        "task_count": len(replay_tasks),
        "tasks": replay_tasks,
        "allowed_next_step": "operator_may_register_replay_tasks_in_replay_queue",
        "forbidden_next_steps": [
            "trade_signal",
            "position_sizing",
            "learning_memory_write_without_outcome_review",
            "model_promotion_without_calibration_gate",
        ],
        "registration_note": (
            "This bundle authorizes only replay-task registration after manual "
            "review. Outcome scoring and learning-memory writes require later "
            "evidence and review gates."
        ),
    }


def _operator_next_steps(status: str, *, can_register: bool) -> list[str]:
    if can_register:
        return [
            "Register only the approved replay tasks in the replay queue.",
            "Wait for fixed-horizon outcomes before scoring hypotheses.",
            "Run a separate outcome/calibration gate before any learning-memory write.",
        ]
    if status == "manual_review_required_for_replay_registration":
        return [
            "Review each hypothesis trigger, evidence role, horizon families, scenario graph, and context gaps.",
            "For a cycle-bound packet, record accept_for_replay, reformulate, defer, or reject for every hypothesis.",
            "Only then rerun with --approve, --reviewer, --review-notes, and the disposition JSON.",
            "Do not register replay tasks directly from the event packet.",
        ]
    if status == "hypothesis_review_complete_reformulation_required":
        return [
            "Apply the recorded claim reformulations in a new world-model packet.",
            "Preserve the original packet and review artifact as reviewed lineage.",
            "Do not approve or register tasks from this mixed packet; carry accepted claims and replacements into a newly hash-bound packet.",
        ]
    if status == "hypothesis_review_complete_deferred":
        return [
            "Collect the evidence named in the disposition rationale before another review.",
            "Do not register deferred hypotheses as replay tasks.",
        ]
    if status == "hypothesis_review_complete_no_replay":
        return [
            "Close the candidate set without replay registration.",
            "Retain the rejected packet and review artifact for audit lineage.",
        ]
    if status == "hypothesis_review_complete_registration_not_requested":
        return [
            "Review the recorded accepted hypotheses before requesting registration approval.",
            "Use --approve only with an identified operator and non-empty review notes.",
        ]
    return [
        "Resolve gate issues before approval.",
        "Do not register replay tasks from a blocked packet.",
    ]


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "live_execution_performed": False,
        "broker_access_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "learning_memory_write_performed": False,
        "outcome_registration_performed": False,
        "replay_task_registration_performed": False,
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT",
    "WorldModelReplayReviewGate",
    "render_world_model_replay_review_gate_markdown",
]
