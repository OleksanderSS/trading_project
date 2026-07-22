from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.relative_return_direction_policy import (
    validate_relative_return_direction_contract,
)
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import artifact_binding
from dean_os.utils import json_ready
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)

WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT = "dean_world_model_review_resolution_v1"
HYPOTHESIS_RESOLUTION_SPECS_CONTRACT = (
    "dean_world_model_hypothesis_resolution_specs_v1"
)
HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2 = (
    "dean_world_model_hypothesis_resolution_specs_v2"
)


class WorldModelReviewResolutionBuilder:
    """Create a new versioned world packet from reviewed claim dispositions.

    The source packet and gate remain immutable. Accepted claims retain their
    identity; reformulated claims receive deterministic new identities and
    explicit lineage. This builder never approves or registers replay tasks.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_review_resolution_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_json: str | Path,
        review_gate_json: str | Path,
        resolution_specs_json: str | Path,
        *,
        save: bool = True,
    ) -> dict[str, Any]:
        packet_path = Path(packet_json)
        gate_path = Path(review_gate_json)
        specs_path = Path(resolution_specs_json)
        packet = _load(packet_path)
        gate = _load(gate_path)
        specs = _load(specs_path)
        packet_binding = artifact_binding(packet_path, packet)
        gate_binding = artifact_binding(gate_path, gate)
        specs_binding = artifact_binding(specs_path, specs)
        _validate_sources(
            packet,
            packet_binding,
            gate,
            gate_binding,
            specs,
        )

        review_by_id = {
            str(item.get("hypothesis_id")): item
            for item in gate.get("hypothesis_review", []) or []
            if item.get("hypothesis_id")
        }
        spec_by_id = dict(specs.get("resolutions") or {})
        original_hypotheses = list(packet.get("hypotheses") or [])
        original_ids = {
            str(item.get("hypothesis_id"))
            for item in original_hypotheses
            if item.get("hypothesis_id")
        }
        active_ids = {
            hypothesis_id
            for hypothesis_id, review in review_by_id.items()
            if review.get("disposition") in {"accept_for_replay", "reformulate"}
        }
        if set(spec_by_id) != active_ids:
            raise ValueError(
                "resolution spec IDs must exactly match accepted/reformulated hypothesis IDs"
            )
        if set(review_by_id) != original_ids:
            raise ValueError("review gate must cover every source packet hypothesis exactly once")

        new_hypotheses: list[dict[str, Any]] = []
        lineage: list[dict[str, Any]] = []
        old_to_new: dict[str, str] = {}
        excluded: list[dict[str, Any]] = []
        for original in original_hypotheses:
            old_id = str(original.get("hypothesis_id"))
            review = review_by_id[old_id]
            disposition = str(review.get("disposition"))
            if disposition in {"defer", "reject"}:
                excluded.append(
                    {
                        "original_hypothesis_id": old_id,
                        "disposition": disposition,
                        "rationale": review.get("rationale"),
                    }
                )
                continue
            spec = _validate_resolution_spec(
                old_id,
                original,
                review,
                spec_by_id[old_id],
                specs_contract=str(specs.get("contract") or ""),
            )
            action = str(spec["resolution_action"])
            claim = str(spec["resolved_hypothesis"]).strip()
            new_id = (
                old_id
                if action == "retain_claim"
                else "hypothesis_"
                + _digest(
                    {
                        "original_hypothesis_id": old_id,
                        "resolved_hypothesis": claim,
                        "source_review_gate_sha256": gate_binding["sha256"],
                    }
                )[:32]
            )
            old_to_new[old_id] = new_id
            resolution_id = "hypothesis_resolution_" + _digest(
                {
                    "old": old_id,
                    "new": new_id,
                    "spec_sha256": specs_binding["sha256"],
                }
            )[:24]
            blockers = list(spec.get("registration_blockers") or [])
            resolved = copy.deepcopy(original)
            resolved.update(
                {
                    "hypothesis_id": new_id,
                    "hypothesis": claim,
                    "expected_observations": list(spec["expected_observations"]),
                    "invalidation_signals": list(spec["invalidation_signals"]),
                    "measurement_spec": copy.deepcopy(spec["measurement_spec"]),
                    "registration_blockers": blockers,
                    "claim_version": int(original.get("claim_version") or 1)
                    + (1 if action == "replace_claim" else 0),
                    "resolution_action": action,
                    "resolution_status": (
                        "retained_after_manual_review"
                        if action == "retain_claim"
                        else "reformulated_pending_new_manual_review"
                    ),
                    "original_hypothesis_id": old_id,
                    "evidence_relationship_status": (
                        "trigger_only_claim_accepted_for_replay_observation"
                        if action == "retain_claim"
                        else "trigger_only_reformulated_claim_pending_new_review"
                    ),
                    "calibration_note": (
                        "Review-resolved candidate. Trigger evidence is not supporting "
                        "proof; measurements and invalidation must be evaluated point-in-time."
                    ),
                    "resolution_lineage": {
                        "contract": WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT,
                        "resolution_id": resolution_id,
                        "original_hypothesis_id": old_id,
                        "source_packet_run_id": packet.get("run_id"),
                        "source_packet_sha256": packet_binding["sha256"],
                        "source_review_gate_run_id": gate.get("run_id"),
                        "source_review_gate_sha256": gate_binding["sha256"],
                        "resolution_specs_sha256": specs_binding["sha256"],
                        "prior_disposition": disposition,
                        "prior_rationale": review.get("rationale"),
                        "prior_proposed_hypothesis": review.get(
                            "proposed_hypothesis"
                        ),
                    },
                }
            )
            resolved["safety"] = _hypothesis_safety()
            new_hypotheses.append(resolved)
            lineage.append(
                {
                    "resolution_id": resolution_id,
                    "original_hypothesis_id": old_id,
                    "resolved_hypothesis_id": new_id,
                    "resolution_action": action,
                    "prior_disposition": disposition,
                    "registration_blockers": blockers,
                }
            )

        source_tasks_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
        for task in packet.get("replay_tasks", []) or []:
            source_tasks_by_hypothesis.setdefault(
                str(task.get("hypothesis_id")), []
            ).append(task)
        resolved_by_id = {
            str(item["hypothesis_id"]): item for item in new_hypotheses
        }
        new_tasks: list[dict[str, Any]] = []
        for old_id, new_id in old_to_new.items():
            hypothesis = resolved_by_id[new_id]
            blockers = list(hypothesis.get("registration_blockers") or [])
            for source_task in source_tasks_by_hypothesis.get(old_id, []):
                task = copy.deepcopy(source_task)
                horizon = int(task.get("horizon_days"))
                task.update(
                    {
                        "task_id": f"replay_{new_id}_{horizon}d",
                        "hypothesis_id": new_id,
                        "scenario_graph_id": None,
                        "scenario_graph_resolution_status": (
                            "source_graph_not_reused_after_claim_resolution"
                        ),
                        "registration_status": (
                            "candidate_blocked_pending_required_context"
                            if blockers
                            else "candidate_pending_new_manual_review"
                        ),
                        "registration_blockers": blockers,
                        "review_action": (
                            "resolve_required_context_then_review_replay"
                            if blockers
                            else "review_resolved_claim_for_replay_observation"
                        ),
                        "resolution_lineage": copy.deepcopy(
                            hypothesis["resolution_lineage"]
                        ),
                    }
                )
                new_tasks.append(task)

        result = copy.deepcopy(packet)
        for key in ("saved_paths", "artifact_safety"):
            result.pop(key, None)
        created_at = utc_now_iso()
        result["run_id"] = _run_id("world_model_review_resolution")
        result["created_at"] = created_at
        result["mode"] = "world_model_review_resolution"
        result["contract"] = WORLD_MODEL_EVENT_LEARNING_CONTRACT
        result["review_resolution_contract"] = (
            WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT
        )
        result["source_review_resolution"] = {
            "source_packet": packet_binding,
            "source_review_gate": gate_binding,
            "resolution_specs": specs_binding,
            "reviewer": specs.get("reviewer"),
            "lineage": lineage,
            "excluded_hypotheses": excluded,
            "source_packet_mutated": False,
            "automatic_claim_generation_performed": False,
            "manual_review_required_again": True,
        }
        result["hypotheses"] = new_hypotheses
        result["replay_tasks"] = new_tasks
        result["scenario_outcome_graph"] = None
        result["hypothesis_alignment_review"] = _updated_alignment(
            packet.get("hypothesis_alignment_review") or {}, old_to_new
        )
        result["summary"] = _updated_summary(
            packet.get("summary") or {}, new_hypotheses, new_tasks, lineage
        )
        analysis_packet = copy.deepcopy(result.get("analysis_packet") or {})
        analysis_packet.update(
            {
                "packet_id": result["run_id"],
                "review_only": True,
                "resolution_packet": True,
            }
        )
        result["analysis_packet"] = analysis_packet
        result["delta_trail"] = list(result.get("delta_trail") or []) + [
            {
                "module_name": "world_model_review_resolution",
                "module_version": "1.0.0",
                "as_of": created_at,
                "source_packet_run_id": packet.get("run_id"),
                "source_review_gate_run_id": gate.get("run_id"),
                "hypothesis_lineage": lineage,
                "automatic_learning_performed": False,
                "replay_registration_performed": False,
            }
        ]
        result["operator_next_steps"] = [
            "Review the four resolved claim versions against their exact trigger sources.",
            "Accept only hypotheses with no unresolved registration blockers; defer blocked claims until the named context is attached.",
            "Do not reuse dispositions from the source packet because reformulated hypotheses have new identities.",
            "Do not register matured checkpoints without historical point-in-time outcome evidence.",
        ]
        result["safety"] = _packet_safety()
        if save:
            result["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=result,
                markdown=render_world_model_review_resolution_markdown(result),
                run_id=result["run_id"],
            )
        return json_ready(result)


def render_world_model_review_resolution_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    source = payload.get("source_review_resolution") or {}
    lines = [
        "# DEAN-OS World Model Review Resolution",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source packet: `{(source.get('source_packet') or {}).get('run_id')}`",
        f"- Source review gate: `{(source.get('source_review_gate') or {}).get('run_id')}`",
        f"- Status: `{summary.get('packet_status')}`",
        f"- Resolved hypotheses: {summary.get('hypothesis_count')}",
        f"- Retained/replaced: {summary.get('retained_hypothesis_count')}/{summary.get('reformulated_hypothesis_count')}",
        f"- Registration-blocked hypotheses: {summary.get('registration_blocked_hypothesis_count')}",
        f"- Candidate checkpoints: {summary.get('replay_task_count')}",
        f"- Matured/scheduled: {summary.get('matured_replay_checkpoint_count')}/{summary.get('scheduled_replay_checkpoint_count')}",
        "- Replay registration performed: false",
        "- Learning/trading performed: false",
        "",
        "## Resolved Hypotheses",
        "",
    ]
    for hypothesis in payload.get("hypotheses", []) or []:
        lineage = hypothesis.get("resolution_lineage") or {}
        blockers = hypothesis.get("registration_blockers") or []
        measurement = hypothesis.get("measurement_spec") or {}
        lines.extend(
            [
                f"### `{hypothesis.get('hypothesis_id')}`",
                "",
                f"- Claim: {hypothesis.get('hypothesis')}",
                f"- Original ID: `{lineage.get('original_hypothesis_id')}`",
                f"- Action/version: `{hypothesis.get('resolution_action')}` / `{hypothesis.get('claim_version')}`",
                f"- Evidence relation: `{hypothesis.get('evidence_relationship_status')}`",
                f"- Primary horizon: `{measurement.get('primary_horizon_days')}` days",
                f"- Measurement rule: {measurement.get('assessment_rule')}",
                f"- Registration blockers: {', '.join(blockers) if blockers else 'none'}",
                "- Expected observations:",
            ]
        )
        lines.extend(
            f"  - {item}" for item in hypothesis.get("expected_observations") or []
        )
        lines.append("- Invalidation signals:")
        lines.extend(
            f"  - {item}" for item in hypothesis.get("invalidation_signals") or []
        )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "This artifact versions reviewed claims. It does not approve replay registration, backfill missing context, reuse the source scenario graph, write learning memory, or trade.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_sources(
    packet: dict[str, Any],
    packet_binding: dict[str, Any],
    gate: dict[str, Any],
    gate_binding: dict[str, Any],
    specs: dict[str, Any],
) -> None:
    if packet.get("contract") != WORLD_MODEL_EVENT_LEARNING_CONTRACT:
        raise ValueError("unsupported source world-model packet contract")
    if gate.get("contract") != WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT:
        raise ValueError("unsupported source replay review gate contract")
    if specs.get("contract") not in {
        HYPOTHESIS_RESOLUTION_SPECS_CONTRACT,
        HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2,
    }:
        raise ValueError("unsupported hypothesis resolution specs contract")
    gate_source = gate.get("source_packet") or {}
    if gate_source.get("run_id") != packet.get("run_id"):
        raise ValueError("review gate points to a different source packet")
    if gate_source.get("sha256") != packet_binding["sha256"]:
        raise ValueError("source packet changed after manual review")
    for label, recorded, actual in (
        ("source packet", specs.get("source_packet") or {}, packet_binding),
        ("source review gate", specs.get("source_review_gate") or {}, gate_binding),
    ):
        if recorded.get("run_id") != actual.get("run_id"):
            raise ValueError(f"resolution specs {label} run binding mismatch")
        if recorded.get("sha256") != actual.get("sha256"):
            raise ValueError(f"resolution specs {label} hash binding mismatch")
    summary = gate.get("summary") or {}
    if summary.get("manual_hypothesis_review_complete") is not True:
        raise ValueError("source hypothesis review is incomplete")
    if summary.get("pending_hypothesis_disposition_count") != 0:
        raise ValueError("source review still has pending hypothesis dispositions")


def _validate_resolution_spec(
    hypothesis_id: str,
    original: dict[str, Any],
    review: dict[str, Any],
    raw_spec: Any,
    *,
    specs_contract: str = HYPOTHESIS_RESOLUTION_SPECS_CONTRACT,
) -> dict[str, Any]:
    if not isinstance(raw_spec, dict):
        raise ValueError(f"resolution spec must be an object: {hypothesis_id}")
    spec = copy.deepcopy(raw_spec)
    action = str(spec.get("resolution_action") or "")
    expected_action = (
        "retain_claim"
        if review.get("disposition") == "accept_for_replay"
        else "replace_claim"
    )
    if action != expected_action:
        raise ValueError(
            f"resolution action does not match reviewed disposition: {hypothesis_id}"
        )
    claim = str(spec.get("resolved_hypothesis") or "").strip()
    if not claim:
        raise ValueError(f"resolved hypothesis is required: {hypothesis_id}")
    original_claim = str(original.get("hypothesis") or "").strip()
    if action == "retain_claim" and claim != original_claim:
        raise ValueError(f"retained claim text changed: {hypothesis_id}")
    if action == "replace_claim" and claim == original_claim:
        raise ValueError(f"replacement claim is unchanged: {hypothesis_id}")
    if action == "replace_claim" and claim != str(
        review.get("proposed_hypothesis") or ""
    ).strip():
        raise ValueError(
            f"replacement claim differs from the manually reviewed proposal: {hypothesis_id}"
        )
    for field in ("expected_observations", "invalidation_signals"):
        values = spec.get(field)
        if not isinstance(values, list) or not all(
            str(value).strip() for value in values
        ):
            raise ValueError(f"{field} must be a non-empty string list: {hypothesis_id}")
    measurement = spec.get("measurement_spec")
    if not isinstance(measurement, dict):
        raise ValueError(f"measurement_spec is required: {hypothesis_id}")
    if int(measurement.get("primary_horizon_days") or 0) not in list(
        original.get("horizons_to_check") or []
    ):
        raise ValueError(f"measurement horizon is outside replay family: {hypothesis_id}")
    if not list(measurement.get("target_metrics") or []):
        raise ValueError(f"measurement target metrics are required: {hypothesis_id}")
    if not str(measurement.get("assessment_rule") or "").strip():
        raise ValueError(f"measurement assessment rule is required: {hypothesis_id}")
    relative_return_metrics = [
        str(item)
        for item in measurement.get("target_metrics") or []
        if "relative" in str(item).lower() and "return" in str(item).lower()
    ]
    direction_contract = measurement.get("relative_return_direction_contract")
    if direction_contract is not None:
        validate_relative_return_direction_contract(
            direction_contract,
            primary_horizon_days=int(measurement["primary_horizon_days"]),
        )
    if (
        specs_contract == HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2
        and relative_return_metrics
        and direction_contract is None
    ):
        raise ValueError(
            "v2 directional relative-return measurement requires a calibrated "
            f"direction contract: {hypothesis_id}"
        )
    measurement_context = measurement.get("measurement_context")
    if measurement_context is not None:
        _validate_measurement_context(hypothesis_id, measurement_context)
    blockers = spec.get("registration_blockers") or []
    if not isinstance(blockers, list) or not all(str(item).strip() for item in blockers):
        raise ValueError(f"registration blockers must be a string list: {hypothesis_id}")
    return spec


def _validate_measurement_context(
    hypothesis_id: str,
    raw_context: Any,
) -> None:
    """Validate common point-in-time rules while allowing domain payloads.

    Domain-specific contexts may add arbitrary target sets and metrics, but a
    ready context must remain review-only and prove that its baseline sources
    existed no later than the trigger.
    """

    if not isinstance(raw_context, dict):
        raise ValueError(
            f"measurement_context must be an object: {hypothesis_id}"
        )
    context = dict(raw_context)
    if context.get("context_contract") != "dean_hypothesis_measurement_context_v1":
        raise ValueError(
            f"measurement context contract mismatch: {hypothesis_id}"
        )
    context_as_of = parse_timezone_aware(context.get("context_as_of"))
    trigger_at = parse_timezone_aware(context.get("trigger_event_at"))
    if context_as_of is None or trigger_at is None:
        raise ValueError(
            f"measurement context timestamps must be timezone-aware: {hypothesis_id}"
        )
    if context_as_of > trigger_at:
        raise ValueError(
            f"measurement context as_of is after trigger: {hypothesis_id}"
        )
    if context.get("automatic_outcome_scoring_allowed") is not False:
        raise ValueError(
            f"measurement context must disable automatic outcome scoring: {hypothesis_id}"
        )

    source_records = _measurement_source_records(context)
    if not source_records:
        raise ValueError(
            f"measurement context has no baseline source records: {hypothesis_id}"
        )
    for index, record in enumerate(source_records):
        published_at = parse_timezone_aware(record.get("published_at"))
        if published_at is None:
            raise ValueError(
                "measurement baseline source timestamp must be timezone-aware: "
                f"{hypothesis_id}:{index}"
            )
        if published_at > trigger_at:
            raise ValueError(
                f"measurement baseline source is after trigger: {hypothesis_id}:{index}"
            )
        locator = str(record.get("source_locator") or "").strip()
        if not locator.startswith(("https://", "http://")):
            raise ValueError(
                f"measurement baseline source locator invalid: {hypothesis_id}:{index}"
            )

    buyer_basket = context.get("buyer_basket")
    if isinstance(buyer_basket, dict):
        members = list(buyer_basket.get("members") or [])
        _validate_context_coverage(
            hypothesis_id,
            "buyer_basket",
            members,
            buyer_basket.get("minimum_checkpoint_coverage"),
        )
        tickers: list[str] = []
        for index, member in enumerate(members):
            if not isinstance(member, dict):
                raise ValueError(
                    f"buyer basket member must be an object: {hypothesis_id}:{index}"
                )
            ticker = str(member.get("ticker") or "").strip().upper()
            if not ticker:
                raise ValueError(
                    f"buyer basket ticker missing: {hypothesis_id}:{index}"
                )
            tickers.append(ticker)
            low = _positive_number(member.get("baseline_low_usd_billions"))
            midpoint = _positive_number(
                member.get("baseline_midpoint_usd_billions")
            )
            high = _positive_number(member.get("baseline_high_usd_billions"))
            if low is None or midpoint is None or high is None or not (
                low <= midpoint <= high
            ):
                raise ValueError(
                    f"buyer basket baseline range invalid: {hypothesis_id}:{ticker}"
                )
        if len(tickers) != len(set(tickers)):
            raise ValueError(f"buyer basket tickers must be unique: {hypothesis_id}")

    equipment_basket = context.get("capital_equipment_basket")
    if isinstance(equipment_basket, dict):
        members = [
            str(item).strip().upper()
            for item in equipment_basket.get("members", []) or []
            if str(item).strip()
        ]
        _validate_context_coverage(
            hypothesis_id,
            "capital_equipment_basket",
            members,
            equipment_basket.get("minimum_checkpoint_coverage"),
        )
        if len(members) != len(set(members)):
            raise ValueError(
                f"capital equipment basket tickers must be unique: {hypothesis_id}"
            )
        benchmark = str(equipment_basket.get("benchmark") or "").strip().upper()
        if not benchmark or benchmark in set(members):
            raise ValueError(
                f"capital equipment benchmark invalid: {hypothesis_id}"
            )


def _measurement_source_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("source_locator") is not None:
            records.append(value)
        for child in value.values():
            records.extend(_measurement_source_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_measurement_source_records(child))
    return records


def _validate_context_coverage(
    hypothesis_id: str,
    label: str,
    members: list[Any],
    raw_minimum: Any,
) -> None:
    if not members:
        raise ValueError(f"{label} must have members: {hypothesis_id}")
    try:
        minimum = int(raw_minimum)
    except (TypeError, ValueError):
        minimum = 0
    if minimum < 1 or minimum > len(members):
        raise ValueError(f"{label} minimum coverage invalid: {hypothesis_id}")


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _updated_alignment(
    source_alignment: dict[str, Any], old_to_new: dict[str, str]
) -> dict[str, Any]:
    alignment = copy.deepcopy(source_alignment)
    rows = []
    for row in alignment.get("alignments", []) or []:
        updated = copy.deepcopy(row)
        updated["world_hypothesis_ids"] = [
            old_to_new[world_id]
            for world_id in row.get("world_hypothesis_ids", []) or []
            if world_id in old_to_new
        ]
        updated["alignment_status"] = "review_resolved_event_response_candidate_mapped"
        updated["manual_review_required"] = True
        rows.append(updated)
    alignment["alignments"] = rows
    summary = copy.deepcopy(alignment.get("summary") or {})
    summary.update(
        {
            "status": "all_upstream_mechanisms_mapped_to_review_resolved_claims",
            "world_hypothesis_count": len(old_to_new),
            "aligned_upstream_hypothesis_count": sum(
                bool(row.get("world_hypothesis_ids")) for row in rows
            ),
            "unaligned_upstream_hypothesis_count": sum(
                not bool(row.get("world_hypothesis_ids")) for row in rows
            ),
            "manual_review_required": True,
            "horizon_substitution_allowed": False,
        }
    )
    alignment["summary"] = summary
    return alignment


def _updated_summary(
    source: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = copy.deepcopy(source)
    blocked = sum(bool(item.get("registration_blockers")) for item in hypotheses)
    blocker_values = sorted(
        {
            str(blocker)
            for hypothesis in hypotheses
            for blocker in hypothesis.get("registration_blockers") or []
        }
    )
    summary.update(
        {
            "packet_status": (
                "review_resolution_ready_for_manual_review_with_context_gaps"
                if blocked
                else "review_resolution_ready_for_manual_review"
            ),
            "hypothesis_count": len(hypotheses),
            "replay_task_count": len(tasks),
            "event_anchored_replay_task_count": sum(
                bool(task.get("trigger_event_at")) for task in tasks
            ),
            "matured_replay_checkpoint_count": sum(
                task.get("checkpoint_state_at_packet") == "matured" for task in tasks
            ),
            "scheduled_replay_checkpoint_count": sum(
                task.get("checkpoint_state_at_packet") == "scheduled" for task in tasks
            ),
            "resolved_hypothesis_count": len(lineage),
            "retained_hypothesis_count": sum(
                item.get("resolution_action") == "retain_claim" for item in lineage
            ),
            "reformulated_hypothesis_count": sum(
                item.get("resolution_action") == "replace_claim" for item in lineage
            ),
            "registration_blocked_hypothesis_count": blocked,
            "registration_blocker_count": len(blocker_values),
            "registration_blockers": blocker_values,
            "unblocked_hypothesis_count": len(hypotheses) - blocked,
            "hypothesis_alignment_status": (
                "all_upstream_mechanisms_mapped_to_review_resolved_claims"
            ),
            "scenario_graph_available": False,
            "scenario_probability_mass_valid": False,
            "scenario_graph_resolution_status": (
                "source_graph_not_reused_after_claim_resolution"
            ),
            "manual_review_required": True,
            "manual_review_gate": "world_model_replay_review_gate",
            "can_register_replay_after_manual_review": bool(hypotheses) and blocked < len(hypotheses),
            "replay_task_registration_performed": False,
            "can_write_learning_memory": False,
            "can_promote_model": False,
            "can_write_config": False,
            "can_trade": False,
        }
    )
    return summary


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _digest(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _hypothesis_safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "no_live_execution": True,
        "no_broker_access": True,
        "no_production_config_write": True,
        "no_learning_memory_write": True,
        "no_model_promotion": True,
        "can_trade": False,
    }


def _packet_safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "source_packet_mutated": False,
        "automatic_claim_generation_performed": False,
        "replay_registration_performed": False,
        "outcome_scoring_performed": False,
        "learning_memory_write_performed": False,
        "production_config_write_performed": False,
        "model_promotion_performed": False,
        "broker_access_performed": False,
        "can_trade": False,
    }


__all__ = [
    "HYPOTHESIS_RESOLUTION_SPECS_CONTRACT",
    "HYPOTHESIS_RESOLUTION_SPECS_CONTRACT_V2",
    "WORLD_MODEL_REVIEW_RESOLUTION_CONTRACT",
    "WorldModelReviewResolutionBuilder",
    "render_world_model_review_resolution_markdown",
]
