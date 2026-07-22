from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.analyst_core.contracts import (
    ANALYST_REASONING_RECEIPT_CONTRACT as REASONING_RECEIPT_CONTRACT,
    ANALYST_REASONING_SNAPSHOT_CONTRACT as SNAPSHOT_CONTRACT,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.research_corpus.hypothesis_measurement_policy_preparer import (
    HypothesisMeasurementPolicyPreparer,
)
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_replay_review_gate import WorldModelReplayReviewGate
from dean_os.world_model.world_model_review_resolution import WorldModelReviewResolutionBuilder


WORLD_MODEL_HYPOTHESIS_LIFECYCLE_CONTRACT = (
    "dean_world_model_hypothesis_lifecycle_orchestrator_v1"
)


class WorldModelHypothesisLifecycleOrchestrator:
    """Compose preparation, resolution and the next manual review gate."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/world_model_hypothesis_lifecycle_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        packet_json: str | Path,
        source_review_gate_json: str | Path,
        resolution_specs_v2_json: str | Path,
        reasoning_snapshot_json: str | Path | None = None,
        price_paths: list[str | Path] | None = None,
        pipeline_paths: list[str | Path] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        packet_path = Path(packet_json)
        gate_path = Path(source_review_gate_json)
        draft_path = Path(resolution_specs_v2_json)
        reasoning_path = (
            Path(reasoning_snapshot_json)
            if reasoning_snapshot_json is not None
            else None
        )
        reasoning = _load_reasoning_snapshot(reasoning_path)
        if reasoning is not None:
            packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
            packet_domain = str(
                (packet_payload.get("summary") or {}).get("domain_id") or ""
            )
            reasoning_domain = str(
                (reasoning.get("inputs") or {}).get("domain_id") or ""
            )
            if not packet_domain or reasoning_domain != packet_domain:
                raise ValueError(
                    "analyst reasoning snapshot domain does not match lifecycle packet"
                )
        preparation = HypothesisMeasurementPolicyPreparer(
            self.output_dir / "measurement_policy"
        ).build(
            draft_path,
            price_paths=price_paths,
            pipeline_paths=pipeline_paths,
            save=save,
        )
        prep_summary = preparation["measurement_policy_preparation"]["summary"]
        blocked = int(prep_summary["blocked_hypothesis_count"])
        resolution: dict[str, Any] | None = None
        review_gate: dict[str, Any] | None = None
        if blocked == 0:
            prepared_path: str | Path
            if save:
                prepared_path = preparation["saved_paths"]["latest_json"]
            else:
                # The resolution builder intentionally verifies an immutable
                # file binding, so unsaved orchestration stops at preparation.
                prepared_path = draft_path
            if save:
                resolution = WorldModelReviewResolutionBuilder(
                    self.output_dir / "resolved_packet"
                ).build(
                    packet_path,
                    gate_path,
                    prepared_path,
                    save=True,
                )
                resolved_path = resolution["saved_paths"]["latest_json"]
                review_gate = WorldModelReplayReviewGate(
                    self.output_dir / "manual_review_gate"
                ).build(resolved_path, approve=False, save=True)
        review_inbox = _review_inbox(preparation, review_gate, reasoning)
        orphan_proposal_blocked = any(
            "orphan_reasoning_review_proposal" in (item.get("blockers") or [])
            for item in review_inbox["blockers"]
        )
        status = (
            "blocked_measurement_policy_inputs"
            if blocked
            else "blocked_reasoning_proposal_binding"
            if orphan_proposal_blocked
            else "prepared_resolved_pending_manual_review"
            if resolution and review_gate
            else "prepared_only_save_required_for_hash_bound_resolution"
        )
        payload: dict[str, Any] = {
            "run_id": "world_model_hypothesis_lifecycle_"
            + utc_now_iso().replace(":", "").replace("+00:00", "Z"),
            "created_at": utc_now_iso(),
            "mode": "world_model_hypothesis_lifecycle_orchestrator",
            "contract": WORLD_MODEL_HYPOTHESIS_LIFECYCLE_CONTRACT,
            "inputs": {
                "packet": _binding(packet_path),
                "source_review_gate": _binding(gate_path),
                "resolution_specs_v2_draft": _binding(draft_path),
                "reasoning_snapshot": (
                    _binding(reasoning_path) if reasoning_path is not None else None
                ),
                "price_paths": [str(item) for item in price_paths or []],
                "pipeline_paths": [str(item) for item in pipeline_paths or []],
            },
            "summary": {
                "status": status,
                "measurement_contract_ready_count": prep_summary[
                    "relative_return_contract_ready_count"
                ],
                "measurement_blocked_hypothesis_count": blocked,
                "resolved_packet_created": resolution is not None,
                "manual_review_gate_created": review_gate is not None,
                "inbox_blocker_count": len(review_inbox["blockers"]),
                "inbox_contract_count": len(review_inbox["proposed_contracts"]),
                "inbox_pending_decision_count": len(
                    review_inbox["pending_decisions"]
                ),
                "machine_review_proposal_count": review_inbox[
                    "machine_review_proposal_count"
                ],
                "manual_review_required": True,
                "hypothesis_approval_performed": False,
                "replay_registration_performed": False,
                "learning_memory_write_performed": False,
                "can_trade": False,
            },
            "stages": {
                "measurement_policy_preparation": {
                    "summary": prep_summary,
                    "saved_paths": preparation.get("saved_paths"),
                },
                "world_model_resolution": {
                    "summary": (resolution or {}).get("summary"),
                    "saved_paths": (resolution or {}).get("saved_paths"),
                },
                "manual_review_gate": {
                    "summary": (review_gate or {}).get("summary"),
                    "saved_paths": (review_gate or {}).get("saved_paths"),
                },
            },
            "review_inbox": review_inbox,
            "safety": {
                "review_only": True,
                "source_artifacts_mutated": False,
                "automatic_hypothesis_approval_performed": False,
                "replay_registration_performed": False,
                "outcome_scoring_performed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _review_inbox(
    preparation: dict[str, Any],
    review_gate: dict[str, Any] | None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(
        (preparation.get("measurement_policy_preparation") or {}).get(
            "hypothesis_rows", []
        )
        or []
    )
    blockers = [
        {
            "hypothesis_id": row.get("hypothesis_id"),
            "blockers": list(row.get("blockers_added") or []),
            "status": row.get("status"),
        }
        for row in rows
        if row.get("blockers_added")
    ]
    contracts = [
        {
            "hypothesis_id": row.get("hypothesis_id"),
            "expected_direction": row.get("expected_direction"),
            "horizon_days": _contract_horizon(preparation, row.get("hypothesis_id")),
            "neutral_band_absolute_return": row.get(
                "neutral_band_absolute_return"
            ),
            "historical_sample_count": row.get("historical_sample_count"),
            "status": row.get("status"),
        }
        for row in rows
        if row.get("status")
        in {
            "calibrated_contract_attached",
            "existing_calibrated_contract_preserved",
        }
    ]
    pending = []
    for item in (review_gate or {}).get("hypothesis_review", []) or []:
        if item.get("disposition") is None:
            pending.append(
                {
                    "decision_type": "hypothesis_disposition",
                    "hypothesis_id": item.get("hypothesis_id"),
                    "claim": item.get("hypothesis"),
                    "recommended_next_action": item.get(
                        "recommended_next_action"
                    ),
                    "allowed_decisions": list(
                        item.get("allowed_dispositions")
                        or ["accept_for_replay", "reformulate", "defer", "reject"]
                    ),
                }
            )
    if blockers and not pending:
        pending = [
            {
                "decision_type": "resolve_measurement_blockers",
                "hypothesis_id": item["hypothesis_id"],
                "allowed_decisions": ["supply_missing_inputs", "defer", "reject"],
            }
            for item in blockers
        ]
    proposals = list((reasoning or {}).get("hypothesis_review_proposals") or [])
    known_hypothesis_ids = {
        str(row.get("hypothesis_id")) for row in rows if row.get("hypothesis_id")
    }
    pending_by_hypothesis = {
        str(item.get("hypothesis_id")): item
        for item in pending
        if item.get("hypothesis_id")
    }
    for proposal in proposals:
        hypothesis_id = str(proposal.get("hypothesis_id") or "")
        if hypothesis_id not in known_hypothesis_ids:
            blockers.append(
                {
                    "hypothesis_id": hypothesis_id or None,
                    "blockers": ["orphan_reasoning_review_proposal"],
                    "status": "proposal_not_bound_to_lifecycle_hypothesis",
                    "proposal_id": proposal.get("proposal_id"),
                }
            )
            continue
        decision = pending_by_hypothesis.get(hypothesis_id)
        if decision is None:
            decision = {
                "decision_type": "hypothesis_disposition",
                "hypothesis_id": hypothesis_id,
                "allowed_decisions": [
                    "accept_for_replay",
                    "reformulate",
                    "defer",
                    "reject",
                ],
            }
            pending.append(decision)
            pending_by_hypothesis[hypothesis_id] = decision
        decision.setdefault("machine_review_proposals", []).append(
            {
                **proposal,
                "proposal_only": True,
                "status_changed": False,
                "automatic_disposition_allowed": False,
            }
        )
    return {
        "status": (
            "blocked_measurement_inputs"
            if blockers
            else "pending_hypothesis_decisions"
            if pending
            else "no_pending_hypothesis_decisions"
        ),
        "blockers": blockers,
        "proposed_contracts": contracts,
        "pending_decisions": pending,
        "machine_review_proposal_count": len(proposals),
        "reasoning_receipt": (reasoning or {}).get("reasoning_receipt"),
        "display_policy": "show_blockers_contracts_and_pending_decisions_only",
        "automatic_decision_allowed": False,
    }


def _load_reasoning_snapshot(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != SNAPSHOT_CONTRACT:
        raise ValueError("unsupported analyst reasoning snapshot contract")
    receipt = payload.get("reasoning_receipt") or {}
    if receipt.get("contract") != REASONING_RECEIPT_CONTRACT:
        raise ValueError("analyst reasoning receipt contract is missing")
    if not receipt.get("receipt_id"):
        raise ValueError("analyst reasoning receipt_id is required")
    for proposal in payload.get("hypothesis_review_proposals", []) or []:
        if (
            proposal.get("requires_manual_review") is not True
            or proposal.get("status_changed") is not False
        ):
            raise ValueError("unsafe analyst hypothesis review proposal")
    return payload


def _contract_horizon(preparation: dict[str, Any], hypothesis_id: Any) -> Any:
    spec = (preparation.get("resolutions") or {}).get(str(hypothesis_id), {})
    return (spec.get("measurement_spec") or {}).get("primary_horizon_days")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    inbox = payload["review_inbox"]
    lines = [
            "# World Model Hypothesis Lifecycle",
            "",
            f"- Status: `{summary['status']}`",
            f"- Measurement contracts ready: {summary['measurement_contract_ready_count']}",
            f"- Measurement blockers: {summary['measurement_blocked_hypothesis_count']}",
            f"- Resolved packet created: {summary['resolved_packet_created']}",
            f"- Manual review gate created: {summary['manual_review_gate_created']}",
            "- Replay registration performed: false",
            "- Can trade: false",
            "",
            "## Compact Review Inbox",
            "",
            f"- Blockers: {len(inbox['blockers'])}",
            f"- Proposed contracts: {len(inbox['proposed_contracts'])}",
            f"- Pending decisions: {len(inbox['pending_decisions'])}",
            "",
        ]
    for item in inbox["blockers"]:
        lines.append(
            f"- BLOCKED `{item['hypothesis_id']}`: {', '.join(item['blockers'])}"
        )
    for item in inbox["proposed_contracts"]:
        lines.append(
            f"- CONTRACT `{item['hypothesis_id']}`: direction={item['expected_direction']} "
            f"horizon={item['horizon_days']}d band={item['neutral_band_absolute_return']}"
        )
    for item in inbox["pending_decisions"]:
        lines.append(
            f"- DECISION `{item['hypothesis_id']}`: {item['decision_type']}"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "WORLD_MODEL_HYPOTHESIS_LIFECYCLE_CONTRACT",
    "WorldModelHypothesisLifecycleOrchestrator",
]
