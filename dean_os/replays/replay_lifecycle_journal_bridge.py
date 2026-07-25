from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding


class ReplayLifecycleJournalBridge:
    """Append replay lifecycle, refresh, outcome, and diagnosis events once."""

    contract = "dean_replay_lifecycle_journal_bridge_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/replay_lifecycle_journal_bridge_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        lifecycle_json: str | Path,
        registration_json: str | Path,
        refresh_json: str | Path | None = None,
        ingestion_json: str | Path | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        apply: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        lifecycle_path = Path(lifecycle_json)
        registration_path = Path(registration_json)
        lifecycle = _load(lifecycle_path)
        registration = _load(registration_path)
        refresh_path = Path(refresh_json) if refresh_json is not None else None
        refresh = _load(refresh_path) if refresh_path and refresh_path.is_file() else None
        ingestion_path = Path(ingestion_json) if ingestion_json is not None else None
        ingestion = (
            _load(ingestion_path)
            if ingestion_path is not None and ingestion_path.is_file()
            else None
        )
        # lifecycle["inputs"] never actually contains "domain_id" (see
        # ReplayOutcomeLifecycleOrchestrator.build()'s inputs dict) -- the
        # real domain lives on the loaded registration artifact instead.
        # Without this, every journaled event silently mis-tagged its
        # domain_id as the hardcoded fallback.
        domain_id = str(
            ((lifecycle.get("inputs") or {}).get("domain_id"))
            or (registration.get("source_packet") or {}).get("domain_id")
            or "semiconductor_ai_infrastructure"
        )
        events: list[dict[str, Any]] = []
        registration_binding = artifact_binding(registration_path, registration)

        for item in (lifecycle.get("review_inbox") or {}).get("data_actions") or []:
            events.append(
                _event(
                    "action_proposed",
                    effective_at=str(item.get("due_at") or lifecycle.get("created_at")),
                    actor="replay_outcome_lifecycle",
                    domain_id=domain_id,
                    entity_type="verified_evidence_refresh",
                    entity_id="refresh_" + str(item.get("task_id")),
                    source_artifact=registration_binding,
                    context={
                        "task_id": item.get("task_id"),
                        "hypothesis_id": item.get("hypothesis_id"),
                    },
                    payload={
                        "action": "refresh_verified_checkpoint_evidence",
                        "route_state": item.get("route_state"),
                        "execution_status": "proposed_not_retried_automatically",
                        "can_trade": False,
                    },
                )
            )

        if refresh is not None and refresh_path is not None:
            refresh_binding = artifact_binding(refresh_path, refresh)
            refresh_summary = refresh.get("summary") or {}
            if (refresh.get("inputs") or {}).get("apply_refresh"):
                events.append(
                    _event(
                        "action_executed",
                        effective_at=str(refresh.get("created_at")),
                        actor="replay_evidence_refresh_controller",
                        domain_id=domain_id,
                        entity_type="verified_evidence_refresh_attempt",
                        entity_id=str(refresh.get("run_id")),
                        source_artifact=refresh_binding,
                        context={"refresh_job_count": refresh_summary.get("refresh_job_count")},
                        payload={
                            "status": refresh_summary.get("status"),
                            "network_access_performed": (refresh.get("safety") or {}).get(
                                "network_access_performed"
                            ),
                            "database_write_performed": False,
                            "broker_access_performed": False,
                            "can_trade": False,
                        },
                    )
                )
            failure = refresh.get("refresh_failure")
            if failure:
                events.append(
                    _event(
                        "incident_recorded",
                        effective_at=str(refresh.get("created_at")),
                        actor="replay_evidence_refresh_controller",
                        domain_id=domain_id,
                        entity_type="evidence_refresh_failure",
                        entity_id=str(refresh.get("run_id")),
                        source_artifact=refresh_binding,
                        context={"refresh_jobs": refresh.get("refresh_jobs") or []},
                        payload={**failure, "hypothesis_outcome_changed": False},
                    )
                )
            snapshot_summary = refresh.get("snapshot") or {}
            snapshot_path = _immutable_saved_path(snapshot_summary)
            if snapshot_path is not None:
                snapshot = _load(snapshot_path)
                events.append(
                    _event(
                        "source_snapshot_recorded",
                        effective_at=str(snapshot.get("created_at")),
                        actor="replay_evidence_refresh_controller",
                        domain_id=domain_id,
                        entity_type="verified_market_snapshot",
                        entity_id=str(snapshot.get("run_id")),
                        source_artifact=artifact_binding(snapshot_path, snapshot),
                        context={"refresh_run_id": refresh.get("run_id")},
                        payload={
                            "summary": snapshot.get("summary"),
                            "snapshot": snapshot.get("snapshot"),
                            "can_trade": False,
                        },
                    )
                )

        if ingestion is not None and ingestion_path is not None:
            ingestion_binding = artifact_binding(ingestion_path, ingestion)
            ingestion_summary = ingestion.get("summary") or {}
            if (ingestion.get("inputs") or {}).get("apply_ingestion"):
                events.append(
                    _event(
                        "action_executed",
                        effective_at=str(ingestion.get("created_at")),
                        actor="verified_local_snapshot_ingestion",
                        domain_id=domain_id,
                        entity_type="verified_local_snapshot_ingestion",
                        entity_id=str(ingestion.get("run_id")),
                        source_artifact=ingestion_binding,
                        context={"candidate": (ingestion.get("inputs") or {}).get("candidate_path")},
                        payload={
                            "status": ingestion_summary.get("status"),
                            "candidate_valid": ingestion_summary.get("candidate_valid"),
                            "snapshot_ingested": ingestion_summary.get("snapshot_ingested"),
                            "source_candidate_mutated": False,
                            "can_trade": False,
                        },
                    )
                )
            snapshot = ingestion.get("snapshot") or {}
            snapshot_path_value = snapshot.get("path")
            if snapshot_path_value and Path(str(snapshot_path_value)).is_file():
                events.append(
                    _event(
                        "source_snapshot_recorded",
                        effective_at=str(ingestion.get("created_at")),
                        actor="verified_local_snapshot_ingestion",
                        domain_id=domain_id,
                        entity_type="verified_market_snapshot",
                        entity_id=str(ingestion.get("run_id")),
                        source_artifact={
                            "path": str(snapshot_path_value),
                            "sha256": snapshot.get("sha256"),
                            "run_id": ingestion.get("run_id"),
                            "contract": ingestion.get("contract"),
                        },
                        context={
                            "source_candidate_sha256": snapshot.get(
                                "source_candidate_sha256"
                            )
                        },
                        payload={
                            "format": snapshot.get("format"),
                            "validated_before_ingestion": True,
                            "can_trade": False,
                        },
                    )
                )

        outcome_path = _immutable_saved_path(lifecycle.get("outcome_review") or {})
        if outcome_path is not None:
            outcome = _load(outcome_path)
            binding = artifact_binding(outcome_path, outcome)
            for item in outcome.get("checkpoint_reviews") or []:
                events.append(
                    _event(
                        "replay_checkpoint_matured",
                        effective_at=str(item.get("due_at") or outcome.get("created_at")),
                        actor="replay_outcome_lifecycle",
                        domain_id=domain_id,
                        entity_type="replay_checkpoint",
                        entity_id=str(item.get("task_id")),
                        source_artifact=binding,
                        context={
                            "hypothesis_id": item.get("hypothesis_id"),
                            "checkpoint_role": item.get("checkpoint_role"),
                        },
                        payload={
                            "review_status": item.get("review_status"),
                            "result_label": item.get("result_label"),
                            "missing_point_in_time_evidence": item.get(
                                "missing_point_in_time_evidence"
                            ),
                            "outcome_scoring_performed": False,
                        },
                    )
                )
            for item in outcome.get("outcomes") or []:
                events.append(
                    _event(
                        "outcome_recorded",
                        effective_at=str(outcome.get("created_at")),
                        actor="replay_outcome_lifecycle",
                        domain_id=domain_id,
                        entity_type="hypothesis_outcome",
                        entity_id=str(item.get("outcome_id")),
                        source_artifact=binding,
                        context={
                            "task_id": item.get("task_id"),
                            "hypothesis_id": item.get("hypothesis_id"),
                        },
                        payload=item,
                    )
                )

        learning_path = _immutable_saved_path(lifecycle.get("learning_review") or {})
        if learning_path is not None:
            learning = _load(learning_path)
            binding = artifact_binding(learning_path, learning)
            for card in learning.get("reverse_analysis_cards") or []:
                events.append(
                    _event(
                        "hypothesis_assessed",
                        effective_at=str(learning.get("created_at")),
                        actor="hypothesis_reverse_analysis",
                        domain_id=domain_id,
                        entity_type="hypothesis_reverse_analysis",
                        entity_id=str(card.get("hypothesis_id")),
                        source_artifact=binding,
                        context={
                            "outcome_id": card.get("outcome_id"),
                            "analysis_stage": card.get("analysis_stage"),
                        },
                        payload=card,
                    )
                )
            for proposal in learning.get("learning_proposals") or []:
                events.append(
                    _event(
                        "learning_proposal_created",
                        effective_at=str(learning.get("created_at")),
                        actor="hypothesis_learning_reviewer",
                        domain_id=domain_id,
                        entity_type="learning_proposal",
                        entity_id=str(proposal.get("proposal_id")),
                        source_artifact=binding,
                        context={"pattern_key": proposal.get("pattern_key")},
                        payload=proposal,
                    )
                )

        journal = SystemJournal(journal_path)
        write_result = (
            journal.append_many(events)
            if apply
            else {
                "requested_count": len(events),
                "appended_count": 0,
                "existing_count": 0,
                "dry_run": True,
            }
        )
        status = journal.status()
        created_at = utc_now_iso()
        run_id = "replay_lifecycle_journal_" + created_at.replace(":", "").replace(
            "+00:00", "Z"
        )
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_lifecycle_journal_bridge",
            "contract": self.contract,
            "summary": {
                "apply_requested": apply,
                "requested_event_count": len(events),
                "new_event_count": write_result.get("appended_count"),
                "existing_event_count": write_result.get("existing_count"),
                "journal_record_count": status.get("record_count"),
                "journal_chain_valid": status.get("chain_valid"),
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "can_trade": False,
            },
            "event_type_counts": _counts(events),
            "journal_write_result": write_result,
            "journal_status": status,
            "safety": {
                "append_only_audit_log": True,
                "action_authority_changed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=run_id,
            )
        return payload


def _immutable_saved_path(summary: dict[str, Any]) -> Path | None:
    saved = summary.get("saved_paths") or {}
    value = saved.get("json") or saved.get("latest_json")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _event(
    event_type: str,
    *,
    effective_at: str,
    actor: str,
    domain_id: str,
    entity_type: str,
    entity_id: str,
    source_artifact: dict[str, Any],
    context: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "effective_at": effective_at,
        "actor": actor,
        "domain_id": domain_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source_artifact": source_artifact,
        "parent_event_ids": [],
        "context": context,
        "payload": payload,
    }


def _counts(events: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in events:
        key = str(item.get("event_type"))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Replay Lifecycle Journal Bridge\n\n"
        f"- Apply requested: `{summary['apply_requested']}`\n"
        f"- Requested events: `{summary['requested_event_count']}`\n"
        f"- New events: `{summary['new_event_count']}`\n"
        f"- Existing events: `{summary['existing_event_count']}`\n"
        f"- Journal records: `{summary['journal_record_count']}`\n"
        f"- Chain valid: `{summary['journal_chain_valid']}`\n\n"
        "The bridge writes audit history only. It cannot promote rules or trade.\n"
    )


__all__ = ["ReplayLifecycleJournalBridge"]
