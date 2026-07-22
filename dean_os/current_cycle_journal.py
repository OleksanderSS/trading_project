from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.contracts import (
    ANALYST_REASONING_RECEIPT_CONTRACT as REASONING_RECEIPT_CONTRACT,
    ANALYST_REASONING_SNAPSHOT_CONTRACT as SNAPSHOT_CONTRACT,
)
from dean_os.world_model_cycle_binding import verify_world_model_cycle_binding
from dean_os.research_corpus.hypothesis_learning_review import HYPOTHESIS_LEARNING_REVIEW_CONTRACT
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT
from dean_os.world_model.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
)

CURRENT_CYCLE_JOURNAL_CONTRACT = "dean_current_cycle_journal_v1"


class CurrentCycleJournalBuilder:
    """Import one governed analysis cycle into the canonical system journal."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/current_cycle_journal_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        cycle_json: str | Path,
        world_model_json: str | Path,
        review_gate_json: str | Path,
        closure_json: str | Path,
        learning_review_json: str | Path,
        reasoning_snapshot_json: str | Path | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        apply: bool = False,
        include_all_news: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        cycle_path = Path(cycle_json)
        world_path = Path(world_model_json)
        gate_path = Path(review_gate_json)
        closure_path = Path(closure_json)
        learning_path = Path(learning_review_json)
        reasoning_path = (
            Path(reasoning_snapshot_json)
            if reasoning_snapshot_json is not None
            else None
        )
        cycle = _load(cycle_path)
        world = _load(world_path)
        gate = _load(gate_path)
        closure = _load(closure_path)
        learning = _load(learning_path)
        reasoning = _load(reasoning_path) if reasoning_path is not None else None
        _verify_bindings(
            cycle_path,
            cycle,
            world_path,
            world,
            gate_path,
            gate,
            closure,
            learning,
        )
        domain_id = str(world.get("summary", {}).get("domain_id") or "unknown")
        if reasoning is not None and reasoning_path is not None:
            _verify_reasoning_snapshot(
                reasoning_path,
                reasoning,
                expected_domain_id=domain_id,
            )
        events: list[dict[str, Any]] = []
        source_snapshots: list[dict[str, Any]] = []
        input_artifacts = dict(cycle.get("inputs", {}).get("artifacts") or {})
        timeframe = cycle.get("inputs", {}).get("timeframe_lane_readiness")
        if isinstance(timeframe, dict):
            input_artifacts["timeframe_lane_readiness"] = timeframe

        news_payload: dict[str, Any] | None = None
        news_binding: dict[str, Any] | None = None
        for lane, source in sorted(input_artifacts.items()):
            source_path = Path(str(source.get("path") or ""))
            if not source_path.is_file():
                raise FileNotFoundError(f"cycle input artifact missing: {source_path}")
            binding = artifact_binding(source_path)
            expected_sha = source.get("sha256")
            if expected_sha and binding["sha256"] != expected_sha:
                raise ValueError(f"cycle input artifact changed after cycle: {lane}")
            loaded = _load(source_path)
            source_snapshots.append(binding)
            events.append(
                _event(
                    "source_snapshot_recorded",
                    effective_at=str(loaded.get("created_at") or cycle.get("created_at")),
                    actor="cycle_journal_importer",
                    domain_id=domain_id,
                    entity_type="source_snapshot",
                    entity_id=str(binding.get("run_id") or binding["sha256"]),
                    source_artifact=binding,
                    context={"cycle_run_id": cycle.get("run_id"), "lane": lane},
                    payload={
                        "lane": lane,
                        "status": loaded.get("status")
                        or (loaded.get("summary") or {}).get("status"),
                        "summary": _compact_summary(loaded.get("summary") or {}),
                    },
                )
            )
            if lane == "news":
                news_payload = loaded
                news_binding = binding

        news_events = (
            _news_events(
                news_payload or {},
                source_binding=news_binding or {},
                cycle_run_id=str(cycle.get("run_id")),
                domain_id=domain_id,
            )
            if include_all_news and news_payload is not None
            else []
        )
        events.extend(news_events)
        events.append(
            _event(
                "analysis_cycle_recorded",
                effective_at=str(
                    world.get("summary", {}).get("as_of") or cycle.get("created_at")
                ),
                actor="domain_orchestrator",
                domain_id=domain_id,
                entity_type="analysis_cycle",
                entity_id=str(cycle.get("run_id")),
                source_artifact=artifact_binding(cycle_path, cycle),
                context={"world_model_run_id": world.get("run_id")},
                payload={
                    "cycle_status": cycle.get("summary", {}).get("cycle_status"),
                    "evidence_count": cycle.get("summary", {}).get("evidence_count"),
                    "hypothesis_count": cycle.get("summary", {}).get(
                        "hypothesis_count"
                    ),
                    "source_snapshot_count": len(source_snapshots),
                },
            )
        )
        if reasoning is not None and reasoning_path is not None:
            events.extend(
                _reasoning_events(
                    reasoning,
                    source_binding=artifact_binding(reasoning_path, reasoning),
                    cycle_run_id=str(cycle.get("run_id")),
                    world_model_run_id=str(world.get("run_id")),
                    domain_id=domain_id,
                )
            )
        for evidence in world.get("classified_events", []) or []:
            provenance = evidence.get("provenance") or {}
            events.append(
                _event(
                    "evidence_observed",
                    effective_at=str(
                        provenance.get("published_at")
                        or world.get("summary", {}).get("as_of")
                    ),
                    actor="world_model_event_classifier",
                    domain_id=domain_id,
                    entity_type="evidence",
                    entity_id=str(evidence.get("evidence_id") or evidence.get("event_id")),
                    source_artifact=artifact_binding(world_path, world),
                    context={
                        "cycle_run_id": cycle.get("run_id"),
                        "world_model_run_id": world.get("run_id"),
                    },
                    payload={
                        "title": evidence.get("title"),
                        "source_id": evidence.get("source_id"),
                        "source_type": evidence.get("source_type"),
                        "event_class": evidence.get("event_class"),
                        "evidence_lane": evidence.get("source_evidence_type"),
                        "source_tier": provenance.get("source_tier"),
                        "record_sha256": provenance.get("record_sha256"),
                        "trigger_evidence_only": True,
                    },
                )
            )
        hypotheses = {
            str(item.get("hypothesis_id")): item
            for item in world.get("hypotheses", []) or []
            if item.get("hypothesis_id")
        }
        for hypothesis_id, hypothesis in hypotheses.items():
            events.append(
                _event(
                    "hypothesis_created",
                    effective_at=str(
                        hypothesis.get("as_of") or world.get("summary", {}).get("as_of")
                    ),
                    actor="world_model_hypothesis_generator",
                    domain_id=domain_id,
                    entity_type="hypothesis",
                    entity_id=hypothesis_id,
                    source_artifact=artifact_binding(world_path, world),
                    context={
                        "cycle_run_id": cycle.get("run_id"),
                        "world_model_run_id": world.get("run_id"),
                        "trigger_evidence_ids": hypothesis.get("trigger_evidence_ids") or [],
                    },
                    payload={
                        "hypothesis": hypothesis.get("hypothesis"),
                        "confidence": hypothesis.get("confidence"),
                        "status": hypothesis.get("status"),
                        "horizon_family": hypothesis.get("horizon_family"),
                        "horizons_to_check": hypothesis.get("horizons_to_check") or [],
                        "invalidation_signals": hypothesis.get("invalidation_signals")
                        or [],
                        "evidence_relationship_status": hypothesis.get(
                            "evidence_relationship_status"
                        ),
                    },
                )
            )
        action_events: list[dict[str, Any]] = []
        for review in gate.get("hypothesis_review", []) or []:
            hypothesis_id = str(review.get("hypothesis_id"))
            events.append(
                _event(
                    "hypothesis_reviewed",
                    effective_at=str(gate.get("created_at")),
                    actor=str(
                        gate.get("operator_decision", {}).get("reviewer")
                        or "manual_reviewer"
                    ),
                    domain_id=domain_id,
                    entity_type="hypothesis",
                    entity_id=hypothesis_id,
                    source_artifact=artifact_binding(gate_path, gate),
                    context={
                        "world_model_run_id": world.get("run_id"),
                        "review_gate_run_id": gate.get("run_id"),
                    },
                    payload={
                        "hypothesis": review.get("hypothesis"),
                        "disposition": review.get("disposition"),
                        "rationale": review.get("rationale"),
                        "source_assessment": review.get("source_assessment"),
                        "proposed_hypothesis": review.get("proposed_hypothesis"),
                        "trigger_event": review.get("trigger_event"),
                        "expectation_context_available": review.get(
                            "expectation_context_available"
                        ),
                        "quality_assessment": review.get("quality_assessment"),
                    },
                )
            )
            if review.get("disposition") == "reformulate":
                action_events.append(
                    _event(
                        "action_proposed",
                        effective_at=str(gate.get("created_at")),
                        actor=str(
                            gate.get("operator_decision", {}).get("reviewer")
                            or "manual_reviewer"
                        ),
                        domain_id=domain_id,
                        entity_type="hypothesis_reformulation",
                        entity_id="reformulate_" + hypothesis_id,
                        source_artifact=artifact_binding(gate_path, gate),
                        context={"hypothesis_id": hypothesis_id},
                        payload={
                            "action": "reformulate_hypothesis",
                            "proposed_hypothesis": review.get("proposed_hypothesis"),
                            "rationale": review.get("rationale"),
                            "execution_status": "not_executed_review_only",
                        },
                    )
                )
        events.extend(action_events)
        for card in learning.get("reverse_analysis_cards", []) or []:
            if card.get("analysis_stage") == "post_outcome" and card.get("outcome_id"):
                events.append(
                    _event(
                        "outcome_recorded",
                        effective_at=str(learning.get("created_at")),
                        actor="historical_replay_outcome_review",
                        domain_id=domain_id,
                        entity_type="hypothesis_outcome",
                        entity_id=str(card.get("outcome_id")),
                        source_artifact=artifact_binding(learning_path, learning),
                        context={
                            "hypothesis_id": card.get("hypothesis_id"),
                            "review_gate_run_id": gate.get("run_id"),
                        },
                        payload={
                            "result_label": card.get("result_label"),
                            "outcome_decomposition": card.get(
                                "outcome_decomposition"
                            ),
                            "machine_analysis_status": card.get(
                                "machine_analysis_status"
                            ),
                            "outcome_scoring_performed": False,
                            "can_trade": False,
                        },
                    )
                )
            events.append(
                _event(
                    "hypothesis_assessed",
                    effective_at=str(learning.get("created_at")),
                    actor="hypothesis_reverse_analysis",
                    domain_id=domain_id,
                    entity_type="hypothesis_reverse_analysis",
                    entity_id=str(card.get("hypothesis_id")),
                    source_artifact=artifact_binding(learning_path, learning),
                    context={
                        "review_gate_run_id": gate.get("run_id"),
                        "outcome_id": card.get("outcome_id"),
                        "analysis_stage": card.get("analysis_stage"),
                    },
                    payload=card,
                )
            )
        for proposal in learning.get("learning_proposals", []) or []:
            events.append(
                _event(
                    "learning_proposal_created",
                    effective_at=str(learning.get("created_at")),
                    actor="hypothesis_learning_reviewer",
                    domain_id=domain_id,
                    entity_type="learning_proposal",
                    entity_id=str(proposal.get("proposal_id")),
                    source_artifact=artifact_binding(learning_path, learning),
                    context={
                        "review_gate_run_id": gate.get("run_id"),
                        "pattern_key": proposal.get("pattern_key"),
                    },
                    payload=proposal,
                )
            )
        events.append(
            _event(
                "governance_closure_recorded",
                effective_at=str(closure.get("created_at")),
                actor="full_system_cycle_closure",
                domain_id=domain_id,
                entity_type="cycle_closure",
                entity_id=str(closure.get("run_id")),
                source_artifact=artifact_binding(closure_path, closure),
                context={
                    "cycle_run_id": cycle.get("run_id"),
                    "world_model_run_id": world.get("run_id"),
                    "review_gate_run_id": gate.get("run_id"),
                },
                payload={
                    "closure_status": closure.get("summary", {}).get(
                        "closure_status"
                    ),
                    "decision_state": closure.get("summary", {}).get(
                        "current_cycle_decision_state"
                    ),
                    "can_register_new_replay_tasks": closure.get("summary", {}).get(
                        "can_register_new_replay_tasks"
                    ),
                    "can_write_learning_memory": closure.get("summary", {}).get(
                        "can_write_learning_memory"
                    ),
                    "can_trade": closure.get("summary", {}).get("can_trade"),
                },
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
        journal_status = journal.status()
        event_counts = _event_counts(events)
        payload: dict[str, Any] = {
            "run_id": _run_id("current_cycle_journal"),
            "created_at": utc_now_iso(),
            "mode": "current_cycle_journal",
            "contract": CURRENT_CYCLE_JOURNAL_CONTRACT,
            "inputs": {
                "cycle": artifact_binding(cycle_path, cycle),
                "world_model": artifact_binding(world_path, world),
                "review_gate": artifact_binding(gate_path, gate),
                "closure": artifact_binding(closure_path, closure),
                "learning_review": artifact_binding(learning_path, learning),
                **(
                    {
                        "reasoning_snapshot": artifact_binding(
                            reasoning_path, reasoning
                        )
                    }
                    if reasoning is not None and reasoning_path is not None
                    else {}
                ),
            },
            "summary": {
                "apply_requested": apply,
                "source_snapshot_event_count": event_counts.get(
                    "source_snapshot_recorded", 0
                ),
                "news_event_count": event_counts.get("news_observed", 0),
                "evidence_event_count": event_counts.get("evidence_observed", 0),
                "hypothesis_created_count": event_counts.get(
                    "hypothesis_created", 0
                ),
                "hypothesis_reviewed_count": event_counts.get(
                    "hypothesis_reviewed", 0
                ),
                "hypothesis_reverse_analysis_count": event_counts.get(
                    "hypothesis_assessed", 0
                ),
                "reasoning_receipt_count": sum(
                    1
                    for event in events
                    if event.get("entity_type") == "reasoning_receipt"
                ),
                "machine_review_proposal_count": sum(
                    1
                    for event in events
                    if event.get("entity_type") == "hypothesis_review_proposal"
                ),
                "outcome_recorded_count": event_counts.get("outcome_recorded", 0),
                "action_proposal_count": event_counts.get("action_proposed", 0),
                "learning_proposal_count": event_counts.get(
                    "learning_proposal_created", 0
                ),
                "requested_journal_event_count": len(events),
                "new_journal_event_count": write_result.get("appended_count"),
                "idempotent_existing_event_count": write_result.get("existing_count"),
                "journal_record_count": journal_status.get("record_count"),
                "journal_chain_valid": journal_status.get("chain_valid"),
                "actions_executed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "can_trade": False,
            },
            "event_type_counts": event_counts,
            "journal_write_result": write_result,
            "journal_status": journal_status,
            "hypothesis_decisions": [
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "hypothesis": item.get("hypothesis"),
                    "disposition": item.get("disposition"),
                    "source_assessment": item.get("source_assessment"),
                    "proposed_hypothesis": item.get("proposed_hypothesis"),
                    "quality_assessment": item.get("quality_assessment"),
                }
                for item in gate.get("hypothesis_review", []) or []
            ],
            "reverse_analysis_cards": learning.get("reverse_analysis_cards") or [],
            "learning_proposals": learning.get("learning_proposals") or [],
            "reasoning_receipt": (reasoning or {}).get("reasoning_receipt"),
            "hypothesis_review_proposals": (reasoning or {}).get(
                "hypothesis_review_proposals", []
            ),
            "recommended_report_catalog": _recommended_reports(),
            "safety": _safety(),
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_current_cycle_journal_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_current_cycle_journal_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Current Cycle Journal",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Journal applied: {summary.get('apply_requested')}",
        f"- New immutable records: {summary.get('new_journal_event_count')}",
        f"- Existing idempotent records: {summary.get('idempotent_existing_event_count')}",
        f"- Journal chain valid: {summary.get('journal_chain_valid')}",
        f"- Total journal records: {summary.get('journal_record_count')}",
        f"- News recorded: {summary.get('news_event_count')}",
        f"- Hypotheses created/reviewed: {summary.get('hypothesis_created_count')}/{summary.get('hypothesis_reviewed_count')}",
        f"- Actions proposed/executed: {summary.get('action_proposal_count')}/0",
        f"- Learning proposals/rules applied: {summary.get('learning_proposal_count')}/0",
        "",
        "## Hypothesis Lifecycle",
        "",
    ]
    for item in payload.get("hypothesis_decisions", []) or []:
        lines.extend(
            [
                f"- `{item.get('hypothesis_id')}` — `{item.get('disposition')}`",
                f"  - claim: {item.get('hypothesis')}",
                f"  - diagnosis: `{item.get('source_assessment')}`",
            ]
        )
        if item.get("proposed_hypothesis"):
            lines.append(f"  - reformulation: {item.get('proposed_hypothesis')}")
        quality = item.get("quality_assessment") or {}
        if quality:
            lines.extend(
                [
                    f"  - pre-outcome quality: `{quality.get('quality_band')}` ({quality.get('hypothesis_quality_score')}/100)",
                    f"  - replay quality floor met: {quality.get('replay_eligible')}",
                    "  - calibrated truth probability: not available",
                ]
            )
    lines.extend(["", "## Hypothesis Reverse Analysis", ""])
    for card in payload.get("reverse_analysis_cards", []) or []:
        lines.append(
            f"- `{card.get('hypothesis_id')}` — `{card.get('machine_analysis_status')}`; "
            f"next: `{(card.get('recommended_next_action') or {}).get('action')}`"
        )
        for diagnosis in card.get("machine_diagnosis_candidates", []) or []:
            lines.append(
                f"  - `{diagnosis.get('error_code')}`: `{diagnosis.get('diagnostic_strength')}` "
                f"({diagnosis.get('failure_layer')})"
            )
    if not payload.get("reverse_analysis_cards"):
        lines.append("- none")
    lines.extend(["", "## Learning Proposals", ""])
    for proposal in payload.get("learning_proposals", []) or []:
        lines.append(
            f"- `{proposal.get('error_code')}` → `{proposal.get('target_component')}`: "
            f"{proposal.get('promotion_status')} "
            f"({proposal.get('current_independent_case_count')}/{proposal.get('minimum_independent_case_count')} cases)"
        )
    if not payload.get("learning_proposals"):
        lines.append("- none")
    lines.extend(["", "## Recommended Report Set", ""])
    for report in payload.get("recommended_report_catalog", []) or []:
        lines.extend(
            [
                f"### {report.get('name')}",
                "",
                f"- Cadence: `{report.get('cadence')}`",
                f"- Purpose: {report.get('purpose')}",
                f"- Required sections: {', '.join(report.get('required_sections') or [])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Safety Boundary",
            "",
            "The journal is append-only. Proposed actions and learning rules were recorded but not executed or promoted. No learning memory, production configuration, model or trading state was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _news_events(
    news: dict[str, Any],
    *,
    source_binding: dict[str, Any],
    cycle_run_id: str,
    domain_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in news.get("candidates", []) or []:
        candidate_id = str(candidate.get("candidate_sha256") or "").strip()
        if not candidate_id:
            candidate_id = _sha256_json(
                {
                    "locator": candidate.get("source_locator"),
                    "published_at": candidate.get("published_at"),
                    "title": candidate.get("title"),
                    "evidence_type": candidate.get("evidence_type"),
                }
            )
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        events.append(
            _event(
                "news_observed",
                effective_at=str(candidate.get("published_at")),
                actor="saved_news_evidence_producer",
                domain_id=domain_id,
                entity_type="news_item",
                entity_id=candidate_id,
                source_artifact=source_binding,
                context={
                    "cycle_run_id": cycle_run_id,
                    "news_snapshot_run_id": news.get("run_id"),
                },
                payload={
                    "title": candidate.get("title"),
                    "summary_preview": _preview(candidate.get("summary")),
                    "source": candidate.get("source"),
                    "source_identity": candidate.get("source_identity"),
                    "source_tier": candidate.get("source_tier"),
                    "source_locator": candidate.get("source_locator"),
                    "evidence_type": candidate.get("evidence_type"),
                    "matched_terms": candidate.get("matched_terms") or [],
                    "domain_terms": candidate.get("domain_terms") or [],
                    "candidate_sha256": candidate.get("candidate_sha256"),
                    "full_record_location": {
                        "artifact_path": source_binding.get("path"),
                        "artifact_sha256": source_binding.get("sha256"),
                        "collection": "candidates",
                        "candidate_sha256": candidate.get("candidate_sha256"),
                    },
                },
            )
        )
    return events


def _reasoning_events(
    reasoning: dict[str, Any],
    *,
    source_binding: dict[str, Any],
    cycle_run_id: str,
    world_model_run_id: str,
    domain_id: str,
) -> list[dict[str, Any]]:
    """Translate a reasoning receipt into canonical lifecycle events.

    Machine review proposals are assessments only. They never become
    ``hypothesis_reviewed`` events and cannot change hypothesis status.
    """
    receipt = dict(reasoning.get("reasoning_receipt") or {})
    effective_at = str(
        receipt.get("as_of")
        or reasoning.get("inputs", {}).get("as_of")
        or reasoning.get("created_at")
    )
    context = {
        "cycle_run_id": cycle_run_id,
        "world_model_run_id": world_model_run_id,
        "reasoning_snapshot_run_id": reasoning.get("run_id"),
        "reasoning_receipt_id": receipt.get("receipt_id"),
        "analysis_output_sha256": receipt.get("analysis_output_sha256"),
    }
    events = [
        _event(
            "analysis_cycle_recorded",
            effective_at=effective_at,
            actor="analyst_core_lens_orchestrator",
            domain_id=domain_id,
            entity_type="reasoning_receipt",
            entity_id=str(receipt["receipt_id"]),
            source_artifact=source_binding,
            context=context,
            payload={
                **receipt,
                "snapshot_status": reasoning.get("status"),
                "review_checks": reasoning.get("review_checks") or [],
            },
        )
    ]
    for hypothesis in reasoning.get("hypothesis_ledger", []) or []:
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            raise ValueError("reasoning snapshot hypothesis_id is required")
        events.append(
            _event(
                "hypothesis_created",
                effective_at=str(hypothesis.get("as_of") or effective_at),
                actor="analyst_core_hypothesis_ledger",
                domain_id=domain_id,
                entity_type="hypothesis",
                entity_id=hypothesis_id,
                source_artifact=source_binding,
                context={
                    **context,
                    "trigger_evidence_ids": hypothesis.get(
                        "trigger_evidence_ids"
                    )
                    or [],
                },
                payload={
                    **hypothesis,
                    "trigger_evidence_only": True,
                    "manual_review_required": True,
                    "status_change_performed": False,
                },
            )
        )
    for proposal in reasoning.get("hypothesis_review_proposals", []) or []:
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        hypothesis_id = str(proposal.get("hypothesis_id") or "").strip()
        if not proposal_id or not hypothesis_id:
            raise ValueError(
                "reasoning review proposal requires proposal_id and hypothesis_id"
            )
        events.append(
            _event(
                "hypothesis_assessed",
                effective_at=str(proposal.get("as_of") or effective_at),
                actor="analyst_core_hypothesis_ledger",
                domain_id=domain_id,
                entity_type="hypothesis_review_proposal",
                entity_id=proposal_id,
                source_artifact=source_binding,
                context={**context, "hypothesis_id": hypothesis_id},
                payload={
                    **proposal,
                    "assessment_kind": "machine_review_proposal",
                    "requires_manual_review": True,
                    "status_changed": False,
                    "automatic_disposition_allowed": False,
                },
            )
        )
    return events


def _verify_reasoning_snapshot(
    path: Path,
    reasoning: dict[str, Any],
    *,
    expected_domain_id: str,
) -> None:
    if reasoning.get("contract") != SNAPSHOT_CONTRACT:
        raise ValueError("unsupported analyst reasoning snapshot contract")
    inputs = reasoning.get("inputs") or {}
    if str(inputs.get("domain_id") or "") != expected_domain_id:
        raise ValueError("reasoning snapshot domain does not match current cycle")
    receipt = reasoning.get("reasoning_receipt") or {}
    if receipt.get("contract") != REASONING_RECEIPT_CONTRACT:
        raise ValueError("analyst reasoning receipt contract is missing")
    if not receipt.get("receipt_id") or not receipt.get("analysis_output_sha256"):
        raise ValueError("analyst reasoning receipt identity is incomplete")
    runtime_path = Path(str(inputs.get("runtime_json") or ""))
    expected_runtime_sha = str(inputs.get("runtime_sha256") or "")
    if runtime_path.is_file() and _file_sha256(runtime_path) != expected_runtime_sha:
        raise ValueError("reasoning runtime artifact changed after analysis")
    delta_rows = reasoning.get("reasoning_delta_trail", []) or []
    if list(receipt.get("delta_sha256s") or []) != [
        _sha256_json(item) for item in delta_rows
    ]:
        raise ValueError("reasoning delta trail does not match its receipt")
    proposal_ids = [
        str(item.get("proposal_id"))
        for item in reasoning.get("hypothesis_review_proposals", []) or []
    ]
    if list(receipt.get("proposal_ids") or []) != proposal_ids:
        raise ValueError("reasoning review proposals do not match their receipt")
    if not path.is_file():
        raise FileNotFoundError(path)


def _verify_bindings(
    cycle_path: Path,
    cycle: dict[str, Any],
    world_path: Path,
    world: dict[str, Any],
    gate_path: Path,
    gate: dict[str, Any],
    closure: dict[str, Any],
    learning: dict[str, Any],
) -> None:
    if world.get("contract") != WORLD_MODEL_EVENT_LEARNING_CONTRACT:
        raise ValueError("unsupported world-model event learning contract")
    if gate.get("contract") != WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT:
        raise ValueError("unsupported world-model replay review gate contract")
    if learning.get("contract") != HYPOTHESIS_LEARNING_REVIEW_CONTRACT:
        raise ValueError("unsupported hypothesis learning review contract")
    verify_world_model_cycle_binding(cycle_path, cycle, world_path, world)
    gate_source = gate.get("source_packet") or {}
    world_binding = artifact_binding(world_path, world)
    if gate_source.get("run_id") != world.get("run_id"):
        raise ValueError("review gate points to a different world-model packet")
    if gate_source.get("sha256") != world_binding.get("sha256"):
        raise ValueError("world-model packet changed after hypothesis review")
    closure_inputs = closure.get("inputs") or {}
    _assert_input_binding(closure_inputs.get("cycle"), cycle_path, cycle, "closure cycle")
    _assert_input_binding(
        closure_inputs.get("world_model"), world_path, world, "closure world model"
    )
    _assert_input_binding(
        closure_inputs.get("replay_review_gate"), gate_path, gate, "closure review gate"
    )
    learning_gate = learning.get("inputs", {}).get("review_gate") or {}
    if learning_gate.get("run_id") != gate.get("run_id"):
        raise ValueError("learning review points to a different review gate")
    if learning_gate.get("sha256") != artifact_binding(gate_path, gate).get("sha256"):
        raise ValueError("review gate changed after learning diagnosis")


def _assert_input_binding(
    recorded: dict[str, Any] | None,
    path: Path,
    payload: dict[str, Any],
    label: str,
) -> None:
    recorded = recorded or {}
    actual = artifact_binding(path, payload)
    if recorded.get("run_id") is not None and recorded.get("run_id") != payload.get("run_id"):
        raise ValueError(f"{label} run binding mismatch")
    if recorded.get("sha256") != actual.get("sha256"):
        raise ValueError(f"{label} hash binding mismatch")


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
        "context": context,
        "payload": payload,
    }


def _recommended_reports() -> list[dict[str, Any]]:
    return [
        {
            "report_id": "hypothesis_quality_card",
            "name": "Hypothesis quality card",
            "cadence": "before every content disposition or replay approval",
            "purpose": "Show structural evidence strength and bottlenecks without pretending to know the future.",
            "required_sections": [
                "claim and causal mechanism",
                "evidence quality and independence",
                "expectation/surprise context",
                "exposure and observability",
                "invalidation and confounders",
                "quality score caps and maximum allowed use",
            ],
        },
        {
            "report_id": "daily_analyst_journal",
            "name": "Daily analyst journal",
            "cadence": "each governed analysis cycle",
            "purpose": "Reconstruct what the analyst knew, inferred and left unresolved as-of the cycle.",
            "required_sections": [
                "regime and as-of",
                "source/news delta",
                "event classification",
                "expectation gaps",
                "hypotheses and confidence",
                "open evidence requests",
            ],
        },
        {
            "report_id": "hypothesis_lifecycle",
            "name": "Hypothesis lifecycle report",
            "cadence": "on creation, review, reformulation and outcome",
            "purpose": "Show one trace from trigger evidence through disposition, checkpoints and final assessment.",
            "required_sections": [
                "claim version history",
                "trigger/support/contradiction",
                "horizon anchors",
                "manual decisions",
                "outcomes and invalidation",
            ],
        },
        {
            "report_id": "failure_learning_review",
            "name": "Failure and learning review",
            "cadence": "after reformulation, rejection or outcome miss",
            "purpose": "Separate a wrong thesis, wrong market reaction, data failure and inconclusive evidence, then propose bounded improvements.",
            "required_sections": [
                "root-cause error codes",
                "conditions of failure",
                "counterfactual correct behavior",
                "pattern case count",
                "learning proposal status",
                "regression test requirement",
            ],
        },
        {
            "report_id": "action_governance_ledger",
            "name": "Action and governance ledger",
            "cadence": "daily and on every state-changing ceremony",
            "purpose": "Distinguish proposed, approved and executed actions and expose every authorization boundary.",
            "required_sections": [
                "proposed actions",
                "review decisions",
                "executed actions",
                "artifact hashes",
                "authorization state",
                "safety flags",
            ],
        },
        {
            "report_id": "news_source_coverage",
            "name": "News and source coverage report",
            "cadence": "each collection cycle",
            "purpose": "Track what was collected, deduplicated, excluded, stale, weak or missing by evidence lane.",
            "required_sections": [
                "accepted and excluded counts",
                "source independence",
                "source tiers",
                "lane coverage",
                "duplicate/future records",
                "open acquisition requests",
            ],
        },
        {
            "report_id": "weekly_calibration",
            "name": "Weekly calibration and replay report",
            "cadence": "weekly after matured checkpoints exist",
            "purpose": "Measure whether confidence, direction, timing and causal channel were calibrated across repeated cases.",
            "required_sections": [
                "Brier/calibration by class",
                "direction and horizon accuracy",
                "overconfidence",
                "missed channels",
                "false analogies",
                "promotable/rejected learning patterns",
            ],
        },
    ]


def _compact_summary(value: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "accepted_news_record_count",
        "accepted_evidence_count",
        "evidence_count",
        "ready_required_lanes",
        "missing_required_lanes",
        "can_enter_market_context_review",
        "can_influence_ticker_prediction",
        "can_train",
        "can_trade",
        "status",
    }
    return {key: item for key, item in value.items() if key in keep}


def _preview(value: Any, limit: int = 600) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for event in events:
        event_type = str(event["event_type"])
        result[event_type] = result.get(event_type, 0) + 1
    return dict(sorted(result.items()))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "journal_append_only": True,
        "action_execution_performed": False,
        "learning_memory_write_performed": False,
        "production_rule_update_performed": False,
        "model_promotion_performed": False,
        "broker_access_performed": False,
        "can_trade": False,
    }


__all__ = [
    "CURRENT_CYCLE_JOURNAL_CONTRACT",
    "CurrentCycleJournalBuilder",
    "render_current_cycle_journal_markdown",
]
