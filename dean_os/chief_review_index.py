from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REVIEW_INDEX_PATH = "reports/dean_os/review_index/latest.json"
DEFAULT_HYPOTHESIS_LIFECYCLE_PATH = (
    "reports/dean_os/world_model_hypothesis_lifecycle_current/latest.json"
)
DEFAULT_CHECKPOINT_DUE_ROUTER_PATH = (
    "reports/dean_os/replay_checkpoint_due_router_current/latest.json"
)
DEFAULT_OUTCOME_LIFECYCLE_PATH = (
    "reports/dean_os/replay_outcome_lifecycle_current/latest.json"
)
DEFAULT_EVIDENCE_REFRESH_PATH = (
    "reports/dean_os/replay_evidence_refresh_controller_current/latest.json"
)
DEFAULT_VERIFIED_SOURCE_ROUTER_PATH = (
    "reports/dean_os/verified_market_source_router_current/latest.json"
)


class ChiefReviewIndexBuilder:
    """Consolidates ReviewIndex artifacts into a supervised review decision.

    This is a review-only synthesis layer. It reads the ReviewIndex artifact,
    classifies the current state, and writes a human-readable review artifact.
    It does not approve, execute, tune, train, promote, or write learning memory.
    """

    def __init__(
        self,
        review_index_path: str | Path = DEFAULT_REVIEW_INDEX_PATH,
        hypothesis_lifecycle_path: str | Path | None = (
            DEFAULT_HYPOTHESIS_LIFECYCLE_PATH
        ),
        checkpoint_due_router_path: str | Path | None = (
            DEFAULT_CHECKPOINT_DUE_ROUTER_PATH
        ),
        outcome_lifecycle_path: str | Path | None = (
            DEFAULT_OUTCOME_LIFECYCLE_PATH
        ),
        evidence_refresh_path: str | Path | None = DEFAULT_EVIDENCE_REFRESH_PATH,
        verified_source_router_path: str | Path | None = (
            DEFAULT_VERIFIED_SOURCE_ROUTER_PATH
        ),
        output_dir: str | Path = "reports/dean_os/chief_review_index",
    ):
        self.review_index_path = Path(review_index_path)
        self.hypothesis_lifecycle_path = (
            Path(hypothesis_lifecycle_path)
            if hypothesis_lifecycle_path is not None
            else None
        )
        self.checkpoint_due_router_path = (
            Path(checkpoint_due_router_path)
            if checkpoint_due_router_path is not None
            else None
        )
        self.outcome_lifecycle_path = (
            Path(outcome_lifecycle_path)
            if outcome_lifecycle_path is not None
            else None
        )
        self.evidence_refresh_path = (
            Path(evidence_refresh_path) if evidence_refresh_path is not None else None
        )
        self.verified_source_router_path = (
            Path(verified_source_router_path)
            if verified_source_router_path is not None
            else None
        )
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        review_index = self._load_review_index()
        hypothesis_lifecycle = self._load_hypothesis_lifecycle()
        checkpoint_due_router = self._load_checkpoint_due_router()
        outcome_lifecycle = self._load_outcome_lifecycle()
        evidence_refresh = self._load_evidence_refresh()
        verified_source_router = self._load_verified_source_router()
        lifecycle_inbox = dict(hypothesis_lifecycle.get("review_inbox") or {})
        checkpoint_inbox = dict(
            checkpoint_due_router.get("chief_review_inbox") or {}
        )
        outcome_inbox = dict(outcome_lifecycle.get("review_inbox") or {})
        decision = _apply_verified_source_router_overlay(
            _apply_evidence_refresh_overlay(
                _apply_outcome_lifecycle_overlay(
                    _apply_checkpoint_due_overlay(
                        _apply_hypothesis_lifecycle_overlay(
                            classify_review_index(review_index), hypothesis_lifecycle
                        ),
                        checkpoint_due_router,
                    ),
                    outcome_lifecycle,
                ),
                evidence_refresh,
            ),
            verified_source_router,
        )
        payload = {
            "run_id": _run_id("chief_review_index"),
            "created_at": utc_now_iso(),
            "mode": "chief_review_index",
            "input": {
                "review_index_path": str(self.review_index_path),
                "review_index_run_id": review_index.get("run_id"),
                "hypothesis_lifecycle_path": (
                    str(self.hypothesis_lifecycle_path)
                    if self.hypothesis_lifecycle_path is not None
                    else None
                ),
                "hypothesis_lifecycle_run_id": hypothesis_lifecycle.get("run_id"),
                "checkpoint_due_router_path": (
                    str(self.checkpoint_due_router_path)
                    if self.checkpoint_due_router_path is not None
                    else None
                ),
                "checkpoint_due_router_run_id": checkpoint_due_router.get("run_id"),
                "outcome_lifecycle_path": (
                    str(self.outcome_lifecycle_path)
                    if self.outcome_lifecycle_path is not None
                    else None
                ),
                "outcome_lifecycle_run_id": outcome_lifecycle.get("run_id"),
                "evidence_refresh_path": (
                    str(self.evidence_refresh_path)
                    if self.evidence_refresh_path is not None
                    else None
                ),
                "evidence_refresh_run_id": evidence_refresh.get("run_id"),
                "verified_source_router_path": (
                    str(self.verified_source_router_path)
                    if self.verified_source_router_path is not None
                    else None
                ),
                "verified_source_router_run_id": verified_source_router.get("run_id"),
            },
            "decision": decision,
            "review_index_summary": review_index.get("summary", {}),
            "entries": review_index.get("entries", []),
            "hypothesis_lifecycle_inbox": lifecycle_inbox,
            "checkpoint_due_inbox": checkpoint_inbox,
            "outcome_lifecycle_inbox": outcome_inbox,
            "evidence_refresh": evidence_refresh,
            "verified_source_router": verified_source_router,
            "safety": {
                "review_only": True,
                "read_existing_artifacts_only": True,
                "approval_performed": False,
                "live_execution_allowed": False,
                "broker_access_performed": False,
                "production_config_write_performed": False,
                "learning_write_performed": False,
                "training_or_tuning_run_performed": False,
                "model_promotion_performed": False,
            },
        }

        if save:
            markdown = render_chief_review_index_markdown(payload)
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=markdown,
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths

        return json_ready(payload)

    def _load_review_index(self) -> dict[str, Any]:
        if not self.review_index_path.exists():
            return {
                "run_id": None,
                "mode": "review_index",
                "summary": {
                    "source_count": 0,
                    "available_count": 0,
                    "missing_count": 1,
                    "ready_for_chief_review": False,
                    "missing_sources": ["review_index"],
                },
                "entries": [],
                "errors": [f"Review index not found: {self.review_index_path}"],
            }

        try:
            payload = json.loads(self.review_index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "mode": "review_index",
                "summary": {
                    "source_count": 0,
                    "available_count": 0,
                    "missing_count": 1,
                    "ready_for_chief_review": False,
                    "missing_sources": ["review_index"],
                },
                "entries": [],
                "errors": [repr(exc)],
            }

        if not isinstance(payload, dict):
            return {
                "run_id": None,
                "mode": "review_index",
                "summary": {
                    "source_count": 0,
                    "available_count": 0,
                    "missing_count": 1,
                    "ready_for_chief_review": False,
                    "missing_sources": ["review_index"],
                },
                "entries": [],
                "errors": ["Review index JSON is not an object"],
            }

        return payload

    def _load_hypothesis_lifecycle(self) -> dict[str, Any]:
        path = self.hypothesis_lifecycle_path
        if path is None or not path.is_file():
            return {
                "run_id": None,
                "summary": {"status": "not_available"},
                "review_inbox": {
                    "status": "not_available",
                    "blockers": [],
                    "proposed_contracts": [],
                    "pending_decisions": [],
                },
                "safety": {"can_trade": False},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "summary": {"status": "unreadable", "error": repr(exc)},
                "review_inbox": {
                    "status": "unreadable",
                    "blockers": [
                        {
                            "hypothesis_id": "lifecycle_artifact",
                            "blockers": ["hypothesis_lifecycle_artifact_unreadable"],
                        }
                    ],
                    "proposed_contracts": [],
                    "pending_decisions": [],
                },
                "safety": {"can_trade": False},
            }
        return payload if isinstance(payload, dict) else {}

    def _load_checkpoint_due_router(self) -> dict[str, Any]:
        path = self.checkpoint_due_router_path
        if path is None or not path.is_file():
            return {
                "run_id": None,
                "summary": {"status": "not_available"},
                "chief_review_inbox": {
                    "status": "not_available",
                    "matured_checkpoints": [],
                    "data_accrual_actions": [],
                    "pending_decisions": [],
                    "future_checkpoints_are_operator_actions": False,
                },
                "safety": {"can_trade": False},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "summary": {"status": "unreadable", "error": repr(exc)},
                "chief_review_inbox": {
                    "status": "unreadable",
                    "matured_checkpoints": [],
                    "data_accrual_actions": [],
                    "pending_decisions": [],
                    "future_checkpoints_are_operator_actions": False,
                },
                "safety": {"can_trade": False},
            }
        return payload if isinstance(payload, dict) else {}

    def _load_outcome_lifecycle(self) -> dict[str, Any]:
        path = self.outcome_lifecycle_path
        if path is None or not path.is_file():
            return {
                "run_id": None,
                "summary": {"status": "not_available"},
                "review_inbox": {
                    "status": "not_available",
                    "data_actions": [],
                    "outcome_packets": [],
                    "learning_proposals": [],
                    "pending_decisions": [],
                },
                "safety": {"can_trade": False},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "summary": {"status": "unreadable", "error": repr(exc)},
                "review_inbox": {
                    "status": "unreadable",
                    "data_actions": [],
                    "outcome_packets": [],
                    "learning_proposals": [],
                    "pending_decisions": [],
                },
                "safety": {"can_trade": False},
            }
        return payload if isinstance(payload, dict) else {}

    def _load_evidence_refresh(self) -> dict[str, Any]:
        path = self.evidence_refresh_path
        if path is None or not path.is_file():
            return {
                "run_id": None,
                "summary": {"status": "not_available"},
                "refresh_failure": None,
                "safety": {"can_trade": False},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "summary": {"status": "unreadable", "error": repr(exc)},
                "refresh_failure": None,
                "safety": {"can_trade": False},
            }
        return payload if isinstance(payload, dict) else {}

    def _load_verified_source_router(self) -> dict[str, Any]:
        path = self.verified_source_router_path
        if path is None or not path.is_file():
            return {
                "run_id": None,
                "summary": {"status": "not_available"},
                "next_system_actions": [],
                "safety": {"can_trade": False},
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "run_id": None,
                "summary": {"status": "unreadable", "error": repr(exc)},
                "next_system_actions": [],
                "safety": {"can_trade": False},
            }
        return payload if isinstance(payload, dict) else {}


def classify_review_index(review_index: dict[str, Any]) -> dict[str, Any]:
    summary = review_index.get("summary") or {}
    entries = list(review_index.get("entries") or [])
    reasons: list[str] = []
    risks: list[str] = [
        "ChiefReviewIndex is review-only and cannot approve or execute actions.",
        "Human review remains required before tuning, promotion, config changes, or any real-money action.",
    ]
    next_actions: list[str] = []

    dangerous = _dangerous_safety_entries(entries)
    if dangerous:
        return {
            "decision": "blocked",
            "verdict": "blocked",
            "confidence": 0.9,
            "data_quality_score": 0.4,
            "reasons": [
                "One or more review artifacts reported unsafe execution/config/learning side effects.",
                *dangerous,
            ],
            "risks": risks,
            "next_actions": [
                "Stop integration and inspect unsafe artifact sources before continuing.",
                "Do not use any proposal from the current review set.",
            ],
            "autonomy_recommendation": "Pause all automation until unsafe artifact flags are resolved.",
        }

    available_count = int(summary.get("available_count") or 0)
    missing_count = int(summary.get("missing_count") or 0)
    if available_count == 0:
        return {
            "decision": "needs_more_evidence",
            "verdict": "needs_more_data",
            "confidence": 0.75,
            "data_quality_score": 0.25,
            "reasons": ["No available review artifacts were found."],
            "risks": risks,
            "next_actions": ["Run domain analyst and/or pipeline tuning controller review artifacts first."],
            "autonomy_recommendation": "Pause automation until review artifacts exist.",
        }

    domain_entries = [entry for entry in entries if entry.get("source_name") == "domain_analyst" and entry.get("available")]
    tuning_entries = [entry for entry in entries if entry.get("source_name") == "pipeline_tuning_controller" and entry.get("available")]
    model_case_entries = [entry for entry in entries if entry.get("source_name") == "pipeline_model_case" and entry.get("available")]
    model_feedback_entries = [entry for entry in entries if entry.get("source_name") == "pipeline_model_feedback" and entry.get("available")]
    other_available = [
        entry
        for entry in entries
        if entry.get("available")
        and entry not in domain_entries
        and entry not in tuning_entries
        and entry not in model_case_entries
        and entry not in model_feedback_entries
    ]

    domain_state = _domain_state(domain_entries)
    tuning_state = _tuning_state(tuning_entries)
    model_case_state = _model_case_state(model_case_entries)
    model_feedback_state = _model_feedback_state(
        model_feedback_entries
    )

    reasons.extend(domain_state["reasons"])
    reasons.extend(tuning_state["reasons"])
    reasons.extend(model_case_state["reasons"])
    reasons.extend(model_feedback_state["reasons"])

    if missing_count:
        reasons.append(f"{missing_count} expected review artifact(s) are missing.")
        next_actions.append("Run missing review producers or explicitly accept partial review coverage.")

    if model_case_state["blocked"]:
        decision = "model_candidate_blocked"
        verdict = "blocked"
        next_actions.append(
            "Retain the negative model case and wait for accepted new "
            "forward data; do not tune the same folds."
        )
        next_actions.append(
            "Continue unrelated pipeline engineering, analyzer review, "
            "research, and safe forward-data collection."
        )
    elif tuning_state["blocked"]:
        decision = "blocked"
        verdict = "blocked"
        next_actions.append("Resolve blocked tuning/control-surface state before approving experiments.")
    elif model_feedback_state["blocked"]:
        decision = "needs_more_evidence"
        verdict = "needs_more_data"
        next_actions.append(
            "Repair the model-feedback labels or source bindings before "
            "reviewing proposed lessons."
        )
    elif model_case_state["needs_more_evidence"]:
        decision = "needs_more_evidence"
        verdict = "needs_more_data"
        next_actions.append(
            "Repair the model-case evidence binding before using it in "
            "Chief Review."
        )
    elif domain_state["needs_more_evidence"]:
        decision = "needs_more_evidence"
        verdict = "needs_more_data"
        next_actions.append("Backfill missing domain/ticker evidence before using the analyst thesis.")
    elif tuning_state["validate_before_tuning"]:
        decision = "validate_before_tuning"
        verdict = "caution"
        next_actions.append("Review tuning plan, validate metrics/control surface, and dry-run before approval.")
    elif model_case_state["caution"]:
        decision = "validate_before_tuning"
        verdict = "caution"
        next_actions.append(
            "Review the model caution case before any tuning proposal."
        )
    elif (
        domain_state["ready"]
        or model_case_state["ready"]
        or model_feedback_state["pending_manual_feedback"]
        or model_feedback_state["has_proposal_candidates"]
        or other_available
    ):
        decision = "ready_for_human_review"
        verdict = "caution" if missing_count else "clear"
        next_actions.append("Human reviewer can inspect available artifacts and decide mark-reviewed / needs-more-data.")
    else:
        decision = "needs_more_evidence"
        verdict = "needs_more_data"
        next_actions.append("Provide stronger review artifacts before ChiefReview acceptance.")

    if not reasons:
        reasons.append("Available review artifacts were summarized without blockers.")

    confidence = 0.65 + min(0.2, available_count * 0.05)
    if missing_count:
        confidence = min(confidence, 0.7)

    data_quality_score = 0.35 + min(0.45, available_count * 0.15)
    if decision in {
        "needs_more_evidence",
        "blocked",
        "model_candidate_blocked",
    }:
        data_quality_score = min(data_quality_score, 0.55)

    return {
        "decision": decision,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "data_quality_score": round(data_quality_score, 3),
        "reasons": reasons,
        "risks": risks,
        "next_actions": next_actions,
        "domain_state": domain_state,
        "tuning_state": tuning_state,
        "model_case_state": model_case_state,
        "model_feedback_state": model_feedback_state,
        "autonomy_recommendation": _autonomy_recommendation(decision),
    }


def render_chief_review_index_markdown(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") or {}
    lines = [
        "# DEAN-OS Chief Review Index Decision",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Created at: `{payload.get('created_at')}`",
        f"- Decision: `{decision.get('decision')}`",
        f"- Verdict: `{decision.get('verdict')}`",
        f"- Confidence: `{decision.get('confidence')}`",
        f"- Data quality score: `{decision.get('data_quality_score')}`",
        "",
        "## Hypothesis Lifecycle Inbox",
        "",
    ]
    inbox = payload.get("hypothesis_lifecycle_inbox") or {}
    blockers = list(inbox.get("blockers") or [])
    contracts = list(inbox.get("proposed_contracts") or [])
    pending = list(inbox.get("pending_decisions") or [])
    lines.extend(
        [
            f"- Status: `{inbox.get('status') or 'not_available'}`",
            f"- Blockers: {len(blockers)}",
            f"- Proposed contracts: {len(contracts)}",
            f"- Pending decisions: {len(pending)}",
            "",
        ]
    )
    for item in blockers:
        lines.append(
            f"- BLOCKED `{item.get('hypothesis_id')}`: "
            + ", ".join(item.get("blockers") or [])
        )
    for item in contracts:
        lines.append(
            f"- CONTRACT `{item.get('hypothesis_id')}`: "
            f"direction={item.get('expected_direction')} "
            f"horizon={item.get('horizon_days')}d "
            f"band={item.get('neutral_band_absolute_return')}"
        )
    for item in pending:
        lines.append(
            f"- DECISION `{item.get('hypothesis_id')}`: "
            f"{item.get('decision_type')} options="
            + ",".join(item.get("allowed_decisions") or [])
        )
    checkpoint_inbox = payload.get("checkpoint_due_inbox") or {}
    matured_checkpoints = list(checkpoint_inbox.get("matured_checkpoints") or [])
    data_actions = list(checkpoint_inbox.get("data_accrual_actions") or [])
    if matured_checkpoints or data_actions:
        lines.extend(["", "## Matured Checkpoints", ""])
        for item in matured_checkpoints:
            lines.append(
                f"- OUTCOME REVIEW `{item.get('task_id')}`: "
                f"{item.get('horizon_days')}d, session={item.get('checkpoint_session')}"
            )
        for item in data_actions:
            lines.append(
                f"- WAITING FOR DATA `{item.get('task_id')}`: "
                "no verified checkpoint session yet"
            )
    outcome_inbox = payload.get("outcome_lifecycle_inbox") or {}
    outcome_packets = list(outcome_inbox.get("outcome_packets") or [])
    outcome_data_actions = list(outcome_inbox.get("data_actions") or [])
    outcome_pending = list(outcome_inbox.get("pending_decisions") or [])
    if outcome_packets or outcome_data_actions or outcome_pending:
        lines.extend(["", "## Outcome Lifecycle", ""])
        for item in outcome_data_actions:
            lines.append(
                f"- DATA `{item.get('task_id')}`: verified checkpoint evidence required"
            )
        for item in outcome_packets:
            lines.append(
                f"- PACKET `{item.get('task_id')}`: "
                f"result={item.get('result_label')} role={item.get('checkpoint_role')}"
            )
        for item in outcome_pending:
            lines.append(
                f"- CAUSAL REVIEW `{item.get('task_id')}`: "
                f"{item.get('decision_type')}"
            )
    refresh = payload.get("evidence_refresh") or {}
    refresh_summary = refresh.get("summary") or {}
    refresh_failure = refresh.get("refresh_failure")
    if refresh.get("run_id"):
        lines.extend(["", "## Evidence Refresh", ""])
        lines.append(f"- Status: `{refresh_summary.get('status')}`")
        lines.append(
            f"- Network refresh executed: `{refresh_summary.get('refresh_executed')}`"
        )
        if refresh_failure:
            lines.append(
                f"- FAILURE `{refresh_failure.get('error_type')}`: "
                f"{refresh_failure.get('error')}"
            )
            lines.append(
                f"- Next system action: {refresh_failure.get('next_action')}"
            )
    source_router = payload.get("verified_source_router") or {}
    source_summary = source_router.get("summary") or {}
    if source_router.get("run_id"):
        lines.extend(["", "## Verified Source Route", ""])
        lines.append(f"- Status: `{source_summary.get('status')}`")
        for item in source_router.get("next_system_actions") or []:
            lines.append(
                f"- `{item.get('action_type')}` task={item.get('task_id')} "
                f"tickers={','.join(item.get('required_tickers') or [])}"
            )
    lines.extend(
        [
        "",
        "## Reasons",
        "",
        ]
    )

    for reason in decision.get("reasons") or []:
        lines.append(f"- {reason}")

    lines.extend(["", "## Next Actions", ""])
    for action in decision.get("next_actions") or []:
        lines.append(f"- {action}")

    lines.extend(["", "## Risks", ""])
    for risk in decision.get("risks") or []:
        lines.append(f"- {risk}")

    lines.extend(["", "## Entries", "", "| Source | Available | Status | Recommendation | Path |", "|---|---:|---|---|---|"])
    for entry in payload.get("entries") or []:
        lines.append(
            "| {source} | {available} | {status} | {recommendation} | `{path}` |".format(
                source=entry.get("source_name"),
                available=entry.get("available"),
                status=entry.get("status"),
                recommendation=entry.get("recommendation"),
                path=entry.get("path"),
            )
        )

    lines.extend(["", "## Safety", ""])
    safety = payload.get("safety") or {}
    for key in sorted(safety):
        lines.append(f"- {key}: `{safety[key]}`")

    lines.extend(["", "## Autonomy Recommendation", "", str(decision.get("autonomy_recommendation") or "")])
    return "\n".join(lines).strip() + "\n"


def _apply_hypothesis_lifecycle_overlay(
    base: dict[str, Any], lifecycle: dict[str, Any]
) -> dict[str, Any]:
    decision = dict(base)
    inbox = dict(lifecycle.get("review_inbox") or {})
    blockers = list(inbox.get("blockers") or [])
    contracts = list(inbox.get("proposed_contracts") or [])
    pending = list(inbox.get("pending_decisions") or [])
    state = {
        "available": bool(lifecycle.get("run_id")),
        "status": inbox.get("status") or "not_available",
        "blocker_count": len(blockers),
        "proposed_contract_count": len(contracts),
        "pending_decision_count": len(pending),
    }
    decision["hypothesis_lifecycle_state"] = state
    if not state["available"] and not blockers:
        return decision
    reasons = list(decision.get("reasons") or [])
    actions = list(decision.get("next_actions") or [])
    if lifecycle.get("safety", {}).get("can_trade") is True:
        decision.update(
            {
                "decision": "blocked",
                "verdict": "blocked",
                "autonomy_recommendation": "Pause lifecycle automation because an unsafe authority flag was detected.",
            }
        )
        reasons.append("Hypothesis lifecycle artifact reported can_trade=true.")
        actions.insert(0, "Inspect the unsafe lifecycle artifact before continuing.")
    elif blockers and decision.get("decision") not in {
        "blocked",
        "model_candidate_blocked",
    }:
        decision.update(
            {
                "decision": "hypothesis_measurement_blocked",
                "verdict": "blocked",
                "autonomy_recommendation": "Let the measurement agent collect missing inputs; do not approve or register the blocked hypotheses.",
            }
        )
        reasons.append(
            f"{len(blockers)} hypothesis measurement blocker set(s) require resolution."
        )
        actions.insert(0, "Resolve the listed measurement blockers or defer/reject the affected hypothesis.")
    elif pending and decision.get("decision") not in {
        "blocked",
        "model_candidate_blocked",
    }:
        decision.update(
            {
                "decision": "hypothesis_review_required",
                "verdict": "caution",
                "autonomy_recommendation": "Human reviews only the proposed contracts and pending hypothesis dispositions; no automatic approval.",
            }
        )
        reasons.append(
            f"{len(pending)} hypothesis decision(s) are pending after measurement preparation."
        )
        actions.insert(0, "Review the compact lifecycle inbox and choose an allowed disposition.")
    decision["reasons"] = list(dict.fromkeys(reasons))
    decision["next_actions"] = list(dict.fromkeys(actions))
    return decision


def _apply_checkpoint_due_overlay(
    base: dict[str, Any], router: dict[str, Any]
) -> dict[str, Any]:
    decision = dict(base)
    inbox = dict(router.get("chief_review_inbox") or {})
    matured = list(inbox.get("matured_checkpoints") or [])
    waiting = list(inbox.get("data_accrual_actions") or [])
    state = {
        "available": bool(router.get("run_id")),
        "status": inbox.get("status") or "not_available",
        "matured_checkpoint_count": len(matured),
        "waiting_for_verified_data_count": len(waiting),
        "pending_decision_count": len(inbox.get("pending_decisions") or []),
        "future_checkpoints_are_operator_actions": bool(
            inbox.get("future_checkpoints_are_operator_actions")
        ),
    }
    decision["checkpoint_due_state"] = state
    if not state["available"]:
        return decision
    reasons = list(decision.get("reasons") or [])
    actions = list(decision.get("next_actions") or [])
    if router.get("safety", {}).get("can_trade") is True:
        decision.update(
            {
                "decision": "blocked",
                "verdict": "blocked",
                "autonomy_recommendation": (
                    "Pause checkpoint routing because an unsafe authority flag was detected."
                ),
            }
        )
        reasons.append("Checkpoint due router reported can_trade=true.")
        actions.insert(0, "Inspect the unsafe checkpoint router artifact.")
    elif matured and decision.get("decision") not in {
        "blocked",
        "model_candidate_blocked",
        "hypothesis_measurement_blocked",
        "hypothesis_review_required",
    }:
        decision.update(
            {
                "decision": "checkpoint_outcome_review_required",
                "verdict": "caution",
                "autonomy_recommendation": (
                    "Review only matured checkpoint outcomes; keep scoring, learning, and trading disabled."
                ),
            }
        )
        reasons.append(
            f"{len(matured)} checkpoint(s) have both reached their due time and acquired verified checkpoint data."
        )
        actions.insert(0, "Review the matured checkpoint outcome packet.")
    if waiting:
        reasons.append(
            f"{len(waiting)} due checkpoint(s) are waiting for verified data and are not ready for judgment."
        )
    decision["reasons"] = list(dict.fromkeys(reasons))
    decision["next_actions"] = list(dict.fromkeys(actions))
    return decision


def _apply_outcome_lifecycle_overlay(
    base: dict[str, Any], lifecycle: dict[str, Any]
) -> dict[str, Any]:
    decision = dict(base)
    inbox = dict(lifecycle.get("review_inbox") or {})
    data_actions = list(inbox.get("data_actions") or [])
    packets = list(inbox.get("outcome_packets") or [])
    proposals = list(inbox.get("learning_proposals") or [])
    pending = list(inbox.get("pending_decisions") or [])
    state = {
        "available": bool(lifecycle.get("run_id")),
        "status": inbox.get("status") or "not_available",
        "data_action_count": len(data_actions),
        "outcome_packet_count": len(packets),
        "learning_proposal_count": len(proposals),
        "pending_decision_count": len(pending),
    }
    decision["outcome_lifecycle_state"] = state
    if not state["available"]:
        return decision
    reasons = list(decision.get("reasons") or [])
    actions = list(decision.get("next_actions") or [])
    if lifecycle.get("safety", {}).get("can_trade") is True:
        decision.update(
            {
                "decision": "blocked",
                "verdict": "blocked",
                "autonomy_recommendation": (
                    "Pause outcome lifecycle because an unsafe authority flag was detected."
                ),
            }
        )
        reasons.append("Replay outcome lifecycle reported can_trade=true.")
        actions.insert(0, "Inspect the unsafe outcome lifecycle artifact.")
    elif pending and decision.get("decision") not in {
        "blocked",
        "model_candidate_blocked",
        "hypothesis_measurement_blocked",
        "hypothesis_review_required",
    }:
        decision.update(
            {
                "decision": "primary_outcome_causal_review_required",
                "verdict": "caution",
                "autonomy_recommendation": (
                    "Review the machine diagnosis and causal attribution; no automatic rule promotion."
                ),
            }
        )
        reasons.append(
            f"{len(pending)} primary outcome causal disposition(s) require review."
        )
        actions.insert(0, "Review the primary outcome and reverse-analysis packet.")
    if data_actions:
        reasons.append(
            f"{len(data_actions)} outcome lifecycle task(s) are waiting for verified evidence."
        )
    decision["reasons"] = list(dict.fromkeys(reasons))
    decision["next_actions"] = list(dict.fromkeys(actions))
    return decision


def _apply_evidence_refresh_overlay(
    base: dict[str, Any], refresh: dict[str, Any]
) -> dict[str, Any]:
    decision = dict(base)
    summary = refresh.get("summary") or {}
    failure = refresh.get("refresh_failure")
    state = {
        "available": bool(refresh.get("run_id")),
        "status": summary.get("status") or "not_available",
        "refresh_executed": bool(summary.get("refresh_executed")),
        "lifecycle_rerun": bool(summary.get("lifecycle_rerun")),
        "failure_recorded": failure is not None,
        "automatic_retry_allowed": False,
    }
    decision["evidence_refresh_state"] = state
    if not state["available"]:
        return decision
    reasons = list(decision.get("reasons") or [])
    actions = list(decision.get("next_actions") or [])
    if refresh.get("safety", {}).get("can_trade") is True:
        decision.update(
            {
                "decision": "blocked",
                "verdict": "blocked",
                "autonomy_recommendation": (
                    "Pause evidence refresh because an unsafe authority flag was detected."
                ),
            }
        )
        reasons.append("Evidence refresh controller reported can_trade=true.")
    elif failure:
        reasons.append(
            "The allowlisted evidence refresh failed without changing the hypothesis outcome."
        )
        next_action = failure.get("next_action")
        if next_action:
            actions.append(str(next_action))
    decision["reasons"] = list(dict.fromkeys(reasons))
    decision["next_actions"] = list(dict.fromkeys(actions))
    return decision


def _apply_verified_source_router_overlay(
    base: dict[str, Any], router: dict[str, Any]
) -> dict[str, Any]:
    decision = dict(base)
    summary = router.get("summary") or {}
    actions = list(router.get("next_system_actions") or [])
    state = {
        "available": bool(router.get("run_id")),
        "status": summary.get("status") or "not_available",
        "next_system_action_count": len(actions),
        "automatic_provider_loop_allowed": False,
        "ready_local_snapshot_count": int(
            summary.get("ready_local_snapshot_count") or 0
        ),
    }
    decision["verified_source_router_state"] = state
    if not state["available"]:
        return decision
    reasons = list(decision.get("reasons") or [])
    next_actions = list(decision.get("next_actions") or [])
    if router.get("safety", {}).get("can_trade") is True:
        decision.update(
            {
                "decision": "blocked",
                "verdict": "blocked",
                "autonomy_recommendation": (
                    "Pause source routing because an unsafe authority flag was detected."
                ),
            }
        )
        reasons.append("Verified source router reported can_trade=true.")
    elif state["status"] == "awaiting_operator_supplied_verified_snapshot":
        reasons.append(
            "The ranked network provider is exhausted; a validated local snapshot is the next bounded evidence route."
        )
        for item in actions:
            if item.get("action_type") == "supply_local_verified_market_snapshot":
                next_actions.append(
                    "Supply a point-in-time market snapshot for: "
                    + ", ".join(item.get("required_tickers") or [])
                    + "."
                )
    decision["reasons"] = list(dict.fromkeys(reasons))
    decision["next_actions"] = list(dict.fromkeys(next_actions))
    return decision


def _domain_state(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "present": False,
            "ready": False,
            "needs_more_evidence": False,
            "reasons": ["Domain analyst artifact is not available."],
        }

    entry = entries[-1]
    recommendation = str(entry.get("recommendation") or "")
    status = str(entry.get("status") or "")
    summary = entry.get("summary") or {}

    needs_more = recommendation in {"needs_more_data", "blocked"} or status in {"basket_blocked", "needs_more_data"}
    ready = recommendation in {"ready_for_review", "partial_ready_for_review"} or status in {
        "basket_ready_for_review",
        "partial_basket_ready",
    }

    reasons = [
        f"Domain analyst artifact status={status}, recommendation={recommendation}.",
    ]
    if summary:
        reasons.append(
            "Domain summary: domain={domain}, stance={stance}, direction={direction}, confidence={confidence}.".format(
                domain=summary.get("domain_id"),
                stance=summary.get("thesis_stance"),
                direction=summary.get("expected_direction"),
                confidence=summary.get("confidence"),
            )
        )

    return {
        "present": True,
        "ready": ready,
        "needs_more_evidence": needs_more,
        "status": status,
        "recommendation": recommendation,
        "summary": summary,
        "reasons": reasons,
    }


def _tuning_state(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "present": False,
            "validate_before_tuning": False,
            "blocked": False,
            "reasons": ["Pipeline tuning controller artifact is not available."],
        }

    entry = entries[-1]
    recommendation = str(entry.get("recommendation") or "")
    status = str(entry.get("status") or "")
    summary = entry.get("summary") or {}

    blocked = status == "blocked" or recommendation == "blocked"
    validate_before = status in {"tuning_candidate", "validate_first"} or recommendation in {
        "tuning_candidate",
        "validate_first",
        "caution",
    }

    reasons = [f"Pipeline tuning artifact status={status}, recommendation={recommendation}."]
    if summary:
        reasons.append(
            "Tuning summary: plan_status={plan_status}, target={target}, planes={planes}, proposals={proposals}.".format(
                plan_status=summary.get("plan_status"),
                target=summary.get("target"),
                planes=summary.get("plane_count"),
                proposals=summary.get("action_proposal_count"),
            )
        )

    return {
        "present": True,
        "validate_before_tuning": validate_before,
        "blocked": blocked,
        "status": status,
        "recommendation": recommendation,
        "summary": summary,
        "reasons": reasons,
    }


def _model_case_state(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not entries:
        return {
            "present": False,
            "blocked": False,
            "needs_more_evidence": False,
            "caution": False,
            "ready": False,
            "reasons": ["Pipeline model case artifact is not available."],
        }

    entry = entries[-1]
    status = str(entry.get("status") or "")
    summary = entry.get("summary") or {}
    blocked = status == "evaluation_block_case_ready"
    needs_more_evidence = status in {
        "pipeline_model_case_rejected",
        "case_binding_invalid",
        "unknown",
    }
    caution = status == "evaluation_caution_case_ready"
    ready = status == "evaluation_clear_case_ready"
    reasons = [
        "Pipeline model case status={status}, classification={classification}, "
        "result={result}.".format(
            status=status,
            classification=summary.get("case_classification"),
            result=summary.get("result_label"),
        )
    ]
    blocked_planes = summary.get("blocked_metric_planes") or []
    root_causes = summary.get("root_cause_categories") or []
    if blocked_planes:
        reasons.append(
            "Blocked model metric planes: "
            + ", ".join(str(item) for item in blocked_planes)
            + "."
        )
    if root_causes:
        reasons.append(
            "Structured root-cause categories: "
            + ", ".join(str(item) for item in root_causes)
            + "."
        )
    return {
        "present": True,
        "blocked": blocked,
        "needs_more_evidence": needs_more_evidence,
        "caution": caution,
        "ready": ready,
        "status": status,
        "summary": summary,
        "reasons": reasons,
    }


def _model_feedback_state(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not entries:
        return {
            "present": False,
            "blocked": False,
            "pending_manual_feedback": False,
            "has_proposal_candidates": False,
            "reasons": [
                "Pipeline model feedback artifact is not available."
            ],
        }
    entry = entries[-1]
    status = str(entry.get("status") or "")
    summary = entry.get("summary") or {}
    pending = (
        status
        == "pipeline_model_feedback_ready_pending_manual_feedback"
    )
    has_candidates = (
        status == "pipeline_model_feedback_ready_with_candidates"
    )
    blocked = status == "pipeline_model_feedback_blocked"
    reasons = [
        "Pipeline model feedback status={status}, records={records}, "
        "proposal_candidates={candidates}.".format(
            status=status,
            records=summary.get("manual_feedback_record_count"),
            candidates=summary.get(
                "learning_candidate_proposal_count"
            ),
        )
    ]
    if pending:
        reasons.append(
            "The negative model case is awaiting optional human "
            "feedback; no learning update is implied."
        )
    if has_candidates:
        reasons.append(
            "Model feedback candidates are proposal-only and cannot use "
            "the directional analyst learning apply loop."
        )
    if blocked:
        reasons.append(
            "Repair feedback labels or source bindings before reviewing "
            "candidate lessons."
        )
    return {
        "present": True,
        "blocked": blocked,
        "pending_manual_feedback": pending,
        "has_proposal_candidates": has_candidates,
        "status": status,
        "summary": summary,
        "reasons": reasons,
    }


def _dangerous_safety_entries(entries: list[dict[str, Any]]) -> list[str]:
    dangerous_flags = {
        "live_execution_allowed",
        "broker_access_performed",
        "production_config_write_performed",
        "learning_write_performed",
        "training_or_tuning_run_performed",
        "model_promotion_performed",
        "approval_performed",
    }
    messages: list[str] = []
    for entry in entries:
        safety = entry.get("safety") or {}
        for flag in dangerous_flags:
            if safety.get(flag) is True:
                messages.append(f"{entry.get('source_name')} has unsafe flag {flag}=true")
    return messages


def _autonomy_recommendation(decision: str) -> str:
    if decision == "blocked":
        return "Pause all automation until blockers are resolved."
    if decision == "model_candidate_blocked":
        return (
            "Block tuning, promotion, recommendations, and trading for "
            "this model candidate; keep unrelated review, research, "
            "pipeline engineering, and forward-data work active."
        )
    if decision == "needs_more_evidence":
        return "Keep paper/review mode only and collect missing evidence."
    if decision == "validate_before_tuning":
        return "Allow review of tuning proposal only after validation/dry-run; no production changes."
    if decision == "ready_for_human_review":
        return "Human reviewer can inspect artifacts; no automatic approval or execution."
    return "Remain in review-only mode."


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
