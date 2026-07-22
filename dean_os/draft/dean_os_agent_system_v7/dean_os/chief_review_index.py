from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REVIEW_INDEX_PATH = "reports/dean_os/review_index/latest.json"


class ChiefReviewIndexBuilder:
    """Consolidates ReviewIndex artifacts into a supervised review decision.

    This is a review-only synthesis layer. It reads the ReviewIndex artifact,
    classifies the current state, and writes a human-readable review artifact.
    It does not approve, execute, tune, train, promote, or write learning memory.
    """

    def __init__(
        self,
        review_index_path: str | Path = DEFAULT_REVIEW_INDEX_PATH,
        output_dir: str | Path = "reports/dean_os/chief_review_index",
    ):
        self.review_index_path = Path(review_index_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        review_index = self._load_review_index()
        decision = classify_review_index(review_index)
        payload = {
            "run_id": _run_id("chief_review_index"),
            "created_at": utc_now_iso(),
            "mode": "chief_review_index",
            "input": {
                "review_index_path": str(self.review_index_path),
                "review_index_run_id": review_index.get("run_id"),
            },
            "decision": decision,
            "review_index_summary": review_index.get("summary", {}),
            "entries": review_index.get("entries", []),
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
        "## Reasons",
        "",
    ]

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
