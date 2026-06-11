from __future__ import annotations

from collections import Counter
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineActionProposal, PipelineReport


class ChiefReviewAgent(BaseAgent):
    """Synthesizes pipeline, specialist, memory, and proposal state for human review."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        review = build_chief_review(context, autonomy_mode=str(self.config.get("autonomy_mode", "paper_supervised")))
        context.metadata["chief_review"] = review

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=review["verdict"],
            confidence=review["confidence"],
            data_quality_score=review["data_quality_score"],
            signal_strength=review["signal_strength"],
            reasons=review["reasons"],
            risks=review["risks"],
            blind_spots=[
                "ChiefReviewAgent summarizes available evidence only; it does not approve trades, execute paper orders, or modify config."
            ],
            evidence=[
                self.evidence("metric", "chief_review", "decision", review["decision"]),
                self.evidence("metric", "chief_review", "stale_sources", review["pipeline_state"]["stale_sources"]),
                self.evidence("metric", "chief_review", "model_failures", review["pipeline_state"]["model_failures"]),
                self.evidence("operation", "context.action_proposals", "proposal_count", review["operations"]["proposal_count"]),
                self.evidence("research_note", "context.research_notes", "note_count", review["specialists"]["note_count"]),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=review,
        )


def build_chief_review(context: MarketContext, autonomy_mode: str = "paper_supervised") -> dict[str, Any]:
    review_snapshot = _as_dict(context.metadata.get("review_snapshot"))
    pipeline_state = _pipeline_state(context, review_snapshot)
    operations = _operations_state(context, review_snapshot)
    specialists = _specialist_state(context, review_snapshot)
    memory = _memory_state(context, review_snapshot)
    decision, verdict, reasons, risks, next_actions = _classify_review(
        pipeline_state=pipeline_state,
        operations=operations,
        specialists=specialists,
        memory=memory,
        autonomy_mode=autonomy_mode,
    )
    input_count = _input_count(pipeline_state, operations, specialists, memory, review_snapshot)
    data_quality_score = min(0.25 + input_count * 0.12, 0.9)
    if decision == "needs_more_data":
        data_quality_score = min(data_quality_score, 0.55)

    return {
        "decision": decision,
        "verdict": verdict,
        "confidence": 0.82 if input_count else 0.55,
        "data_quality_score": round(data_quality_score, 3),
        "signal_strength": _decision_signal(decision),
        "autonomy_mode": autonomy_mode,
        "autonomy_recommendation": _autonomy_recommendation(decision, autonomy_mode),
        "pipeline_state": pipeline_state,
        "operations": operations,
        "specialists": specialists,
        "memory": memory,
        "reasons": reasons,
        "risks": risks,
        "next_actions": next_actions,
    }


def _pipeline_state(context: MarketContext, review_snapshot: dict[str, Any]) -> dict[str, Any]:
    data_freshness = _as_dict(context.metadata.get("data_freshness"))
    stale_sources = sorted(name for name, info in data_freshness.items() if isinstance(info, dict) and info.get("stale"))
    model_performance = _as_dict(context.metadata.get("model_performance"))
    regime_context = _as_dict(context.metadata.get("regime_context"))
    tuning = _as_dict(context.metadata.get("tuning"))
    return {
        "stale_sources": stale_sources,
        "model_verdict": model_performance.get("verdict"),
        "model_failures": list(model_performance.get("threshold_failures") or []),
        "performance_score": model_performance.get("performance_score"),
        "regime": regime_context.get("regime", "UNKNOWN"),
        "regime_tags": regime_context.get("context_tags") or context.metadata.get("regime_tags", []),
        "tuning_status": tuning.get("status"),
        "tuning_guardrails": tuning.get("guardrails", []),
    }


def _operations_state(context: MarketContext, review_snapshot: dict[str, Any]) -> dict[str, Any]:
    proposals = list(context.action_proposals)
    snapshot_ops = _as_dict(review_snapshot.get("operations"))
    proposal_count = len(proposals) if proposals else int(snapshot_ops.get("proposal_count") or 0)
    proposed_count = sum(1 for proposal in proposals if proposal.status == "proposed") if proposals else int(snapshot_ops.get("proposed_count") or 0)
    by_action = Counter(proposal.action_type for proposal in proposals)
    if not by_action and snapshot_ops.get("latest_proposals"):
        by_action = Counter(item.get("action_type") for item in snapshot_ops.get("latest_proposals", []))
    return {
        "proposal_count": proposal_count,
        "proposed_count": proposed_count,
        "action_types": dict(sorted((str(key), int(value)) for key, value in by_action.items() if key)),
        "latest_targets": _latest_targets(proposals, snapshot_ops),
        "has_tuning_proposal": any(proposal.action_type == "tune" for proposal in proposals)
        or "tune" in by_action,
        "has_report_proposal": any(proposal.action_type == "report" for proposal in proposals)
        or "report" in by_action,
    }


def _specialist_state(context: MarketContext, review_snapshot: dict[str, Any]) -> dict[str, Any]:
    report = _as_dict(review_snapshot.get("report"))
    notes = context.research_notes
    latest_thesis = notes[-1].thesis if notes else report.get("latest_thesis", "")
    patterns = sorted({pattern for note in notes for pattern in note.patterns}) or report.get("top_patterns", [])
    cited_note_count = sum(1 for note in notes if note.citations or note.evidence)
    note_count = len(notes) if notes else int(report.get("note_count") or 0)
    return {
        "document_count": len(context.research_documents) or int(report.get("document_count") or 0),
        "note_count": note_count,
        "cited_note_count": cited_note_count,
        "top_patterns": patterns[:10],
        "latest_thesis": latest_thesis,
    }


def _memory_state(context: MarketContext, review_snapshot: dict[str, Any]) -> dict[str, Any]:
    context_performance = _as_dict(context.metadata.get("context_performance")) or _as_dict(review_snapshot.get("context_performance"))
    memory = _as_dict(context.metadata.get("recommendation_memory")) or _as_dict(review_snapshot.get("memory"))
    review_actions = _as_dict(review_snapshot.get("review_actions"))
    open_data_requests = review_actions.get("open_data_requests", [])
    return {
        "memory_record_count": memory.get("record_count", memory.get("relevant_count", 0)),
        "hit_rate": memory.get("hit_rate"),
        "weak_context_count": len(context_performance.get("weak_contexts", [])),
        "strength_count": len(context_performance.get("strengths", [])),
        "open_data_request_count": len(open_data_requests) if isinstance(open_data_requests, list) else 0,
        "recent_lessons": memory.get("recent_lessons", []),
    }


def _classify_review(
    pipeline_state: dict[str, Any],
    operations: dict[str, Any],
    specialists: dict[str, Any],
    memory: dict[str, Any],
    autonomy_mode: str,
) -> tuple[str, str, list[str], list[str], list[str]]:
    reasons: list[str] = []
    risks = [
        "Paper autonomy is acceptable only with immutable logs, dry-run previews, and no live execution.",
        "Human review remains required before promotion, production config changes, or any real-money action.",
    ]
    next_actions: list[str] = []

    if pipeline_state["stale_sources"] or memory["open_data_request_count"]:
        decision = "needs_more_data"
        verdict = "needs_more_data"
        if pipeline_state["stale_sources"]:
            reasons.append(f"Stale data sources need review: {', '.join(pipeline_state['stale_sources'])}.")
            next_actions.append("Refresh or validate stale inputs before accepting new experiments or theses.")
        if memory["open_data_request_count"]:
            reasons.append("Open needs-more-data review actions are still unresolved.")
            next_actions.append("Resolve open data requests before promoting research or tuning proposals.")
        return decision, verdict, reasons, risks, next_actions

    if operations["has_tuning_proposal"] or pipeline_state["tuning_status"] == "tuning_experiment_proposed":
        decision = "experiment_proposal"
        verdict = "caution"
        reasons.append("A guarded tuning experiment is ready for human review.")
        next_actions.append("Dry-run the tuning proposal and verify walk-forward/holdout/risk guardrails.")
        return decision, verdict, reasons, risks, next_actions

    if operations["has_report_proposal"]:
        decision = "watchlist_review"
        verdict = "caution"
        reasons.append("A report/watchlist proposal is available for review.")
        next_actions.append("Review the proposal evidence before adding anything to watchlist.")
        return decision, verdict, reasons, risks, next_actions

    if operations["proposed_count"]:
        decision = "ready_for_review"
        verdict = "caution"
        reasons.append(f"{operations['proposed_count']} proposed operation action(s) need review.")
        next_actions.append("Run operation queue list/dry-run before approving any proposal.")
        return decision, verdict, reasons, risks, next_actions

    if pipeline_state["model_failures"]:
        decision = "experiment_proposal"
        verdict = "caution"
        reasons.append(f"Model performance failures need validation or tuning: {', '.join(pipeline_state['model_failures'])}.")
        next_actions.append("Run TuningAgent to produce a guarded experiment proposal.")
        return decision, verdict, reasons, risks, next_actions

    if specialists["note_count"] and specialists["cited_note_count"] == 0:
        decision = "needs_more_data"
        verdict = "needs_more_data"
        reasons.append("Specialist notes exist but lack citations/evidence in the current context.")
        next_actions.append("Add cited filings, transcripts, articles, or reports before promotion.")
        return decision, verdict, reasons, risks, next_actions

    if specialists["note_count"]:
        decision = "ready_for_review"
        verdict = "clear" if not memory["weak_context_count"] else "caution"
        reasons.append("Specialist research notes are available for review.")
        if memory["weak_context_count"]:
            risks.append("Weak context buckets exist; require stronger evidence in matching regimes.")
            next_actions.append("Compare thesis context tags against weak_contexts before accepting.")
        else:
            next_actions.append("Review latest specialist thesis and decide mark-reviewed or needs-more-data.")
        return decision, verdict, reasons, risks, next_actions

    decision = "needs_more_data"
    verdict = "needs_more_data"
    reasons.append("No sufficient pipeline, specialist, or proposal evidence is available for a chief review.")
    next_actions.append("Run Agent Lab or provide saved review/model/regime/tuning logs.")
    return decision, verdict, reasons, risks, next_actions


def _decision_signal(decision: str) -> float:
    return {
        "ready_for_review": 0.1,
        "watchlist_review": 0.05,
        "experiment_proposal": 0.0,
        "needs_more_data": -0.25,
        "reject": -0.5,
    }.get(decision, 0.0)


def _autonomy_recommendation(decision: str, autonomy_mode: str) -> str:
    if autonomy_mode != "paper_supervised":
        return "Use paper_supervised until review gates, logging, and outcome evaluation are proven stable."
    if decision in {"ready_for_review", "watchlist_review", "experiment_proposal"}:
        return "Paper-only autonomous simulation can continue, but approval is required for promotion or config changes."
    return "Pause autonomous paper actions until missing data or review blockers are resolved."


def _input_count(
    pipeline_state: dict[str, Any],
    operations: dict[str, Any],
    specialists: dict[str, Any],
    memory: dict[str, Any],
    review_snapshot: dict[str, Any],
) -> int:
    return sum(
        [
            bool(pipeline_state["model_verdict"] or pipeline_state["regime"] != "UNKNOWN" or pipeline_state["stale_sources"]),
            bool(operations["proposal_count"]),
            bool(specialists["note_count"] or specialists["document_count"]),
            bool(memory["memory_record_count"] or memory["weak_context_count"]),
            bool(review_snapshot),
        ]
    )


def _latest_targets(proposals: list[PipelineActionProposal], snapshot_ops: dict[str, Any]) -> list[str]:
    if proposals:
        return [proposal.target for proposal in proposals[-5:]]
    return [str(item.get("target")) for item in snapshot_ops.get("latest_proposals", [])[-5:] if item.get("target")]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
