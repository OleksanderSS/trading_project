from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_lifecycle_profile import DomainAnalystLifecycleProfileCompiler
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready

CONTRACT = "dean_domain_macro_binding_quality_review_v1"
DEFAULT_CANDIDATE_PATH = "reports/dean_os/domain_scoped_macro_envelope_current/latest.json"
DEFAULT_BINDING_PLAN_PATH = "reports/dean_os/domain_analyst_binding_plan_current/latest.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_macro_binding_quality_review_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainMacroBindingQualityReview:
    """Score one SHA-bound macro candidate and recommend; never decide."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
        binding_plan_path: str | Path = DEFAULT_BINDING_PLAN_PATH,
        review_as_of: str | None = None,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        candidate_file = Path(candidate_path)
        plan_file = Path(binding_plan_path)
        candidate = _load_json(candidate_file)
        plan = _load_json(plan_file)
        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        policy = dict(
            (profile.get("domain_overlay") or {}).get("macro_binding_quality_policy")
            or {}
        )
        cutoff = review_as_of or utc_now_iso()
        _aware(cutoff)
        candidate_sha = _sha256_file(candidate_file)
        plan_macro = next(
            (item for item in plan.get("family_plans", []) if item.get("context_family") == "macro"),
            {},
        )
        structural_blockers = _structural_blockers(
            domain_id=domain_id,
            candidate=candidate,
            candidate_sha=candidate_sha,
            plan=plan,
            plan_macro=plan_macro,
            policy=policy,
            cutoff=cutoff,
        )
        present = set(
            str(item)
            for item in (candidate.get("domain_binding") or {}).get("present_series_scope") or []
        )
        required = set(str(item) for item in policy.get("required_series") or [])
        supporting = set(str(item) for item in policy.get("supporting_series") or [])
        required_present = sorted(required & present)
        required_missing = sorted(required - present)
        supporting_present = sorted(supporting & present)
        supporting_missing = sorted(supporting - present)
        required_coverage = len(required_present) / len(required) if required else 0.0
        supporting_coverage = len(supporting_present) / len(supporting) if supporting else 0.0
        total_scope = required | supporting
        total_coverage = len(total_scope & present) / len(total_scope) if total_scope else 0.0
        lineage_score = 0.0 if structural_blockers else 1.0
        weights = dict(policy.get("weights") or {})
        quality_score = round(
            required_coverage * float(weights.get("required_coverage") or 0.0)
            + supporting_coverage * float(weights.get("supporting_coverage") or 0.0)
            + lineage_score * float(weights.get("lineage_and_safety") or 0.0),
            6,
        )
        recommendation, reasons = _recommendation(
            blockers=structural_blockers,
            required_missing=required_missing,
            total_coverage=total_coverage,
            supporting_coverage=supporting_coverage,
            policy=policy,
        )
        payload = {
            "run_id": _run_id("domain_macro_binding_quality_review"),
            "created_at": utc_now_iso(),
            "mode": "domain_macro_binding_quality_review",
            "contract": CONTRACT,
            "domain_id": domain_id,
            "inputs": {
                "candidate_path": str(candidate_path),
                "candidate_sha256": candidate_sha,
                "binding_plan_path": str(binding_plan_path),
                "binding_plan_sha256": _sha256_file(plan_file),
                "review_as_of": cutoff,
                "profile_domain_overlay_sha256": profile.get("domain_overlay_sha256"),
            },
            "summary": {
                "status": "quality_review_blocked" if structural_blockers else "quality_review_ready_recommendation_only",
                "recommendation": recommendation,
                "quality_score": quality_score,
                "quality_band": _quality_band(quality_score),
                "required_coverage": round(required_coverage, 6),
                "supporting_coverage": round(supporting_coverage, 6),
                "total_coverage": round(total_coverage, 6),
                "structural_blockers": structural_blockers,
                "decision_recorded": False,
                "binding_accepted": False,
                "can_record_binding_decision": False,
                "can_update_profile_binding": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "series_assessment": {
                "required_series": sorted(required),
                "required_present": required_present,
                "required_missing": required_missing,
                "supporting_series": sorted(supporting),
                "supporting_present": supporting_present,
                "supporting_missing": supporting_missing,
            },
            "scoring_policy": policy,
            "recommendation": {
                "action": recommendation,
                "reasons": reasons,
                "binding_candidate_sha256": candidate_sha,
                "machine_recommendation_only": True,
                "automatic_decision_forbidden": True,
            },
            "manual_gate": {
                "status": "pending_explicit_binding_decision",
                "allowed_decisions": ["accept_binding", "replace_candidate", "defer"],
                "recommended_decision": recommendation,
                "candidate_sha256_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "candidate_mutation_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            candidate_path=candidate_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _structural_blockers(
    *,
    domain_id: str,
    candidate: dict[str, Any],
    candidate_sha: str,
    plan: dict[str, Any],
    plan_macro: dict[str, Any],
    policy: dict[str, Any],
    cutoff: str,
) -> list[str]:
    blockers: list[str] = []
    if candidate.get("domain_id") != domain_id:
        blockers.append("candidate_domain_mismatch")
    if candidate.get("summary", {}).get("candidate_ready_for_binding_review") is not True:
        blockers.append("candidate_not_ready_for_binding_review")
    if candidate.get("summary", {}).get("binding_accepted") is not False:
        blockers.append("candidate_binding_state_invalid")
    if candidate.get("safety", {}).get("review_only") is not True:
        blockers.append("candidate_review_only_boundary_missing")
    if plan.get("summary", {}).get("domain_id") != domain_id:
        blockers.append("binding_plan_domain_mismatch")
    proposed = plan_macro.get("proposed_candidate") or {}
    if proposed.get("sha256") != candidate_sha:
        blockers.append("candidate_sha_not_bound_to_current_plan")
    if plan_macro.get("status") != "reuse_candidate_ready_for_review":
        blockers.append("macro_candidate_not_routed_for_review")
    candidate_as_of = str((candidate.get("inputs") or {}).get("as_of") or "")
    try:
        if not candidate_as_of or _aware(candidate_as_of) > _aware(cutoff):
            blockers.append("candidate_after_review_cutoff")
    except ValueError:
        blockers.append("candidate_as_of_invalid")
    required = set(policy.get("required_series") or [])
    supporting = set(policy.get("supporting_series") or [])
    requested = set((candidate.get("domain_binding") or {}).get("requested_series_scope") or [])
    if not required or not supporting:
        blockers.append("macro_quality_policy_series_missing")
    if required & supporting:
        blockers.append("required_and_supporting_series_overlap")
    if required | supporting != requested:
        blockers.append("quality_policy_and_candidate_scope_mismatch")
    weights = policy.get("weights") or {}
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        blockers.append("macro_quality_weights_must_sum_to_one")
    return sorted(set(blockers))


def _recommendation(
    *,
    blockers: list[str],
    required_missing: list[str],
    total_coverage: float,
    supporting_coverage: float,
    policy: dict[str, Any],
) -> tuple[str, list[str]]:
    if blockers:
        return "defer", ["Structural/lineage blockers must be repaired before evaluating coverage.", *blockers]
    if required_missing:
        return str(policy.get("missing_required_action") or "replace_candidate"), [
            "Energy-critical macro series are missing: " + ", ".join(required_missing),
            "A rates-only macro candidate cannot represent energy demand and commodity conditions.",
        ]
    minimum_total = float(policy.get("minimum_total_coverage_for_accept") or 1.0)
    minimum_supporting = float(policy.get("minimum_supporting_coverage_for_accept") or 1.0)
    if total_coverage < minimum_total or supporting_coverage < minimum_supporting:
        return str(policy.get("insufficient_supporting_action") or "defer"), [
            f"Total coverage {total_coverage:.3f} is below {minimum_total:.3f} or supporting coverage {supporting_coverage:.3f} is below {minimum_supporting:.3f}.",
            "Required series are present, but broader context is not yet sufficient for acceptance.",
        ]
    return "accept_binding", [
        "All required energy macro series are present.",
        "Total and supporting coverage meet the predeclared acceptance thresholds.",
        "Acceptance still requires an explicit SHA-bound decision.",
    ]


def _journal(
    *, payload: dict[str, Any], candidate_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_proposed",
        "effective_at": payload["inputs"]["review_as_of"],
        "actor": "domain_macro_binding_quality_review",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_decision_recommendation",
        "entity_id": "bind_{}_macro:{}".format(
            payload["domain_id"], payload["inputs"]["candidate_sha256"][:16]
        ),
        "source_artifact": artifact_binding(candidate_path),
        "context": {"context_family": "macro", "review_only": True},
        "payload": {
            "recommendation": payload["summary"]["recommendation"],
            "quality_score": payload["summary"]["quality_score"],
            "required_coverage": payload["summary"]["required_coverage"],
            "supporting_coverage": payload["summary"]["supporting_coverage"],
            "total_coverage": payload["summary"]["total_coverage"],
            "decision_recorded": False,
            "binding_accepted": False,
        },
    }
    journal = SystemJournal(journal_path)
    if not apply:
        return {
            "apply_requested": False,
            "events_proposed": 1,
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": journal.status()["chain_valid"],
        }
    result = journal.append_many([event])
    status = journal.status()
    return {"apply_requested": True, **result, "record_count": status["record_count"], "chain_valid": status["chain_valid"], "tip_sha256": status["tip_sha256"]}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    series = payload["series_assessment"]
    lines = [
        "# DEAN-OS Energy Macro Binding Quality Review",
        "",
        f"- Status: `{summary['status']}`",
        f"- Machine recommendation: `{summary['recommendation']}`",
        f"- Quality score: {summary['quality_score']:.3f} (`{summary['quality_band']}`)",
        f"- Required coverage: {summary['required_coverage']:.3f}",
        f"- Supporting coverage: {summary['supporting_coverage']:.3f}",
        f"- Total coverage: {summary['total_coverage']:.3f}",
        f"- Decision recorded: {summary['decision_recorded']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Can invoke analyst: {summary['can_invoke_domain_analysis']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Series assessment",
        "",
        "- Required present: " + (", ".join(series["required_present"]) or "none"),
        "- Required missing: " + (", ".join(series["required_missing"]) or "none"),
        "- Supporting present: " + (", ".join(series["supporting_present"]) or "none"),
        "- Supporting missing: " + (", ".join(series["supporting_missing"]) or "none"),
        "",
        "## Recommendation reasons",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["recommendation"]["reasons"])
    lines.extend(["", "## Boundary", "", "- This packet recommends only; it cannot record or apply a binding decision.", "- Candidate SHA must remain unchanged through the decision gate."])
    return "\n".join(lines).strip() + "\n"


def _quality_band(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.65:
        return "adequate"
    if score >= 0.40:
        return "weak"
    return "insufficient"


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = ["CONTRACT", "DomainMacroBindingQualityReview"]
