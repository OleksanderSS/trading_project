from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.analysts.profiles import get_domain_profile
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_TEMPLATE_PATH = "dean_os/config/domain_analyst_lifecycle.template.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_analyst_lifecycle_profile_current"
ALLOWED_BINDING_STATES = {"configured", "not_configured"}


class DomainAnalystLifecycleProfileCompiler:
    """Materialize a domain-neutral analyst lifecycle with a domain overlay.

    Structural validity and operational data readiness are deliberately separate.
    A clone can therefore prove portability without pretending that its real
    evidence feeds have been configured or that it has authority to trade.
    """

    def __init__(self, template_path: str | Path = DEFAULT_TEMPLATE_PATH):
        self.template_path = Path(template_path)

    def compile(self, domain_id: str) -> dict[str, Any]:
        template = _load_json(self.template_path)
        profile = get_domain_profile(domain_id)
        overlay = dict(profile.analyst_lifecycle_profile or {})
        blockers = _structural_blockers(template, profile.model_dump(), overlay)
        bindings = dict(overlay.get("context_family_bindings") or {})
        missing_bindings = [
            family
            for family in template.get("fixed_context_families", [])
            if bindings.get(family) != "configured"
        ]
        fixed_contract = {
            key: value
            for key, value in template.items()
            if key not in {"fixed_context_families"}
        }
        fixed_contract["fixed_context_families"] = template.get("fixed_context_families", [])
        payload = {
            "contract": template.get("contract"),
            "domain_id": profile.domain_id,
            "display_name": profile.display_name,
            "profile_version": profile.version,
            "profile_status": overlay.get("profile_status", "not_configured"),
            "fixed_contract_sha256": _sha256(fixed_contract),
            "domain_overlay_sha256": _sha256(overlay),
            "fixed_contract": fixed_contract,
            "domain_overlay": {
                "required_evidence_types": list(profile.required_evidence_types),
                "useful_evidence_types": list(profile.useful_evidence_types),
                "core_questions": list(profile.core_questions),
                "contradiction_rules": list(profile.contradiction_rules),
                "ticker_universe_hint": list(profile.ticker_universe_hint),
                **overlay,
            },
            "readiness": {
                "schema_valid": not blockers,
                "structural_blockers": blockers,
                "missing_context_bindings": missing_bindings,
                "can_materialize_review_contract": not blockers,
                "can_run_domain_analysis_now": not blockers and not missing_bindings,
                "manual_template_acceptance_still_required": True,
                "can_activate_clone_now": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
        }
        return json_ready(payload)


class DomainAnalystLifecycleProfileReport:
    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        source_domain_id: str = "semiconductor_ai_infrastructure",
        clone_domain_id: str = "energy",
        template_path: str | Path = DEFAULT_TEMPLATE_PATH,
        save: bool = True,
    ) -> dict[str, Any]:
        compiler = DomainAnalystLifecycleProfileCompiler(template_path)
        source = compiler.compile(source_domain_id)
        clone = compiler.compile(clone_domain_id)
        same_core = source["fixed_contract_sha256"] == clone["fixed_contract_sha256"]
        clone_ready = clone["readiness"]["schema_valid"] and same_core
        payload = {
            "run_id": _run_id("domain_analyst_lifecycle_profile"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_lifecycle_profile_report",
            "summary": {
                "source_domain_id": source_domain_id,
                "clone_domain_id": clone_domain_id,
                "fixed_contract_identical": same_core,
                "source_schema_valid": source["readiness"]["schema_valid"],
                "clone_schema_valid": clone["readiness"]["schema_valid"],
                "can_materialize_clone_contract": clone_ready,
                "can_run_clone_domain_analysis_now": clone["readiness"]["can_run_domain_analysis_now"],
                "clone_missing_context_binding_count": len(clone["readiness"]["missing_context_bindings"]),
                "manual_template_acceptance_still_required": True,
                "can_activate_clone_now": False,
                "can_trade": False,
            },
            "source_profile": source,
            "clone_profile": clone,
            "next_step": (
                "Bind and validate real energy evidence/pipeline sources, then run one review-only vertical slice."
                if clone_ready
                else "Repair structural profile blockers before binding any data sources."
            ),
            "explicit_non_actions": [
                "No source collection was started.",
                "No hypothesis was approved or registered.",
                "No learning rule or production configuration was changed.",
                "No paper or live trading was performed.",
            ],
        }
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _structural_blockers(
    template: dict[str, Any], profile: dict[str, Any], overlay: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    families = list(template.get("fixed_context_families") or [])
    bindings = dict(overlay.get("context_family_bindings") or {})
    if sorted(bindings) != sorted(families):
        blockers.append("context_family_binding_set_mismatch")
    if any(state not in ALLOWED_BINDING_STATES for state in bindings.values()):
        blockers.append("invalid_context_binding_state")

    required = set(profile.get("required_evidence_types") or [])
    mechanisms = list(overlay.get("mechanisms") or [])
    if not mechanisms:
        blockers.append("mechanisms_missing")
    mechanism_ids = [str(item.get("mechanism_id") or "") for item in mechanisms]
    if any(not item for item in mechanism_ids) or len(set(mechanism_ids)) != len(mechanism_ids):
        blockers.append("mechanism_ids_missing_or_duplicate")
    referenced_lanes = {
        lane
        for mechanism in mechanisms
        for lane in (mechanism.get("evidence_lanes") or [])
    }
    if required - referenced_lanes:
        blockers.append("required_evidence_lane_not_mapped_to_mechanism")
    if any(not (item.get("observable_metrics") or []) for item in mechanisms):
        blockers.append("mechanism_observable_metrics_missing")

    horizons = dict(template.get("horizon_policy") or {})
    sector = list(horizons.get("sector_thesis_days") or [])
    event = list(horizons.get("event_response_days") or [])
    if sector != [30, 90, 180] or event != [1, 5, 20, 60, 120]:
        blockers.append("horizon_contract_mismatch")
    if set(sector) & set(event):
        blockers.append("sector_and_event_horizons_overlap")

    measurement = dict(overlay.get("market_measurement") or {})
    if not measurement.get("benchmark_ticker"):
        blockers.append("benchmark_ticker_missing")
    if not measurement.get("primary_universe"):
        blockers.append("primary_universe_missing")
    if not overlay.get("source_policy_ref"):
        blockers.append("source_policy_ref_missing")
    if overlay.get("report_inbox") != "chief_review":
        blockers.append("chief_review_inbox_binding_missing")

    authority = dict(template.get("authority_boundary") or {})
    if not authority or any(value is not False for value in authority.values()):
        blockers.append("authority_boundary_not_fail_closed")
    return sorted(set(blockers))


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    clone = payload["clone_profile"]
    lines = [
        "# DEAN-OS Reusable Domain Analyst Lifecycle Profile",
        "",
        f"- Source: `{summary['source_domain_id']}`",
        f"- Control clone: `{summary['clone_domain_id']}`",
        f"- Fixed lifecycle contract identical: {summary['fixed_contract_identical']}",
        f"- Clone schema valid: {summary['clone_schema_valid']}",
        f"- Can materialize clone contract: {summary['can_materialize_clone_contract']}",
        f"- Can run clone analysis now: {summary['can_run_clone_domain_analysis_now']}",
        f"- Can activate clone now: {summary['can_activate_clone_now']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Missing real bindings",
        "",
    ]
    missing = clone["readiness"]["missing_context_bindings"]
    lines.extend(f"- `{item}`" for item in missing)
    if not missing:
        lines.append("- none")
    lines.extend(["", "## Next step", "", f"- {payload['next_step']}"])
    lines.extend(["", "## Explicit non-actions", ""])
    lines.extend(f"- {item}" for item in payload["explicit_non_actions"])
    return "\n".join(lines).strip() + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "DomainAnalystLifecycleProfileCompiler",
    "DomainAnalystLifecycleProfileReport",
    "render_markdown",
]
