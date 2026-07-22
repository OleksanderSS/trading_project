from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_lifecycle_profile import DomainAnalystLifecycleProfileCompiler
from dean_os.domain_macro_binding_quality_review import CONTRACT as QUALITY_CONTRACT
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready

CONTRACT = "dean_domain_macro_collection_request_v1"
DEFAULT_QUALITY_PATH = "reports/dean_os/domain_macro_binding_quality_review_current/latest.json"
DEFAULT_CANDIDATE_PATH = "reports/dean_os/domain_scoped_macro_envelope_current/latest.json"
DEFAULT_REGISTRY_PATH = "dean_os/config/macro_series_registry.yaml"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_macro_collection_request_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainMacroCollectionRequest:
    """Prepare one exact, point-in-time macro request; never execute it."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = "energy",
        quality_review_path: str | Path = DEFAULT_QUALITY_PATH,
        candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        request_as_of: str | None = None,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        quality_file = Path(quality_review_path)
        candidate_file = Path(candidate_path)
        registry_file = Path(registry_path)
        quality = _load_json(quality_file)
        candidate = _load_json(candidate_file)
        registry = _load_yaml(registry_file)
        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        cutoff = request_as_of or str((quality.get("inputs") or {}).get("review_as_of") or utc_now_iso())
        _aware(cutoff)

        policy = dict((profile.get("domain_overlay") or {}).get("macro_binding_quality_policy") or {})
        required = sorted({str(item) for item in policy.get("required_series") or []})
        supporting = sorted({str(item) for item in policy.get("supporting_series") or []})
        full_scope = sorted(set(required) | set(supporting))
        present = sorted(
            set(str(item) for item in (candidate.get("domain_binding") or {}).get("present_series_scope") or [])
            & set(full_scope)
        )
        missing_required = sorted(set(required) - set(present))
        missing_supporting = sorted(set(supporting) - set(present))
        gap_scope = sorted(set(missing_required) | set(missing_supporting))
        registry_series = dict(registry.get("series") or {})
        candidate_sha = _sha256_file(candidate_file)
        blockers = _blockers(
            domain_id=domain_id,
            quality=quality,
            candidate=candidate,
            candidate_sha=candidate_sha,
            profile=profile,
            required=required,
            supporting=supporting,
            registry=registry,
            cutoff=cutoff,
        )
        recommendation = str((quality.get("summary") or {}).get("recommendation") or "")
        request_required = not blockers and recommendation in {"replace_candidate", "defer"} and bool(gap_scope)
        if blockers:
            status = "macro_collection_request_blocked"
        elif request_required:
            status = "macro_collection_request_ready"
        else:
            status = "macro_collection_not_required"

        collection_scope = [
            {
                "series_id": series_id,
                "role": "required" if series_id in required else "supporting",
                "gap_status": "missing" if series_id in gap_scope else "refresh_for_coherent_snapshot",
                "context_key": (registry_series.get(series_id) or {}).get("context_key"),
                "name": (registry_series.get(series_id) or {}).get("name"),
                "unit": (registry_series.get(series_id) or {}).get("unit"),
                "source_family": registry.get("source_family"),
                "source_locator": f"https://fred.stlouisfed.org/series/{series_id}",
            }
            for series_id in full_scope
        ]
        request_id = f"macro_collect_{domain_id}_{candidate_sha[:16]}"
        payload = {
            "run_id": _run_id("domain_macro_collection_request"),
            "created_at": utc_now_iso(),
            "mode": "domain_macro_collection_request",
            "contract": CONTRACT,
            "request_id": request_id,
            "domain_id": domain_id,
            "inputs": {
                "quality_review_path": str(quality_review_path),
                "quality_review_sha256": _sha256_file(quality_file),
                "candidate_path": str(candidate_path),
                "candidate_sha256": candidate_sha,
                "registry_path": str(registry_path),
                "registry_sha256": _sha256_file(registry_file),
                "profile_domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "request_as_of": cutoff,
            },
            "summary": {
                "status": status,
                "quality_recommendation": recommendation,
                "request_required": request_required,
                "replacement_scope_count": len(full_scope),
                "gap_series_count": len(gap_scope),
                "missing_required_count": len(missing_required),
                "missing_supporting_count": len(missing_supporting),
                "structural_blockers": blockers,
                "execution_eligible_after_separate_authorization": request_required,
                "execution_authorized": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "snapshot_written": False,
                "binding_accepted": False,
                "can_invoke_domain_analysis": False,
                "can_trade": False,
            },
            "collection_request": {
                "source_family": "FRED",
                "collector": "FredCollector",
                "single_coherent_replacement_snapshot": True,
                "replacement_series_scope": full_scope,
                "gap_closure_series_scope": gap_scope,
                "missing_required_series": missing_required,
                "missing_supporting_series": missing_supporting,
                "refresh_existing_series": present,
                "series": collection_scope,
                "runtime_parameters": {
                    "series_ids": full_scope,
                    "as_of": cutoff,
                    "runtime_override_supported": True,
                    "fred_vintage_dates_parameter_required": True,
                    "maximum_collection_runs": 1,
                    "automatic_retry_allowed": False,
                },
            },
            "point_in_time_contract": {
                "required_row_fields": [
                    "series_id",
                    "date",
                    "realtime_start",
                    "value",
                    "source_locator",
                ],
                "identity_hash_fields": ["series_id", "date", "realtime_start", "value"],
                "availability_field": "realtime_start",
                "observation_date_is_not_availability": True,
                "file_mtime_is_not_availability": True,
                "missing_availability_action": "reject_snapshot",
                "canonical_pipeline_target": "data/processed/features/macro_data.parquet",
                "atomic_validated_write_required": True,
            },
            "execution_gate": {
                "status": "pending_separate_bounded_collection_authorization" if request_required else "not_open",
                "allowed_decisions": ["authorize_one_collection_run", "defer", "cancel"],
                "request_sha_binding_required": True,
                "candidate_sha_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "proposal_only": True,
                "maximum_collection_runs": 1,
                "automatic_retry_allowed": False,
                "collector_config_mutation_performed": False,
                "network_access_performed": False,
                "snapshot_write_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            source_path=quality_file,
            journal_path=Path(journal_path),
            apply=apply_journal,
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def _blockers(
    *,
    domain_id: str,
    quality: dict[str, Any],
    candidate: dict[str, Any],
    candidate_sha: str,
    profile: dict[str, Any],
    required: list[str],
    supporting: list[str],
    registry: dict[str, Any],
    cutoff: str,
) -> list[str]:
    blockers: list[str] = []
    summary = quality.get("summary") or {}
    assessment = quality.get("series_assessment") or {}
    if quality.get("contract") != QUALITY_CONTRACT or quality.get("mode") != "domain_macro_binding_quality_review":
        blockers.append("unsupported_quality_review_contract")
    if quality.get("domain_id") != domain_id or candidate.get("domain_id") != domain_id:
        blockers.append("domain_mismatch")
    if summary.get("status") != "quality_review_ready_recommendation_only":
        blockers.append("quality_review_not_ready")
    if summary.get("structural_blockers"):
        blockers.append("quality_review_has_structural_blockers")
    if summary.get("decision_recorded") is not False or summary.get("binding_accepted") is not False:
        blockers.append("quality_review_boundary_invalid")
    if (quality.get("inputs") or {}).get("candidate_sha256") != candidate_sha:
        blockers.append("candidate_sha_mismatch")
    if (quality.get("inputs") or {}).get("profile_domain_overlay_sha256") != profile.get("domain_overlay_sha256"):
        blockers.append("profile_sha_mismatch")
    if sorted(assessment.get("required_series") or []) != required:
        blockers.append("required_series_policy_mismatch")
    if sorted(assessment.get("supporting_series") or []) != supporting:
        blockers.append("supporting_series_policy_mismatch")
    if registry.get("registry_version") != "dean_macro_series_registry_v1" or registry.get("source_family") != "FRED":
        blockers.append("unsupported_macro_registry")
    catalog = dict(registry.get("series") or {})
    for series_id in sorted(set(required) | set(supporting)):
        item = catalog.get(series_id) or {}
        if not all(item.get(key) for key in ("context_key", "name", "unit")):
            blockers.append(f"registry_mapping_missing:{series_id}")
    candidate_as_of = str((candidate.get("inputs") or {}).get("as_of") or "")
    try:
        if not candidate_as_of or _aware(candidate_as_of) > _aware(cutoff):
            blockers.append("candidate_after_request_cutoff")
    except ValueError:
        blockers.append("candidate_as_of_invalid")
    return sorted(set(blockers))


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    if not payload["summary"]["request_required"]:
        return {
            "apply_requested": apply,
            "events_proposed": 0,
            "appended_count": 0,
            "existing_count": 0,
            "chain_valid": SystemJournal(journal_path).status()["chain_valid"],
        }
    event = {
        "event_type": "action_proposed",
        "effective_at": payload["inputs"]["request_as_of"],
        "actor": "domain_macro_collection_request",
        "domain_id": payload["domain_id"],
        "entity_type": "bounded_macro_collection_request",
        "entity_id": payload["request_id"],
        "source_artifact": artifact_binding(source_path),
        "context": {"context_family": "macro", "proposal_only": True},
        "payload": {
            "candidate_sha256": payload["inputs"]["candidate_sha256"],
            "replacement_series_scope": payload["collection_request"]["replacement_series_scope"],
            "gap_closure_series_scope": payload["collection_request"]["gap_closure_series_scope"],
            "maximum_collection_runs": 1,
            "execution_authorized": False,
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
    return {
        "apply_requested": True,
        **result,
        "record_count": status["record_count"],
        "chain_valid": status["chain_valid"],
        "tip_sha256": status["tip_sha256"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    request = payload["collection_request"]
    lines = [
        "# DEAN-OS Domain Macro Collection Request",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Source: `{request['source_family']}`",
        f"- Replacement snapshot series: {summary['replacement_scope_count']}",
        f"- Gap series: {summary['gap_series_count']}",
        f"- Execution authorized: {summary['execution_authorized']}",
        f"- Collector run performed: {summary['collector_run_performed']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        "",
        "## Exact gap",
        "",
        "- Required: " + (", ".join(request["missing_required_series"]) or "none"),
        "- Supporting: " + (", ".join(request["missing_supporting_series"]) or "none"),
        "- Refresh in coherent replacement: " + (", ".join(request["refresh_existing_series"]) or "none"),
        "",
        "## Point-in-time contract",
        "",
        "- Required fields: " + ", ".join(payload["point_in_time_contract"]["required_row_fields"]),
        "- Availability is `realtime_start`; observation date and file mtime are not availability.",
        "- Missing availability rejects the snapshot.",
        "",
        "## Boundary",
        "",
        "- This artifact prepares one request only; it performs no network call or collector run.",
        "- Automatic retry, binding acceptance, analyst invocation, learning writes and trading remain disabled.",
    ]
    if summary["structural_blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in summary["structural_blockers"])
    return "\n".join(lines).strip() + "\n"


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


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = ["CONTRACT", "DomainMacroCollectionRequest", "render_markdown"]
