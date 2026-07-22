from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso

DEFAULT_EVIDENCE_PACK_JSON = "reports/dean_os/analyst_evidence_pack_cached_source_current/latest.json"
DEFAULT_SOURCE_GATE_JSON = "reports/dean_os/source_evidence_validation_gate_cached_source_current/latest.json"
DEFAULT_AGENT_LAB_PATH = "reports/dean_os/agent_lab_cached_source_current"
DEFAULT_DROPZONE_INVENTORY_JSON = "reports/dean_os/real_source_dropzone_inventory_current/latest.json"
DEFAULT_FUNDAMENTAL_GATE_JSON = "reports/dean_os/fundamental_input_readiness_gate/latest.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"
DEFAULT_DOMAIN_ANALYST_INTAKE_JSON = "reports/dean_os/domain_analyst_intake_packet_current/latest.json"
DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON = None
DEFAULT_DOMAIN_ANALYST_THESIS_REVIEW_JSON = None
DEFAULT_DOMAIN_ANALYST_TEMPLATE_STANDARDIZATION_JSON = None
DEFAULT_DOMAIN_ANALYST_CASE_REGISTRY_JSON = None
DEFAULT_PIPELINE_METRIC_INPUT_READINESS_JSON = None
DEFAULT_PIPELINE_CONTROL_INSTANCE_CONTRACT_JSON = None
DEFAULT_PIPELINE_CONTROL_CAUTION_REVIEW_JSON = None


class CurrentSystemAlignmentReview:
    """Review-only checkpoint for the current DEAN-OS architecture direction."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/current_system_alignment_review"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        evidence_pack_json: str | Path = DEFAULT_EVIDENCE_PACK_JSON,
        source_gate_json: str | Path = DEFAULT_SOURCE_GATE_JSON,
        agent_lab_path: str | Path = DEFAULT_AGENT_LAB_PATH,
        dropzone_inventory_json: str | Path = DEFAULT_DROPZONE_INVENTORY_JSON,
        fundamental_gate_json: str | Path | None = DEFAULT_FUNDAMENTAL_GATE_JSON,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        domain_analyst_intake_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INTAKE_JSON,
        domain_analyst_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
        domain_analyst_thesis_review_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_THESIS_REVIEW_JSON,
        domain_analyst_template_standardization_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_TEMPLATE_STANDARDIZATION_JSON,
        domain_analyst_case_registry_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_CASE_REGISTRY_JSON,
        pipeline_metric_input_readiness_json: str | Path | None = DEFAULT_PIPELINE_METRIC_INPUT_READINESS_JSON,
        pipeline_control_instance_contract_json: str | Path | None = DEFAULT_PIPELINE_CONTROL_INSTANCE_CONTRACT_JSON,
        pipeline_control_caution_review_json: str | Path | None = DEFAULT_PIPELINE_CONTROL_CAUTION_REVIEW_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        evidence_pack = _try_load_json(evidence_pack_json)
        source_gate = _try_load_json(source_gate_json)
        agent_lab = _load_agent_lab_report(agent_lab_path)
        dropzone_inventory = _try_load_json(dropzone_inventory_json)
        fundamental_gate = _try_load_json(fundamental_gate_json) if fundamental_gate_json else _missing_artifact(None)
        architecture_map = _try_load_json(architecture_map_json) if architecture_map_json else _missing_artifact(None)
        domain_analyst_intake = (
            _try_load_json(domain_analyst_intake_json) if domain_analyst_intake_json else _missing_artifact(None)
        )
        domain_analyst_instance_contract = (
            _try_load_json(domain_analyst_instance_contract_json) if domain_analyst_instance_contract_json else _missing_artifact(None)
        )
        domain_analyst_thesis_review = (
            _try_load_json(domain_analyst_thesis_review_json) if domain_analyst_thesis_review_json else _missing_artifact(None)
        )
        domain_analyst_template_standardization = (
            _try_load_json(domain_analyst_template_standardization_json)
            if domain_analyst_template_standardization_json
            else _missing_artifact(None)
        )
        domain_analyst_case_registry = (
            _try_load_json(domain_analyst_case_registry_json) if domain_analyst_case_registry_json else _missing_artifact(None)
        )
        pipeline_metric_input_readiness = (
            _try_load_json(pipeline_metric_input_readiness_json) if pipeline_metric_input_readiness_json else _missing_artifact(None)
        )
        pipeline_control_instance_contract = (
            _try_load_json(pipeline_control_instance_contract_json) if pipeline_control_instance_contract_json else _missing_artifact(None)
        )
        pipeline_control_caution_review = (
            _try_load_json(pipeline_control_caution_review_json) if pipeline_control_caution_review_json else _missing_artifact(None)
        )

        artifact_statuses = {
            "current_architecture_map": _architecture_map_status(architecture_map),
            "cached_evidence_pack": _cached_evidence_pack_status(evidence_pack),
            "source_evidence_gate": _source_gate_status(source_gate),
            "domain_analyst_intake": _domain_analyst_intake_status(domain_analyst_intake),
            "domain_analyst_instance_contract": _domain_analyst_instance_contract_status(domain_analyst_instance_contract),
            "domain_analyst_thesis_review": _domain_analyst_thesis_review_status(domain_analyst_thesis_review),
            "domain_analyst_template_standardization": _domain_analyst_template_standardization_status(domain_analyst_template_standardization),
            "domain_analyst_case_registry": _domain_analyst_case_registry_status(domain_analyst_case_registry),
            "pipeline_metric_input_readiness": _pipeline_metric_input_readiness_status(pipeline_metric_input_readiness),
            "pipeline_control_instance_contract": _pipeline_control_instance_contract_status(pipeline_control_instance_contract),
            "pipeline_control_caution_review": _pipeline_control_caution_review_status(pipeline_control_caution_review),
            "isolated_agent_lab": _agent_lab_status(agent_lab),
            "real_source_dropzone": _dropzone_status(dropzone_inventory),
            "fundamental_input_gate": _fundamental_gate_status(fundamental_gate),
            "legacy_system_map": _legacy_system_map_status(),
        }
        boundary_checks = _boundary_checks(
            artifact_statuses,
            source_gate,
            agent_lab,
            dropzone_inventory,
            fundamental_gate,
            architecture_map,
            domain_analyst_intake,
            domain_analyst_instance_contract,
            domain_analyst_thesis_review,
            domain_analyst_template_standardization,
            domain_analyst_case_registry,
            pipeline_metric_input_readiness,
            pipeline_control_instance_contract,
            pipeline_control_caution_review,
        )
        guidance = _decision_guidance(boundary_checks, artifact_statuses)
        payload = {
            "run_id": _run_id("current_system_alignment_review"),
            "created_at": utc_now_iso(),
            "mode": "current_system_alignment_review",
            "inputs": {
                "evidence_pack_json": str(evidence_pack_json),
                "source_gate_json": str(source_gate_json),
                "agent_lab_path": str(agent_lab_path),
                "dropzone_inventory_json": str(dropzone_inventory_json),
                "fundamental_gate_json": str(fundamental_gate_json) if fundamental_gate_json else None,
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
                "domain_analyst_intake_json": str(domain_analyst_intake_json) if domain_analyst_intake_json else None,
                "domain_analyst_instance_contract_json": str(domain_analyst_instance_contract_json) if domain_analyst_instance_contract_json else None,
                "domain_analyst_thesis_review_json": str(domain_analyst_thesis_review_json) if domain_analyst_thesis_review_json else None,
                "domain_analyst_template_standardization_json": (
                    str(domain_analyst_template_standardization_json)
                    if domain_analyst_template_standardization_json
                    else None
                ),
                "domain_analyst_case_registry_json": str(domain_analyst_case_registry_json)
                if domain_analyst_case_registry_json
                else None,
                "pipeline_metric_input_readiness_json": str(pipeline_metric_input_readiness_json) if pipeline_metric_input_readiness_json else None,
                "pipeline_control_instance_contract_json": str(pipeline_control_instance_contract_json) if pipeline_control_instance_contract_json else None,
                "pipeline_control_caution_review_json": str(pipeline_control_caution_review_json)
                if pipeline_control_caution_review_json
                else None,
            },
            "summary": {
                "alignment_status": guidance["status"],
                "recommended_action": guidance["recommended_action"],
                "next_operation_type": guidance["next_operation_type"],
                "useful_integrations_count": len([item for item in guidance["useful_integrations"] if item["status"] == "useful"]),
                "caution_count": guidance["warning_count"],
                "blocker_count": guidance["fail_count"],
                "can_scale_to_other_sectors_now": False,
                "can_run_live_collectors_now": False,
                "can_promote_learning_now": False,
                "can_generate_recommendations_now": False,
                "can_trade": False,
            },
            "artifact_statuses": artifact_statuses,
            "boundary_checks": boundary_checks,
            "decision_guidance": guidance,
            "usefulness_assessment": guidance["useful_integrations"],
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(),
            "recommendations": _recommendations(guidance),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_current_system_alignment_review_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return payload


def render_current_system_alignment_review_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    lines = [
        "# DEAN-OS Current System Alignment Review",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Alignment status: `{summary.get('alignment_status')}`",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        f"- Next operation type: `{summary.get('next_operation_type')}`",
        f"- Useful integrations: {summary.get('useful_integrations_count')}",
        f"- Cautions: {summary.get('caution_count')}",
        f"- Blockers: {summary.get('blocker_count')}",
        f"- Can scale to other sectors now: {summary.get('can_scale_to_other_sectors_now')}",
        f"- Can promote learning now: {summary.get('can_promote_learning_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Artifact Statuses",
        "",
    ]
    for name, status in payload.get("artifact_statuses", {}).items():
        lines.append(f"- `{name}`: {status.get('status')} - {status.get('summary')}")

    lines.extend(["", "## Boundary Checks", ""])
    for check in payload.get("boundary_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")

    lines.extend(["", "## Usefulness Assessment", ""])
    for item in payload.get("usefulness_assessment", []):
        lines.append(f"- `{item.get('component')}`: {item.get('status')} - {item.get('why')}")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))

    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))

    lines.extend(["", "## Decision Rationale", ""])
    lines.extend(f"- {reason}" for reason in guidance.get("reasons", []))
    return "\n".join(lines).strip() + "\n"


def _try_load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _missing_artifact(None)
    candidate = Path(path)
    if not candidate.exists():
        return _missing_artifact(path)
    try:
        return {
            "available": True,
            "path": str(candidate),
            "payload": json.loads(candidate.read_text(encoding="utf-8")),
        }
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "path": str(candidate),
            "error": f"invalid_json: {exc}",
        }


def _missing_artifact(path: str | Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": str(path) if path is not None else None,
        "error": "missing_artifact",
    }


def _load_agent_lab_report(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_file():
        return _try_load_json(candidate)
    if not candidate.exists():
        return _missing_artifact(candidate)
    json_files = sorted(
        [item for item in candidate.glob("*.json") if item.name != "latest.json"],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        return _missing_artifact(candidate / "*.json")
    return _try_load_json(json_files[0])


def _cached_evidence_pack_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("missing", "Cached evidence pack not found.", artifact)
    payload = artifact["payload"]
    coverage = payload.get("coverage", {})
    document_count = int(coverage.get("document_count") or 0)
    warning_count = int(coverage.get("warning_count") or 0)
    dropped_count = int(coverage.get("dropped_count") or 0)
    quality = coverage.get("data_quality")
    ready = bool(coverage.get("research_ready")) and bool(coverage.get("agent_lab_ready"))
    status = "useful" if ready and document_count > 0 and dropped_count == 0 else "needs_review"
    summary = (
        f"{document_count} cached documents, quality={quality}, warnings={warning_count}, "
        f"dropped={dropped_count}."
    )
    return _status(
        status,
        summary,
        artifact,
        document_count=document_count,
        source_types=coverage.get("by_source_type", {}),
        tickers=coverage.get("tickers", []),
        sectors=coverage.get("sectors", []),
        date_range=coverage.get("date_range", {}),
    )


def _source_gate_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("missing", "Source evidence validation gate report not found.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    guidance = payload.get("decision_guidance", {})
    gate_status = summary.get("gate_status")
    status = "useful" if gate_status == "source_evidence_ready_for_domain_research" else "needs_review"
    return _status(
        status,
        f"Gate={gate_status}, pass={guidance.get('pass_count')}, warn={guidance.get('warning_count')}, fail={guidance.get('fail_count')}.",
        artifact,
        gate_status=gate_status,
        can_enter_domain_research=summary.get("can_enter_domain_research"),
        can_promote_to_evidence=summary.get("can_promote_to_evidence"),
        can_extract_claims_events_entities=summary.get("can_extract_claims_events_entities"),
        can_write_learning_memory=summary.get("can_write_learning_memory"),
        can_create_recommendation=summary.get("can_create_recommendation"),
        can_trade=summary.get("can_trade"),
    )


def _agent_lab_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("missing", "Isolated Agent Lab report not found.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    document_count = int(payload.get("document_count") or summary.get("document_count") or 0)
    learning_count = int(summary.get("learning_record_count") or len(payload.get("learning_records", [])))
    proposal_count = int(summary.get("proposal_count") or len(payload.get("action_proposals", [])))
    status = "useful" if document_count > 0 and learning_count == 0 and proposal_count == 0 else "needs_review"
    return _status(
        status,
        f"Agent Lab processed {document_count} documents with learning_records={learning_count}, proposals={proposal_count}.",
        artifact,
        document_count=document_count,
        note_count=payload.get("note_count"),
        learning_record_count=learning_count,
        proposal_count=proposal_count,
        latest_thesis=summary.get("latest_thesis"),
    )


def _domain_analyst_intake_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Domain analyst intake packet is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("analyst_report_created") is True
        and summary.get("can_create_direct_ticker_thesis_without_bridge") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    status = "useful" if safe and summary.get("intake_status") in {"domain_analyst_intake_ready", "domain_analyst_intake_ready_with_warnings"} else "needs_review"
    return _status(
        status,
        (
            f"Intake={summary.get('intake_status')}, "
            f"evidence_items={summary.get('evidence_item_count')}, "
            f"ticker_direct={summary.get('ticker_direct_count')}."
        ),
        artifact,
        intake_status=summary.get("intake_status"),
        domain_id=summary.get("domain_id"),
        evidence_item_count=summary.get("evidence_item_count"),
        ticker_direct_count=summary.get("ticker_direct_count"),
        analyst_report_created=summary.get("analyst_report_created"),
        can_create_direct_ticker_thesis_without_bridge=summary.get("can_create_direct_ticker_thesis_without_bridge"),
        can_create_recommendation=summary.get("can_create_recommendation"),
        can_trade=summary.get("can_trade"),
    )


def _domain_analyst_instance_contract_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Domain analyst instance contract is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("manual_acceptance_required") is True
        and summary.get("can_scale_to_other_domains_now") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    status = "useful" if safe and summary.get("instance_status") in {"domain_analyst_instance_review_ready", "domain_analyst_instance_review_ready_with_cautions"} else "needs_review"
    return _status(
        status,
        (
            f"Instance={summary.get('instance_status')}, "
            f"domain={summary.get('domain_id')}, "
            f"reuse_after_manual_review={summary.get('can_reuse_as_template_after_manual_review')}."
        ),
        artifact,
        instance_status=summary.get("instance_status"),
        domain_id=summary.get("domain_id"),
        can_reuse_as_template_after_manual_review=summary.get("can_reuse_as_template_after_manual_review"),
        can_scale_to_other_domains_now=summary.get("can_scale_to_other_domains_now"),
        can_trade=summary.get("can_trade"),
    )


def _domain_analyst_thesis_review_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Domain analyst thesis review packet is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_create_direct_ticker_thesis_without_bridge") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_change_analyst_weights") is False
        and summary.get("can_write_config") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    packet_status = summary.get("packet_status")
    if not safe:
        status = "needs_review"
    elif packet_status in {"domain_thesis_review_ready", "domain_thesis_review_ready_with_cautions"}:
        status = "useful"
    elif packet_status == "domain_thesis_review_needs_more_evidence":
        status = "caution"
    else:
        status = "needs_review"
    return _status(
        status,
        (
            f"Thesis review={packet_status}, "
            f"domain={summary.get('domain_id')}, "
            f"standardize_after_review={summary.get('can_standardize_domain_template_after_manual_review')}."
        ),
        artifact,
        packet_status=packet_status,
        domain_id=summary.get("domain_id"),
        can_enter_manual_thesis_review=summary.get("can_enter_manual_thesis_review"),
        can_standardize_domain_template_after_manual_review=summary.get("can_standardize_domain_template_after_manual_review"),
        can_prepare_separate_ticker_bridge_after_manual_review=summary.get("can_prepare_separate_ticker_bridge_after_manual_review"),
        can_trade=summary.get("can_trade"),
    )


def _domain_analyst_template_standardization_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Domain analyst template standardization packet is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("manual_acceptance_required") is True
        and summary.get("can_mark_template_accepted_now") is False
        and summary.get("can_run_sector_to_ticker_bridge_now") is False
        and summary.get("can_scale_to_other_domains_now") is False
        and summary.get("can_create_direct_ticker_thesis_without_bridge") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_change_analyst_weights") is False
        and summary.get("can_write_config") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    candidate_status = summary.get("candidate_status")
    if not safe:
        status = "needs_review"
    elif candidate_status in {
        "ready_for_manual_template_acceptance",
        "ready_for_manual_template_acceptance_with_cautions",
    }:
        status = "useful"
    elif candidate_status == "needs_more_template_review":
        status = "caution"
    else:
        status = "needs_review"
    return _status(
        status,
        (
            f"Template candidate={candidate_status}, "
            f"domain={summary.get('domain_id')}, "
            f"accepted_now={summary.get('can_mark_template_accepted_now')}."
        ),
        artifact,
        candidate_status=candidate_status,
        domain_id=summary.get("domain_id"),
        can_standardize_domain_template_after_manual_acceptance=summary.get(
            "can_standardize_domain_template_after_manual_acceptance"
        ),
        can_prepare_sector_to_ticker_bridge_after_manual_acceptance=summary.get(
            "can_prepare_sector_to_ticker_bridge_after_manual_acceptance"
        ),
        can_mark_template_accepted_now=summary.get("can_mark_template_accepted_now"),
        can_run_sector_to_ticker_bridge_now=summary.get("can_run_sector_to_ticker_bridge_now"),
        can_scale_to_other_domains_now=summary.get("can_scale_to_other_domains_now"),
        can_trade=summary.get("can_trade"),
    )


def _domain_analyst_case_registry_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Domain analyst case registry packet is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_write_case_registry_artifact") is True
        and summary.get("can_promote_learning_now") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_change_analyst_weights") is False
        and summary.get("can_write_config") is False
        and summary.get("can_train_from_hits_only") is False
        and summary.get("can_drop_miss_cases") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    registry_status = summary.get("registry_status")
    if not safe:
        status = "needs_review"
    elif registry_status in {
        "case_registry_ready",
        "case_registry_ready_pending_outcomes",
        "case_registry_ready_with_outcome_buckets",
    }:
        status = "useful"
    elif registry_status == "blocked_case_registry":
        status = "needs_review"
    else:
        status = "caution"
    return _status(
        status,
        (
            f"Case registry={registry_status}, "
            f"cases={summary.get('case_count')}, "
            f"buckets={summary.get('outcome_bucket_counts')}."
        ),
        artifact,
        registry_status=registry_status,
        case_count=summary.get("case_count"),
        source_observation_count=summary.get("source_observation_count"),
        outcome_bucket_counts=summary.get("outcome_bucket_counts", {}),
        can_train_from_hits_only=summary.get("can_train_from_hits_only"),
        can_write_learning_memory=summary.get("can_write_learning_memory"),
        can_trade=summary.get("can_trade"),
    )


def _pipeline_control_instance_contract_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Pipeline control instance contract is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_run_autonomous_tuning_now") is False
        and summary.get("can_write_production_config") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    instance_status = summary.get("instance_status")
    if not safe:
        status = "needs_review"
    elif instance_status in {"pipeline_control_instance_review_ready", "pipeline_control_instance_review_ready_with_cautions"}:
        status = "useful"
    elif instance_status == "blocked_pipeline_control_instance":
        status = "caution"
    else:
        status = "needs_review"
    return _status(
        status,
        (
            f"Pipeline control={instance_status}, "
            f"blocked_planes={summary.get('blocked_metric_planes', [])}, "
            f"proposal_ready={summary.get('can_propose_reviewed_experiments_after_manual_review')}."
        ),
        artifact,
        instance_status=instance_status,
        blocked_metric_planes=summary.get("blocked_metric_planes", []),
        caution_metric_planes=summary.get("caution_metric_planes", []),
        can_propose_reviewed_experiments_after_manual_review=summary.get("can_propose_reviewed_experiments_after_manual_review"),
        can_write_production_config=summary.get("can_write_production_config"),
        can_trade=summary.get("can_trade"),
    )


def _pipeline_metric_input_readiness_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Pipeline metric input readiness gate is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_run_autonomous_tuning_now") is False
        and summary.get("can_write_production_config") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    readiness_status = summary.get("readiness_status")
    if not safe:
        status = "needs_review"
    elif readiness_status == "metric_inputs_ready":
        status = "useful"
    elif readiness_status in {"metric_inputs_ready_with_cautions", "blocked_metric_inputs"}:
        status = "caution"
    else:
        status = "needs_review"
    return _status(
        status,
        (
            f"Metric inputs={readiness_status}, "
            f"blocked_planes={summary.get('blocked_metric_planes', [])}, "
            f"surface_refresh={summary.get('can_refresh_pipeline_control_surface_now')}."
        ),
        artifact,
        readiness_status=readiness_status,
        blocked_metric_planes=summary.get("blocked_metric_planes", []),
        caution_metric_planes=summary.get("caution_metric_planes", []),
        can_refresh_pipeline_control_surface_now=summary.get("can_refresh_pipeline_control_surface_now"),
        can_propose_reviewed_tuning_after_surface_and_manual_review=summary.get(
            "can_propose_reviewed_tuning_after_surface_and_manual_review"
        ),
        can_write_production_config=summary.get("can_write_production_config"),
        can_trade=summary.get("can_trade"),
    )


def _pipeline_control_caution_review_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Pipeline control caution review packet is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    safe = (
        summary.get("can_run_autonomous_tuning_now") is False
        and summary.get("can_write_production_config") is False
        and summary.get("can_write_learning_memory") is False
        and summary.get("can_create_recommendation") is False
        and summary.get("can_trade") is False
    )
    review_status = summary.get("caution_review_status")
    if not safe:
        status = "needs_review"
    elif review_status in {"pipeline_cautions_need_reviewed_inputs", "pipeline_ready_for_manual_proposal_review"}:
        status = "useful"
    elif review_status == "pipeline_caution_review_blocked_by_hard_planes":
        status = "caution"
    else:
        status = "needs_review"
    return _status(
        status,
        (
            f"Pipeline caution review={review_status}, "
            f"missing_evidence={summary.get('missing_evidence_planes', [])}, "
            f"proposal_after_manual_acceptance={summary.get('can_propose_reviewed_experiments_after_manual_caution_acceptance')}."
        ),
        artifact,
        caution_review_status=review_status,
        blocked_metric_planes=summary.get("blocked_metric_planes", []),
        caution_metric_planes=summary.get("caution_metric_planes", []),
        missing_evidence_planes=summary.get("missing_evidence_planes", []),
        can_propose_reviewed_experiments_after_manual_caution_acceptance=summary.get(
            "can_propose_reviewed_experiments_after_manual_caution_acceptance"
        ),
        can_write_production_config=summary.get("can_write_production_config"),
        can_trade=summary.get("can_trade"),
    )


def _dropzone_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("missing", "Real source dropzone inventory not found.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    dropzone_status = summary.get("dropzone_status")
    supported = int(summary.get("supported_file_count") or 0)
    status = "caution" if supported == 0 else "useful"
    return _status(
        status,
        f"Dropzone={dropzone_status}, supported_files={supported}.",
        artifact,
        dropzone_status=dropzone_status,
        supported_file_count=supported,
        can_build_normalized_packet=summary.get("can_build_normalized_packet"),
        can_trade=summary.get("can_trade"),
    )


def _fundamental_gate_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Fundamental gate report not supplied; keep fundamentals separate for now.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    status = "useful" if summary.get("can_feed_value_screening_after_manual_review") else "caution"
    return _status(
        status,
        f"Fundamental readiness={summary.get('readiness_status')}, metrics={summary.get('metric_count')}.",
        artifact,
        readiness_status=summary.get("readiness_status"),
        metric_count=summary.get("metric_count"),
        can_feed_value_screening_after_manual_review=summary.get("can_feed_value_screening_after_manual_review"),
        can_compute_ratios_now=summary.get("can_compute_ratios_now"),
        can_trade=summary.get("can_trade"),
    )


def _architecture_map_status(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("available"):
        return _status("optional_missing", "Current architecture map is not available yet.", artifact)
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    ready = summary.get("architecture_status") == "current_architecture_map_ready"
    safe = (
        summary.get("can_clone_domain_profiles_now") is False
        and summary.get("can_write_production_config_now") is False
        and summary.get("can_trade") is False
    )
    status = "useful" if ready and safe else "needs_review"
    return _status(
        status,
        (
            f"Architecture={summary.get('architecture_status')}, "
            f"metric_planes={summary.get('pipeline_metric_plane_count')}, "
            f"domain_profiles={summary.get('domain_profile_count')}."
        ),
        artifact,
        architecture_status=summary.get("architecture_status"),
        active_design=summary.get("active_design"),
        can_clone_domain_profiles_now=summary.get("can_clone_domain_profiles_now"),
        can_write_production_config_now=summary.get("can_write_production_config_now"),
        can_trade=summary.get("can_trade"),
    )


def _legacy_system_map_status() -> dict[str, Any]:
    return {
        "status": "superseded",
        "path": "dean_os/system_audit_summary.py",
        "summary": "Legacy map describes older review chain names; use CurrentArchitectureMap as the active map.",
        "recommended_use": "historical_reference_only",
    }


def _status(status: str, summary: str, artifact: dict[str, Any], **details: Any) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "path": artifact.get("path"),
        "error": artifact.get("error"),
        **details,
    }


def _boundary_checks(
    artifact_statuses: dict[str, dict[str, Any]],
    source_gate: dict[str, Any],
    agent_lab: dict[str, Any],
    dropzone_inventory: dict[str, Any],
    fundamental_gate: dict[str, Any],
    architecture_map: dict[str, Any],
    domain_analyst_intake: dict[str, Any],
    domain_analyst_instance_contract: dict[str, Any],
    domain_analyst_thesis_review: dict[str, Any],
    domain_analyst_template_standardization: dict[str, Any],
    domain_analyst_case_registry: dict[str, Any],
    pipeline_metric_input_readiness: dict[str, Any],
    pipeline_control_instance_contract: dict[str, Any],
    pipeline_control_caution_review: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(_optional_artifact_check("current_architecture_map_optional", artifact_statuses["current_architecture_map"]))
    checks.append(_check_artifact_available("cached_evidence_pack_available", artifact_statuses["cached_evidence_pack"]))
    checks.append(_check_artifact_available("source_gate_available", artifact_statuses["source_evidence_gate"]))
    checks.append(_optional_artifact_check("domain_analyst_intake_optional", artifact_statuses["domain_analyst_intake"]))
    checks.append(_optional_artifact_check("domain_analyst_instance_contract_optional", artifact_statuses["domain_analyst_instance_contract"]))
    checks.append(_optional_artifact_check("domain_analyst_thesis_review_optional", artifact_statuses["domain_analyst_thesis_review"]))
    checks.append(_optional_artifact_check("domain_analyst_template_standardization_optional", artifact_statuses["domain_analyst_template_standardization"]))
    checks.append(_optional_artifact_check("domain_analyst_case_registry_optional", artifact_statuses["domain_analyst_case_registry"]))
    checks.append(_optional_artifact_check("pipeline_metric_input_readiness_optional", artifact_statuses["pipeline_metric_input_readiness"]))
    checks.append(_optional_artifact_check("pipeline_control_instance_contract_optional", artifact_statuses["pipeline_control_instance_contract"]))
    checks.append(_optional_artifact_check("pipeline_control_caution_review_optional", artifact_statuses["pipeline_control_caution_review"]))
    checks.append(_check_artifact_available("agent_lab_smoke_available", artifact_statuses["isolated_agent_lab"]))
    checks.append(_check_artifact_available("dropzone_inventory_available", artifact_statuses["real_source_dropzone"], warn_if_missing=True))
    checks.append(_optional_artifact_check("fundamental_gate_optional", artifact_statuses["fundamental_input_gate"]))

    checks.extend(_source_gate_boundary_checks(source_gate))
    checks.extend(_domain_analyst_intake_boundary_checks(domain_analyst_intake))
    checks.extend(_domain_analyst_instance_contract_boundary_checks(domain_analyst_instance_contract))
    checks.extend(_domain_analyst_thesis_review_boundary_checks(domain_analyst_thesis_review))
    checks.extend(_domain_analyst_template_standardization_boundary_checks(domain_analyst_template_standardization))
    checks.extend(_domain_analyst_case_registry_boundary_checks(domain_analyst_case_registry))
    checks.extend(_pipeline_metric_input_readiness_boundary_checks(pipeline_metric_input_readiness))
    checks.extend(_pipeline_control_instance_contract_boundary_checks(pipeline_control_instance_contract))
    checks.extend(_pipeline_control_caution_review_boundary_checks(pipeline_control_caution_review))
    checks.extend(_agent_lab_boundary_checks(agent_lab))
    checks.extend(_dropzone_boundary_checks(dropzone_inventory))
    checks.extend(_fundamental_gate_boundary_checks(fundamental_gate))
    checks.extend(_architecture_map_boundary_checks(architecture_map))
    legacy_check_status = "pass" if artifact_statuses["current_architecture_map"]["status"] == "useful" else "warn"
    checks.append(
        {
            "status": legacy_check_status,
            "code": "legacy_system_map_superseded" if legacy_check_status == "pass" else "legacy_system_map_stale",
            "message": artifact_statuses["legacy_system_map"]["summary"],
        }
    )
    checks.append(
        {
            "status": "pass",
            "code": "sector_scaling_deferred",
            "message": "Scaling to more sectors remains disabled until one source-first template is stable.",
        }
    )
    return checks


def _check_artifact_available(code: str, status: dict[str, Any], warn_if_missing: bool = False) -> dict[str, Any]:
    if status.get("status") in {"missing", "optional_missing"}:
        return {
            "status": "warn" if warn_if_missing else "fail",
            "code": code,
            "message": status.get("summary"),
        }
    return {
        "status": "pass",
        "code": code,
        "message": status.get("summary"),
    }


def _optional_artifact_check(code: str, status: dict[str, Any]) -> dict[str, Any]:
    if status.get("status") == "optional_missing":
        return {
            "status": "warn",
            "code": code,
            "message": status.get("summary"),
        }
    return {
        "status": "pass" if status.get("status") == "useful" else "warn",
        "code": code,
        "message": status.get("summary"),
    }


def _source_gate_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_promote_to_evidence", "source_gate_no_evidence_promotion"),
        _must_be_false(summary, "can_extract_claims_events_entities", "source_gate_no_extraction_execution"),
        _must_be_false(summary, "can_write_learning_memory", "source_gate_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "source_gate_no_recommendation"),
        _must_be_false(summary, "can_trade", "source_gate_no_trading"),
    ]
    if summary.get("can_enter_domain_research") is True:
        checks.append(
            {
                "status": "pass",
                "code": "source_gate_domain_research_allowed",
                "message": "Source gate allows manual domain research only.",
            }
        )
    else:
        checks.append(
            {
                "status": "warn",
                "code": "source_gate_domain_research_not_ready",
                "message": "Source gate does not yet allow domain research.",
            }
        )
    return checks


def _domain_analyst_intake_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(
            summary,
            "can_create_direct_ticker_thesis_without_bridge",
            "domain_analyst_intake_requires_ticker_bridge",
        ),
        _must_be_false(summary, "can_write_learning_memory", "domain_analyst_intake_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "domain_analyst_intake_no_recommendation"),
        _must_be_false(summary, "can_trade", "domain_analyst_intake_no_trading"),
    ]


def _domain_analyst_instance_contract_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_scale_to_other_domains_now", "domain_instance_no_auto_scaling"),
        _must_be_false(summary, "can_write_learning_memory", "domain_instance_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "domain_instance_no_recommendation"),
        _must_be_false(summary, "can_trade", "domain_instance_no_trading"),
    ]
    checks.append(
        {
            "status": "pass" if summary.get("manual_acceptance_required") is True else "fail",
            "code": "domain_instance_manual_acceptance_required",
            "message": f"manual_acceptance_required={summary.get('manual_acceptance_required')!r}.",
        }
    )
    return checks


def _domain_analyst_thesis_review_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_create_direct_ticker_thesis_without_bridge", "thesis_review_no_direct_ticker_thesis"),
        _must_be_false(summary, "can_write_learning_memory", "thesis_review_no_learning_write"),
        _must_be_false(summary, "can_change_analyst_weights", "thesis_review_no_weight_change"),
        _must_be_false(summary, "can_write_config", "thesis_review_no_config_write"),
        _must_be_false(summary, "can_create_recommendation", "thesis_review_no_recommendation"),
        _must_be_false(summary, "can_trade", "thesis_review_no_trading"),
    ]
    checks.append(
        {
            "status": "pass" if summary.get("manual_review_required") is True else "fail",
            "code": "thesis_review_manual_review_required",
            "message": f"manual_review_required={summary.get('manual_review_required')!r}.",
        }
    )
    return checks


def _domain_analyst_template_standardization_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_mark_template_accepted_now", "template_packet_no_auto_acceptance"),
        _must_be_false(summary, "can_run_sector_to_ticker_bridge_now", "template_packet_no_bridge_now"),
        _must_be_false(summary, "can_scale_to_other_domains_now", "template_packet_no_domain_scaling"),
        _must_be_false(summary, "can_create_direct_ticker_thesis_without_bridge", "template_packet_no_direct_ticker_thesis"),
        _must_be_false(summary, "can_write_learning_memory", "template_packet_no_learning_write"),
        _must_be_false(summary, "can_change_analyst_weights", "template_packet_no_weight_change"),
        _must_be_false(summary, "can_write_config", "template_packet_no_config_write"),
        _must_be_false(summary, "can_create_recommendation", "template_packet_no_recommendation"),
        _must_be_false(summary, "can_trade", "template_packet_no_trading"),
    ]
    checks.append(
        {
            "status": "pass" if summary.get("manual_acceptance_required") is True else "fail",
            "code": "template_packet_manual_acceptance_required",
            "message": f"manual_acceptance_required={summary.get('manual_acceptance_required')!r}.",
        }
    )
    return checks


def _domain_analyst_case_registry_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(summary, "can_promote_learning_now", "case_registry_no_learning_promotion"),
        _must_be_false(summary, "can_write_learning_memory", "case_registry_no_learning_write"),
        _must_be_false(summary, "can_change_analyst_weights", "case_registry_no_weight_change"),
        _must_be_false(summary, "can_write_config", "case_registry_no_config_write"),
        _must_be_false(summary, "can_train_from_hits_only", "case_registry_no_hits_only_training"),
        _must_be_false(summary, "can_drop_miss_cases", "case_registry_no_drop_misses"),
        _must_be_false(summary, "can_create_recommendation", "case_registry_no_recommendation"),
        _must_be_false(summary, "can_trade", "case_registry_no_trading"),
    ]


def _pipeline_control_instance_contract_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(summary, "can_run_autonomous_tuning_now", "pipeline_instance_no_autonomous_tuning"),
        _must_be_false(summary, "can_write_production_config", "pipeline_instance_no_config_write"),
        _must_be_false(summary, "can_write_learning_memory", "pipeline_instance_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "pipeline_instance_no_recommendation"),
        _must_be_false(summary, "can_trade", "pipeline_instance_no_trading"),
    ]


def _pipeline_metric_input_readiness_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(summary, "can_run_autonomous_tuning_now", "metric_input_gate_no_autonomous_tuning"),
        _must_be_false(summary, "can_write_production_config", "metric_input_gate_no_config_write"),
        _must_be_false(summary, "can_write_learning_memory", "metric_input_gate_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "metric_input_gate_no_recommendation"),
        _must_be_false(summary, "can_trade", "metric_input_gate_no_trading"),
    ]


def _pipeline_control_caution_review_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_run_autonomous_tuning_now", "pipeline_caution_review_no_autonomous_tuning"),
        _must_be_false(summary, "can_write_production_config", "pipeline_caution_review_no_config_write"),
        _must_be_false(summary, "can_write_learning_memory", "pipeline_caution_review_no_learning_write"),
        _must_be_false(summary, "can_create_recommendation", "pipeline_caution_review_no_recommendation"),
        _must_be_false(summary, "can_trade", "pipeline_caution_review_no_trading"),
    ]
    checks.append(
        {
            "status": "pass"
            if summary.get("caution_review_status")
            in {
                "pipeline_cautions_need_reviewed_inputs",
                "pipeline_ready_for_manual_proposal_review",
                "pipeline_caution_review_blocked_by_hard_planes",
            }
            else "warn",
            "code": "pipeline_caution_review_known_status",
            "message": f"caution_review_status={summary.get('caution_review_status')!r}.",
        }
    )
    return checks


def _agent_lab_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    payload = artifact["payload"]
    summary = payload.get("summary", {})
    learning_count = int(summary.get("learning_record_count") or len(payload.get("learning_records", [])))
    proposal_count = int(summary.get("proposal_count") or len(payload.get("action_proposals", [])))
    return [
        _count_must_be_zero(learning_count, "agent_lab_no_learning_records"),
        _count_must_be_zero(proposal_count, "agent_lab_no_operation_proposals"),
    ]


def _dropzone_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    checks = [
        _must_be_false(summary, "can_promote_to_evidence", "dropzone_inventory_no_evidence_promotion"),
        _must_be_false(summary, "can_trade", "dropzone_inventory_no_trading"),
    ]
    if int(summary.get("supported_file_count") or 0) == 0:
        checks.append(
            {
                "status": "warn",
                "code": "dropzone_empty",
                "message": "docs/research has no supported operator source file yet.",
            }
        )
    return checks


def _fundamental_gate_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(summary, "can_compute_ratios_now", "fundamental_gate_no_ratio_engine"),
        _must_be_false(summary, "can_create_recommendation", "fundamental_gate_no_recommendation"),
        _must_be_false(summary, "can_trade", "fundamental_gate_no_trading"),
    ]


def _architecture_map_boundary_checks(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if not artifact.get("available"):
        return []
    summary = artifact["payload"].get("summary", {})
    return [
        _must_be_false(summary, "can_clone_domain_profiles_now", "architecture_map_no_profile_cloning"),
        _must_be_false(summary, "can_write_production_config_now", "architecture_map_no_production_config_write"),
        _must_be_false(summary, "can_trade", "architecture_map_no_trading"),
    ]


def _must_be_false(summary: dict[str, Any], field: str, code: str) -> dict[str, Any]:
    if summary.get(field) is False:
        return {
            "status": "pass",
            "code": code,
            "message": f"{field}=False.",
        }
    return {
        "status": "fail",
        "code": code,
        "message": f"{field} must stay False, got {summary.get(field)!r}.",
    }


def _count_must_be_zero(value: int, code: str) -> dict[str, Any]:
    if value == 0:
        return {
            "status": "pass",
            "code": code,
            "message": f"{code} count is 0.",
        }
    return {
        "status": "fail",
        "code": code,
        "message": f"{code} count must be 0, got {value}.",
    }


def _decision_guidance(
    boundary_checks: list[dict[str, Any]],
    artifact_statuses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fail_count = sum(1 for check in boundary_checks if check.get("status") == "fail")
    warning_count = sum(1 for check in boundary_checks if check.get("status") == "warn")
    if fail_count:
        status = "misaligned_blocked"
        recommended_action = "fix_boundary_violation_before_more_integration"
        next_operation_type = "boundary_fix"
    elif warning_count:
        status = "aligned_with_cautions"
        recommended_action = "continue_cached_source_review_path"
        next_operation_type = "source_first_alignment_followup"
    else:
        status = "aligned"
        recommended_action = "standardize_current_template_before_scaling"
        next_operation_type = "template_standardization_review"

    useful_integrations = _usefulness_assessment(artifact_statuses)
    reasons = _guidance_reasons(status, boundary_checks, useful_integrations)
    return {
        "status": status,
        "recommended_action": recommended_action,
        "next_operation_type": next_operation_type,
        "pass_count": sum(1 for check in boundary_checks if check.get("status") == "pass"),
        "warning_count": warning_count,
        "fail_count": fail_count,
        "useful_integrations": useful_integrations,
        "reasons": reasons,
    }


def _usefulness_assessment(artifact_statuses: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "component": "current_architecture_map",
            "status": artifact_statuses["current_architecture_map"]["status"],
            "why": "Defines the active two-branch source-first architecture and supersedes the stale legacy map.",
        },
        {
            "component": "cached_news_macro_evidence_pack",
            "status": "useful" if artifact_statuses["cached_evidence_pack"]["status"] == "useful" else "needs_review",
            "why": "Turns raw cached news/macro tables into bounded analyst input without live collection.",
        },
        {
            "component": "source_evidence_validation_gate",
            "status": "useful" if artifact_statuses["source_evidence_gate"]["status"] == "useful" else "needs_review",
            "why": "Keeps source intake separate from extraction, learning, recommendation, and trading.",
        },
        {
            "component": "domain_analyst_intake_packet",
            "status": artifact_statuses["domain_analyst_intake"]["status"],
            "why": "Turns source documents into domain analyst evidence items with ticker/directness guardrails.",
        },
        {
            "component": "domain_analyst_instance_contract",
            "status": artifact_statuses["domain_analyst_instance_contract"]["status"],
            "why": "Defines whether one full domain analyst instance is reusable after manual review.",
        },
        {
            "component": "domain_analyst_thesis_review_packet",
            "status": artifact_statuses["domain_analyst_thesis_review"]["status"],
            "why": "Reviews the sector/domain thesis before any sector-to-ticker bridge or domain scaling.",
        },
        {
            "component": "domain_analyst_template_standardization_packet",
            "status": artifact_statuses["domain_analyst_template_standardization"]["status"],
            "why": "Packages one analyst instance and thesis review as a manual acceptance candidate without accepting or scaling it.",
        },
        {
            "component": "domain_analyst_case_registry_packet",
            "status": artifact_statuses["domain_analyst_case_registry"]["status"],
            "why": "Keeps hits, misses, inconclusive, pending, and invalid cases visible before any learning promotion.",
        },
        {
            "component": "pipeline_metric_input_readiness_gate",
            "status": artifact_statuses["pipeline_metric_input_readiness"]["status"],
            "why": "Inventories saved metric inputs before refreshing the pipeline-control surface or allowing tuning proposals.",
        },
        {
            "component": "pipeline_control_instance_contract",
            "status": artifact_statuses["pipeline_control_instance_contract"]["status"],
            "why": "Defines whether saved metric planes allow proposal-only pipeline experiments.",
        },
        {
            "component": "pipeline_control_caution_review_packet",
            "status": artifact_statuses["pipeline_control_caution_review"]["status"],
            "why": "Separates accepted caution states from missing drawdown, validation, and feature-stability evidence before tuning/orchestration.",
        },
        {
            "component": "isolated_agent_lab_smoke",
            "status": "useful" if artifact_statuses["isolated_agent_lab"]["status"] == "useful" else "needs_review",
            "why": "Shows analysts can consume source packets in review-only mode with no learning or proposals.",
        },
        {
            "component": "real_source_dropzone_inventory",
            "status": "caution" if artifact_statuses["real_source_dropzone"]["status"] == "caution" else artifact_statuses["real_source_dropzone"]["status"],
            "why": "Good operator boundary, but currently waits for one supported real source file.",
        },
        {
            "component": "fundamental_input_readiness_gate",
            "status": artifact_statuses["fundamental_input_gate"]["status"],
            "why": "Keeps fundamentals as a separate input-readiness axis before value screening.",
        },
        {
            "component": "legacy_system_audit_summary",
            "status": artifact_statuses["legacy_system_map"]["status"],
            "why": "Useful as history only; CurrentArchitectureMap is the active map when available.",
        },
    ]


def _guidance_reasons(
    status: str,
    boundary_checks: list[dict[str, Any]],
    useful_integrations: list[dict[str, str]],
) -> list[str]:
    reasons = [
        f"Alignment status is {status}.",
        "Current useful path is source-first: cached/local sources -> evidence pack -> source validation gate -> review-only Agent Lab.",
        "Sector research, ticker evidence, fundamentals, learning promotion, recommendations, and trading remain separate lanes.",
    ]
    warnings = [check["code"] for check in boundary_checks if check.get("status") == "warn"]
    failures = [check["code"] for check in boundary_checks if check.get("status") == "fail"]
    if warnings:
        reasons.append(f"Cautions: {', '.join(warnings)}.")
    if failures:
        reasons.append(f"Blockers: {', '.join(failures)}.")
    useful = [item["component"] for item in useful_integrations if item.get("status") == "useful"]
    if useful:
        reasons.append(f"Useful integrations: {', '.join(useful)}.")
    return reasons


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No claim/event/entity extraction execution is performed.",
        "No sector-to-ticker promotion is accepted without direct evidence.",
        "No learning memory write or analyst-weight change is performed.",
        "No recommendation, allocation, price target, or trade signal is generated.",
        "No broker, paper-trading, or live-trading state is touched.",
        "No scaling to new sectors is approved by this checkpoint.",
    ]


def _commands() -> list[dict[str, str]]:
    return [
        {
            "command_id": "refresh_current_alignment_review",
            "command": (
                "python run_agent_current_system_alignment_review.py "
                "--architecture-map-json reports/dean_os/current_architecture_map_current/latest.json "
                "--domain-analyst-intake-json reports/dean_os/domain_analyst_intake_packet_current/latest.json "
                "--domain-analyst-instance-contract-json reports/dean_os/domain_analyst_instance_contract_current/latest.json "
                "--domain-analyst-thesis-review-json reports/dean_os/domain_analyst_thesis_review_packet_current/latest.json "
                "--domain-analyst-template-standardization-json reports/dean_os/domain_analyst_template_standardization_packet_current/latest.json "
                "--domain-analyst-case-registry-json reports/dean_os/domain_analyst_case_registry_packet_current/latest.json "
                "--pipeline-metric-input-readiness-json reports/dean_os/pipeline_metric_input_readiness_gate_current/latest.json "
                "--pipeline-control-instance-contract-json reports/dean_os/pipeline_control_instance_contract_current/latest.json "
                "--pipeline-control-caution-review-json reports/dean_os/pipeline_control_caution_review_packet_current/latest.json "
                "--output-dir reports/dean_os/current_system_alignment_review"
            ),
        },
        {
            "command_id": "refresh_current_architecture_map",
            "command": "python run_agent_current_architecture_map.py --output-dir reports/dean_os/current_architecture_map_current",
        },
        {
            "command_id": "cached_source_smoke",
            "command": (
                "python run_agent_analyst_evidence_pack.py --news-data data\\colab\\backup_20260510_153551\\stage2_news_20260505_151233.parquet "
                "--macro-data data\\colab\\backup_20260510_153551\\stage2_macro_20260507_191104.parquet "
                "--tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor --tags ai_cycle cached_source_smoke "
                "--max-rows-per-table 200 --output-dir reports\\dean_os\\analyst_evidence_pack_cached_source_current"
            ),
        },
        {
            "command_id": "cached_source_gate",
            "command": (
                "python run_agent_source_evidence_validation_gate.py "
                "--source-json reports\\dean_os\\analyst_evidence_pack_cached_source_current\\latest.json "
                "--output-dir reports\\dean_os\\source_evidence_validation_gate_cached_source_current"
            ),
        },
        {
            "command_id": "domain_analyst_intake",
            "command": (
                "python run_agent_domain_analyst_intake_packet.py "
                "--evidence-pack-json reports\\dean_os\\analyst_evidence_pack_cached_source_current\\latest.json "
                "--source-gate-json reports\\dean_os\\source_evidence_validation_gate_cached_source_current\\latest.json "
                "--domain-id semiconductor_ai_infrastructure --tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor "
                "--output-dir reports\\dean_os\\domain_analyst_intake_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_instance_contract",
            "command": (
                "python run_agent_domain_analyst_instance_contract.py "
                "--evidence-pack-json reports\\dean_os\\analyst_evidence_pack_semiconductor_sector_only_strict_current\\latest.json "
                "--source-gate-json reports\\dean_os\\source_evidence_validation_gate_semiconductor_sector_only_strict_current\\latest.json "
                "--domain-intake-json reports\\dean_os\\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_instance_contract_current"
            ),
        },
        {
            "command_id": "domain_analyst_thesis_review_packet",
            "command": (
                "python run_agent_domain_analyst_thesis_review_packet.py "
                "--domain-intake-json reports\\dean_os\\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\\latest.json "
                "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_thesis_review_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_template_standardization_packet",
            "command": (
                "python run_agent_domain_analyst_template_standardization_packet.py "
                "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_template_standardization_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_case_registry_packet",
            "command": (
                "python run_agent_domain_analyst_case_registry_packet.py "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--domain-template-standardization-json reports\\dean_os\\domain_analyst_template_standardization_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_case_registry_packet_current"
            ),
        },
        {
            "command_id": "pipeline_metric_input_readiness_gate",
            "command": (
                "python run_agent_pipeline_metric_input_readiness_gate.py "
                "--model-performance performance_data.json "
                "--replay-batch reports\\dean_os\\historical_replay_batch_repaired_expanded\\latest.json "
                "--data-quality diagnostic_reports\\feature_lineage_report_current_cache.json "
                "--output-dir reports\\dean_os\\pipeline_metric_input_readiness_gate_current"
            ),
        },
        {
            "command_id": "pipeline_control_instance_contract",
            "command": (
                "python run_agent_pipeline_control_instance_contract.py "
                "--pipeline-surface-json reports\\dean_os\\pipeline_control_surface\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
                "--output-dir reports\\dean_os\\pipeline_control_instance_contract_current"
            ),
        },
        {
            "command_id": "pipeline_control_caution_review_packet",
            "command": (
                "python run_agent_pipeline_control_caution_review_packet.py "
                "--pipeline-metric-input-readiness-json reports\\dean_os\\pipeline_metric_input_readiness_gate_current\\latest.json "
                "--pipeline-control-instance-json reports\\dean_os\\pipeline_control_instance_contract_current\\latest.json "
                "--model-performance-report-json reports\\dean_os\\model_performance\\smoke.json "
                "--data-quality-json diagnostic_reports\\feature_lineage_report_current_cache.json "
                "--output-dir reports\\dean_os\\pipeline_control_caution_review_packet_current"
            ),
        },
        {
            "command_id": "isolated_agent_lab_cached_source",
            "command": (
                "python run_agent_lab.py --evidence-pack-json reports\\dean_os\\analyst_evidence_pack_cached_source_current\\latest.json "
                "--corpus reports\\dean_os\\agent_lab_cached_source_current\\corpus.sqlite "
                "--learning-store reports\\dean_os\\agent_lab_cached_source_current\\learning.sqlite "
                "--memory-store reports\\dean_os\\agent_lab_cached_source_current\\memory.sqlite "
                "--log-path reports\\dean_os\\agent_lab_cached_source_current\\events.jsonl "
                "--output-dir reports\\dean_os\\agent_lab_cached_source_current "
                "--tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor --tags ai_cycle cached_source_smoke "
                "--no-learning-records --no-operation-proposals"
            ),
        },
    ]


def _recommendations(guidance: dict[str, Any]) -> list[str]:
    if guidance["status"] == "misaligned_blocked":
        return [
            "Fix failed boundary checks before adding any new templates or sectors.",
            "Do not run Agent Lab, learning promotion, or replay calibration from misaligned artifacts.",
        ]
    return [
        "Keep using cached/local sources as analyst inputs; treat collectors as source suppliers, not the main system axis.",
        "Use CurrentArchitectureMap as the active system map; keep system_audit_summary.py as historical until refreshed or retired.",
        "Review DomainAnalystInstanceContract before accepting the first domain analyst template as reusable.",
        "Review DomainAnalystThesisReviewPacket before any sector-to-ticker bridge or domain scaling.",
        "Review DomainAnalystTemplateStandardizationPacket before recording manual template acceptance.",
        "Use DomainAnalystCaseRegistryPacket before learning promotion so misses, inconclusive, pending, and invalid cases are not filtered out.",
        "Run and review PipelineMetricInputReadinessGate before refreshing PipelineControlSurface.",
        "Review PipelineControlInstanceContract before allowing proposal-only tuning experiments.",
        "Run PipelineControlCautionReviewPacket when pipeline-control remains review-ready with cautions.",
        "Add one supported operator source file to docs/research when ready, then run the real-source normalized packet path.",
        "Do not scale to more sectors until the current semiconductor cached-source template and review packet are accepted.",
        "Keep fundamentals behind FundamentalInputReadinessGate before any value-screening interpretation.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
