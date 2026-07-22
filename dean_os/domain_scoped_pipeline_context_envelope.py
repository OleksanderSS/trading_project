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
from dean_os.world_model.world_model_pipeline_context import WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT


CONTRACT = "dean_domain_scoped_pipeline_context_envelope_v1"
DEFAULT_SOURCE_PATH = "reports/dean_os/world_model_pipeline_context_current/latest.json"
DEFAULT_DISPATCH_PATH = "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_scoped_pipeline_context_envelope_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"
ALLOWED_TIMEFRAMES = {"15m", "60m", "1d"}


class DomainScopedPipelineContextEnvelope:
    """Bind one existing read-only pipeline context bundle to a domain.

    This adapter does not discover files, regenerate stages, run models or
    register replay tasks.  It verifies the exact supplied bundle and its
    declared artifact lineage, then emits a review-only binding candidate.
    """

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str,
        as_of: str,
        source_path: str | Path = DEFAULT_SOURCE_PATH,
        dispatch_path: str | Path = DEFAULT_DISPATCH_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        _aware(as_of)
        source_file = Path(source_path).resolve()
        dispatch_file = Path(dispatch_path).resolve()
        source = _load_json(source_file)
        dispatch = _load_json(dispatch_file)
        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        blockers, lineage = _blockers(
            source=source,
            source_file=source_file,
            dispatch=dispatch,
            domain_id=domain_id,
            as_of=as_of,
            profile=profile,
        )
        requested = source.get("requested") or {}
        tickers = sorted({str(item).upper() for item in requested.get("tickers") or []})
        timeframes = list(requested.get("timeframes") or [])
        source_status = str((source.get("summary") or {}).get("status") or "")
        if blockers:
            status = "domain_pipeline_context_envelope_blocked"
        elif source_status == "pipeline_context_bundle_ready":
            status = "domain_pipeline_context_candidate_ready"
        else:
            status = "domain_pipeline_context_candidate_ready_with_gaps"
        candidate_ready = not blockers
        payload = {
            "run_id": _run_id("domain_scoped_pipeline_context_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_pipeline_context_envelope",
            "contract": CONTRACT,
            "source_producer_contract": WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "as_of": as_of,
                "source_path": str(source_file),
                "source_sha256": _sha256_file(source_file),
                "source_run_id": source.get("run_id"),
                "source_created_at": source.get("created_at"),
                "dispatch_path": str(dispatch_file),
                "dispatch_sha256": _sha256_file(dispatch_file),
                "profile_domain_overlay_sha256": profile.get("domain_overlay_sha256"),
            },
            "domain_binding": {
                "profile_contract": profile.get("contract"),
                "fixed_contract_sha256": profile.get("fixed_contract_sha256"),
                "domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "requested_tickers": tickers,
                "requested_timeframes": timeframes,
                "context_role": "supporting_interpretation_only",
                "may_replace_verified_market_outcome": False,
            },
            "lineage_verification": lineage,
            "pipeline_context": source.get("pipeline_context") or {},
            "indicator_state_grid": source.get("indicator_state_grid") or {},
            "timeframe_lanes": source.get("timeframe_lanes") or [],
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "source_status": source_status,
                "structural_blockers": blockers,
                "available_lane_count": int((source.get("summary") or {}).get("available_lane_count") or 0),
                "exact_context_lane_count": int((source.get("summary") or {}).get("exact_context_lane_count") or 0),
                "missing_lane_count": int((source.get("summary") or {}).get("missing_lane_count") or 0),
                "lineage_reference_count": lineage["reference_count"],
                "lineage_verified_count": lineage["verified_count"],
                "source_reuse_performed": candidate_ready,
                "pipeline_stage_run_performed": False,
                "candidate_ready_for_binding_review": candidate_ready,
                "binding_accepted": False,
                "can_update_profile_binding": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "binding_gate": {
                "status": "candidate_ready_pending_explicit_binding_decision" if candidate_ready else "not_open",
                "allowed_decisions": ["accept_binding", "replace_candidate", "defer"],
                "candidate_sha256_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "pipeline_regeneration_performed": False,
                "stage4_run_performed": False,
                "stage5_run_performed": False,
                "replay_task_registration_performed": False,
                "network_access_performed": False,
                "binding_write_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["journal"] = _journal(
            payload=payload,
            source_path=source_file,
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


def load_verified_domain_pipeline_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Verify a saved domain pipeline envelope and every inventory reference."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain pipeline-context envelope contract")
    if payload.get("mode") != "domain_scoped_pipeline_context_envelope":
        raise ValueError("unsupported domain pipeline-context envelope mode")
    if payload.get("status") not in {
        "domain_pipeline_context_candidate_ready",
        "domain_pipeline_context_candidate_ready_with_gaps",
    }:
        raise ValueError("domain pipeline-context envelope is not ready")
    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of = _aware(str(inputs.get("as_of") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain pipeline-context envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain pipeline-context expected domain mismatch")
    if expected_as_of is not None and as_of != _aware(expected_as_of):
        raise ValueError("domain pipeline-context expected as_of mismatch")

    summary = payload.get("summary") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain pipeline-context envelope has structural blockers")
    for key in (
        "source_reuse_performed",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"domain pipeline-context required flag invalid: {key}")
    for key in (
        "pipeline_stage_run_performed",
        "binding_accepted",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"domain pipeline-context forbidden flag invalid: {key}")
    if safety.get("review_only") is not True:
        raise ValueError("domain pipeline-context review-only boundary invalid")

    source_path = Path(str(inputs.get("source_path") or "")).resolve()
    dispatch_path = Path(str(inputs.get("dispatch_path") or "")).resolve()
    if not source_path.is_file() or _sha256_file(source_path) != inputs.get(
        "source_sha256"
    ):
        raise ValueError("domain pipeline-context source hash mismatch")
    if not dispatch_path.is_file() or _sha256_file(dispatch_path) != inputs.get(
        "dispatch_sha256"
    ):
        raise ValueError("domain pipeline-context dispatch hash mismatch")
    source = _load_json(source_path)
    dispatch = _load_json(dispatch_path)
    profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    blockers, lineage = _blockers(
        source=source,
        source_file=source_path,
        dispatch=dispatch,
        domain_id=domain_id,
        as_of=as_of.isoformat(),
        profile=profile,
    )
    if blockers:
        raise ValueError(
            "domain pipeline-context recursive verification failed: "
            + ",".join(blockers)
        )
    if _sha256_json(lineage) != _sha256_json(payload.get("lineage_verification") or {}):
        raise ValueError("domain pipeline-context lineage summary mismatch")
    for key in ("pipeline_context", "indicator_state_grid", "timeframe_lanes"):
        if _sha256_json(payload.get(key) or ([] if key == "timeframe_lanes" else {})) != _sha256_json(
            source.get(key) or ([] if key == "timeframe_lanes" else {})
        ):
            raise ValueError(f"domain pipeline-context copied fragment mismatch: {key}")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "pipeline_context": payload.get("pipeline_context") or {},
        "indicator_state_grid": payload.get("indicator_state_grid") or {},
        "timeframe_lanes": list(payload.get("timeframe_lanes") or []),
        "metadata": {
            "domain_pipeline_context_envelope_verified": True,
            "domain_pipeline_context_envelope_path": str(path),
            "domain_pipeline_context_envelope_sha256": _sha256_file(path),
            "lineage_reference_count": lineage.get("reference_count"),
        },
    }


def _blockers(
    *,
    source: dict[str, Any],
    source_file: Path,
    dispatch: dict[str, Any],
    domain_id: str,
    as_of: str,
    profile: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    summary = source.get("summary") or {}
    safety = source.get("safety") or {}
    requested = source.get("requested") or {}
    tickers = sorted({str(item).upper() for item in requested.get("tickers") or []})
    timeframes = list(requested.get("timeframes") or [])
    allowed_tickers = {
        str(item).upper()
        for item in (profile.get("domain_overlay") or {}).get("ticker_universe_hint") or []
    }
    if source.get("contract") != WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT:
        blockers.append("unsupported_pipeline_context_contract")
    if source.get("mode") != "world_model_pipeline_context_discovery":
        blockers.append("unsupported_pipeline_context_mode")
    if summary.get("status") not in {
        "pipeline_context_bundle_ready",
        "pipeline_context_bundle_ready_with_gaps",
    }:
        blockers.append("pipeline_context_source_not_ready")
    if int(summary.get("available_lane_count") or 0) <= 0:
        blockers.append("pipeline_context_has_no_available_lanes")
    if not tickers:
        blockers.append("pipeline_context_ticker_scope_missing")
    for ticker in sorted(set(tickers) - allowed_tickers):
        blockers.append(f"pipeline_context_ticker_outside_domain:{ticker}")
    if not timeframes or any(item not in ALLOWED_TIMEFRAMES for item in timeframes):
        blockers.append("pipeline_context_timeframe_scope_invalid")
    source_created = str(source.get("created_at") or "")
    try:
        if not source_created or _aware(source_created) > _aware(as_of):
            blockers.append("pipeline_context_source_after_as_of")
    except ValueError:
        blockers.append("pipeline_context_source_time_invalid")
    required_false = [
        "pipeline_regeneration_performed",
        "stage4_run_performed",
        "stage5_run_performed",
        "replay_task_registration_performed",
        "learning_memory_write_performed",
        "production_config_write_performed",
        "model_promotion_performed",
        "can_trade",
    ]
    if safety.get("review_only") is not True:
        blockers.append("pipeline_context_review_only_boundary_missing")
    for key in required_false:
        if safety.get(key) is not False:
            blockers.append(f"pipeline_context_safety_invalid:{key}")
    if summary.get("can_register_replay_tasks") is not False:
        blockers.append("pipeline_context_replay_authority_invalid")
    if summary.get("can_write_learning_memory") is not False:
        blockers.append("pipeline_context_learning_authority_invalid")
    if summary.get("can_trade") is not False:
        blockers.append("pipeline_context_trading_authority_invalid")
    if profile.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    blockers.extend(_dispatch_blockers(dispatch, domain_id))
    lineage, lineage_blockers = _verify_inventory(source, source_file)
    blockers.extend(lineage_blockers)
    return sorted(set(blockers)), lineage


def _dispatch_blockers(dispatch: dict[str, Any], domain_id: str) -> list[str]:
    blockers: list[str] = []
    if dispatch.get("mode") != "domain_binding_task_dispatch":
        blockers.append("unsupported_binding_dispatch")
    if (dispatch.get("summary") or {}).get("domain_id") != domain_id:
        blockers.append("binding_dispatch_domain_mismatch")
    task = next(
        (
            item
            for item in dispatch.get("task_dispatches") or []
            if item.get("context_family") == "pipeline_context"
        ),
        None,
    )
    if not task:
        blockers.append("pipeline_context_dispatch_task_missing")
    elif task.get("recommended_action") not in {
        "domain_scoped_pipeline_context_envelope",
        "prepare_one_allowlisted_offline_adapter_run",
    }:
        blockers.append("pipeline_context_envelope_not_dispatched")
    return blockers


def _verify_inventory(
    source: dict[str, Any], source_file: Path
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    references: list[dict[str, Any]] = []
    inventory = source.get("artifact_inventory") or {}
    for family, items in inventory.items():
        for item in items or []:
            if item.get("available") is not True:
                continue
            raw_path = item.get("path")
            declared_sha = item.get("sha256")
            path = _resolve_reference(raw_path, source_file)
            current_sha = _sha256_file(path) if path.is_file() else None
            verified = bool(current_sha and current_sha == declared_sha)
            references.append(
                {
                    "artifact_family": family,
                    "path": str(path),
                    "declared_sha256": declared_sha,
                    "current_sha256": current_sha,
                    "verified": verified,
                }
            )
            if not path.is_file():
                blockers.append(f"pipeline_lineage_artifact_missing:{family}")
            elif not verified:
                blockers.append(f"pipeline_lineage_sha_mismatch:{family}")
    if not references:
        blockers.append("pipeline_lineage_inventory_empty")
    return {
        "source_bundle_sha256": _sha256_file(source_file),
        "reference_count": len(references),
        "verified_count": sum(1 for item in references if item["verified"]),
        "all_references_verified": bool(references) and all(item["verified"] for item in references),
        "references": references,
    }, blockers


def _resolve_reference(value: Any, source_file: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    workspace_path = Path.cwd() / path
    if workspace_path.exists():
        return workspace_path.resolve()
    return (source_file.parent / path).resolve()


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["as_of"],
        "actor": "domain_scoped_pipeline_context_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": "bind_{}_pipeline_context:{}".format(
            payload["domain_id"], payload["inputs"]["source_sha256"][:16]
        ),
        "source_artifact": artifact_binding(source_path),
        "context": {"context_family": "pipeline_context", "review_only": True},
        "payload": {
            "status": payload["summary"]["status"],
            "candidate_ready_for_binding_review": payload["summary"]["candidate_ready_for_binding_review"],
            "lineage_verified_count": payload["summary"]["lineage_verified_count"],
            "binding_accepted": False,
            "analyst_invoked": False,
            "learning_written": False,
            "trade_executed": False,
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
    binding = payload["domain_binding"]
    lines = [
        "# DEAN-OS Domain-Scoped Pipeline Context Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Source status: `{summary['source_status']}`",
        f"- Tickers: {', '.join(binding['requested_tickers']) or 'none'}",
        f"- Timeframes: {', '.join(binding['requested_timeframes']) or 'none'}",
        f"- Available lanes: {summary['available_lane_count']}",
        f"- Verified lineage references: {summary['lineage_verified_count']}/{summary['lineage_reference_count']}",
        f"- Candidate ready: {summary['candidate_ready_for_binding_review']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Can invoke analyst: {summary['can_invoke_domain_analysis']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["structural_blockers"] or ["none"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- The exact supplied pipeline-context artifact was reused; no filesystem discovery or pipeline stage ran.",
            "- Pipeline context is supporting interpretation only and cannot replace a verified market outcome.",
            "- Binding acceptance, analyst invocation, hypothesis approval, learning writes and trading remain disabled.",
        ]
    )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "DomainScopedPipelineContextEnvelope",
    "load_verified_domain_pipeline_context_fragment",
    "render_markdown",
]
