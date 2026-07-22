from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.analysts._producers.sector_market import (
    SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT,
    load_verified_sector_market_context_fragment,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


CONTRACT = "dean_domain_scoped_sector_market_envelope_v1"
DEFAULT_SOURCE_PATH = (
    "reports/dean_os/saved_sector_market_evidence_producer_current/latest.json"
)
DEFAULT_DISPATCH_PATH = (
    "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
)
DEFAULT_OUTPUT_DIR = (
    "reports/dean_os/domain_scoped_sector_market_envelope_current"
)
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainScopedSectorMarketEnvelope:
    """Bind one verified saved sector-market artifact to one domain.

    This adapter reuses an explicit local producer artifact. It does not run a
    price pipeline, discover a substitute source, accept the binding, invoke an
    analyst, approve a hypothesis, or trade.
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
        overlay = profile.get("domain_overlay") or {}
        measurement = overlay.get("market_measurement") or {}
        expected_universe = sorted(
            {str(item).strip().upper() for item in measurement.get("primary_universe") or []}
        )
        expected_benchmark = str(
            measurement.get("benchmark_ticker") or ""
        ).strip().upper()

        blockers = _structural_blockers(
            source=source,
            dispatch=dispatch,
            domain_id=domain_id,
            as_of=as_of,
            expected_universe=expected_universe,
            expected_benchmark=expected_benchmark,
            profile=profile,
        )
        verified_fragment: dict[str, Any] = {}
        verification_error: str | None = None
        if not blockers:
            try:
                verified_fragment = load_verified_sector_market_context_fragment(
                    source_file,
                    expected_as_of=as_of,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                verification_error = f"{type(exc).__name__}: {exc}"
                blockers.append("sector_market_source_verification_failed")

        blockers = sorted(set(blockers))
        candidate_ready = not blockers
        status = (
            "domain_sector_market_candidate_ready"
            if candidate_ready
            else "domain_sector_market_envelope_blocked"
        )
        source_sha = _sha256_file(source_file)
        payload: dict[str, Any] = {
            "run_id": _run_id("domain_scoped_sector_market_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_sector_market_envelope",
            "contract": CONTRACT,
            "source_producer_contract": SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "as_of": as_of,
                "source_path": str(source_file),
                "source_sha256": source_sha,
                "source_run_id": source.get("run_id"),
                "dispatch_path": str(dispatch_file),
                "dispatch_sha256": _sha256_file(dispatch_file),
                "profile_domain_overlay_sha256": profile.get(
                    "domain_overlay_sha256"
                ),
            },
            "domain_binding": {
                "profile_contract": profile.get("contract"),
                "fixed_contract_sha256": profile.get("fixed_contract_sha256"),
                "domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "expected_primary_universe": expected_universe,
                "source_sector_tickers": sorted(
                    {
                        str(item).strip().upper()
                        for item in (source.get("inputs") or {}).get(
                            "sector_tickers", []
                        )
                    }
                ),
                "expected_benchmark": expected_benchmark,
                "source_benchmark": str(
                    (source.get("inputs") or {}).get("benchmark") or ""
                ).strip().upper(),
                "context_role": "market_confirmation_only",
                "may_create_sector_thesis": False,
                "may_create_ticker_forecast": False,
            },
            "source_verification": {
                "verified": candidate_ready,
                "verification_error": verification_error,
                "producer_status": source.get("status"),
                "producer_metric_count": len(source.get("metrics") or []),
                "lineage": source.get("lineage") or {},
            },
            "market_context_fragment": {
                **verified_fragment,
                "domain_id": domain_id,
                "metadata": {
                    **dict(verified_fragment.get("metadata") or {}),
                    "domain_sector_market_envelope_contract": CONTRACT,
                    "domain_id": domain_id,
                    "supporting_market_confirmation_only": True,
                },
            }
            if candidate_ready
            else {"as_of": as_of, "domain_id": domain_id, "sector_data": {}},
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "structural_blockers": blockers,
                "source_reuse_performed": candidate_ready,
                "source_lineage_verified": candidate_ready,
                "candidate_ready_for_binding_review": candidate_ready,
                "producer_run_performed": False,
                "pipeline_stage_run_performed": False,
                "binding_accepted": False,
                "can_update_profile_binding": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "binding_gate": {
                "status": (
                    "candidate_ready_pending_explicit_binding_decision"
                    if candidate_ready
                    else "not_open"
                ),
                "allowed_decisions": [
                    "accept_binding",
                    "replace_candidate",
                    "defer",
                ],
                "candidate_sha256_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "producer_run_performed": False,
                "pipeline_stage_run_performed": False,
                "network_access_performed": False,
                "binding_write_performed": False,
                "analyst_invocation_performed": False,
                "hypothesis_approval_performed": False,
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


def load_verified_domain_sector_market_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Recursively verify a domain sector-market envelope and source lineage."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain sector-market envelope contract")
    if payload.get("mode") != "domain_scoped_sector_market_envelope":
        raise ValueError("unsupported domain sector-market envelope mode")
    if payload.get("status") != "domain_sector_market_candidate_ready":
        raise ValueError("domain sector-market envelope is not ready")
    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of = _aware(str(inputs.get("as_of") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain sector-market envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain sector-market expected domain mismatch")
    if expected_as_of is not None and as_of != _aware(expected_as_of):
        raise ValueError("domain sector-market expected as_of mismatch")
    summary = payload.get("summary") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain sector-market envelope has structural blockers")
    for key in (
        "source_reuse_performed",
        "source_lineage_verified",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"domain sector-market required flag invalid: {key}")
    for key in (
        "producer_run_performed",
        "pipeline_stage_run_performed",
        "binding_accepted",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"domain sector-market forbidden flag invalid: {key}")
    if safety.get("review_only") is not True:
        raise ValueError("domain sector-market review-only boundary invalid")
    source_path = Path(str(inputs.get("source_path") or "")).resolve()
    dispatch_path = Path(str(inputs.get("dispatch_path") or "")).resolve()
    if not source_path.is_file() or _sha256_file(source_path) != inputs.get(
        "source_sha256"
    ):
        raise ValueError("domain sector-market source hash mismatch")
    if not dispatch_path.is_file() or _sha256_file(dispatch_path) != inputs.get(
        "dispatch_sha256"
    ):
        raise ValueError("domain sector-market dispatch hash mismatch")
    source = _load_json(source_path)
    dispatch = _load_json(dispatch_path)
    profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    measurement = (profile.get("domain_overlay") or {}).get("market_measurement") or {}
    expected_universe = sorted(
        {str(item).strip().upper() for item in measurement.get("primary_universe") or []}
    )
    expected_benchmark = str(measurement.get("benchmark_ticker") or "").strip().upper()
    blockers = _structural_blockers(
        source=source,
        dispatch=dispatch,
        domain_id=domain_id,
        as_of=as_of.isoformat(),
        expected_universe=expected_universe,
        expected_benchmark=expected_benchmark,
        profile=profile,
    )
    if blockers:
        raise ValueError(
            "domain sector-market recursive verification failed: "
            + ",".join(blockers)
        )
    verified = load_verified_sector_market_context_fragment(
        source_path,
        expected_as_of=as_of.isoformat(),
    )
    fragment = payload.get("market_context_fragment") or {}
    if (
        fragment.get("domain_id") != domain_id
        or _aware(str(fragment.get("as_of") or "")) != as_of
        or _sha256_json(fragment.get("sector_data") or {})
        != _sha256_json(verified.get("sector_data") or {})
    ):
        raise ValueError("domain sector-market fragment mismatch")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "sector_data": verified.get("sector_data") or {},
        "metadata": {
            **dict(fragment.get("metadata") or {}),
            "domain_sector_market_envelope_verified": True,
            "domain_sector_market_envelope_path": str(path),
            "domain_sector_market_envelope_sha256": _sha256_file(path),
        },
    }


def _structural_blockers(
    *,
    source: dict[str, Any],
    dispatch: dict[str, Any],
    domain_id: str,
    as_of: str,
    expected_universe: list[str],
    expected_benchmark: str,
    profile: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if source.get("producer_contract") != SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT:
        blockers.append("unsupported_sector_market_producer_contract")
    if source.get("status") != "sector_market_evidence_ready":
        blockers.append("sector_market_source_not_ready")
    source_inputs = source.get("inputs") or {}
    source_as_of = str(source_inputs.get("as_of") or "")
    try:
        if _aware(source_as_of) != _aware(as_of):
            blockers.append("sector_market_as_of_mismatch")
    except ValueError:
        blockers.append("sector_market_source_as_of_invalid")
    source_universe = sorted(
        {str(item).strip().upper() for item in source_inputs.get("sector_tickers") or []}
    )
    if not expected_universe:
        blockers.append("domain_sector_market_universe_missing")
    elif source_universe != expected_universe:
        blockers.append("sector_market_universe_mismatch")
    if not expected_benchmark:
        blockers.append("domain_sector_market_benchmark_missing")
    elif str(source_inputs.get("benchmark") or "").strip().upper() != expected_benchmark:
        blockers.append("sector_market_benchmark_mismatch")
    safety = source.get("safety") or {}
    if safety.get("review_only") is not True:
        blockers.append("sector_market_review_only_boundary_missing")
    for key in (
        "pipeline_run_performed",
        "training_run_performed",
        "tuning_run_performed",
        "learning_write_performed",
        "production_config_write_performed",
        "broker_access_performed",
        "live_execution_performed",
    ):
        if safety.get(key) is not False:
            blockers.append(f"sector_market_safety_invalid:{key}")
    if profile.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    blockers.extend(_dispatch_blockers(dispatch, domain_id))
    return sorted(set(blockers))


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
            if item.get("context_family") == "sector_market"
        ),
        None,
    )
    if not task:
        blockers.append("sector_market_dispatch_task_missing")
    elif task.get("recommended_action") not in {
        "domain_scoped_sector_market_evidence_producer",
        "domain_scoped_sector_market_envelope",
        "prepare_one_allowlisted_offline_adapter_run",
    }:
        blockers.append("sector_market_envelope_not_dispatched")
    return blockers


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["as_of"],
        "actor": "domain_scoped_sector_market_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": "bind_{}_sector_market:{}".format(
            payload["domain_id"], payload["inputs"]["source_sha256"][:16]
        ),
        "source_artifact": artifact_binding(source_path),
        "context": {"context_family": "sector_market", "review_only": True},
        "payload": {
            "status": payload["summary"]["status"],
            "candidate_ready_for_binding_review": payload["summary"][
                "candidate_ready_for_binding_review"
            ],
            "source_lineage_verified": payload["summary"][
                "source_lineage_verified"
            ],
            "binding_accepted": False,
            "analyst_invoked": False,
            "hypothesis_approved": False,
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
    return {"apply_requested": True, **result, **journal.status()}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    binding = payload["domain_binding"]
    lines = [
        "# DEAN-OS Domain-Scoped Sector Market Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Source lineage verified: {summary['source_lineage_verified']}",
        f"- Expected/source benchmark: `{binding['expected_benchmark']}` / `{binding['source_benchmark']}`",
        f"- Expected/source universe size: {len(binding['expected_primary_universe'])}/{len(binding['source_sector_tickers'])}",
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
            "- The exact supplied sector-market artifact and its price lineage were verified; no producer or pipeline stage ran.",
            "- Sector-market data is confirmation context only and cannot create a thesis or ticker forecast.",
            "- Binding acceptance, analyst invocation, hypothesis approval, learning writes and trading remain disabled.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _aware(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    "DomainScopedSectorMarketEnvelope",
    "load_verified_domain_sector_market_context_fragment",
    "render_markdown",
]
