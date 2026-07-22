from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.analysts._producers.news import (
    SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT,
    load_verified_semiconductor_news_context_fragment,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


CONTRACT = "dean_domain_scoped_news_envelope_v1"
DEFAULT_SOURCE_PATH = (
    "reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json"
)
DEFAULT_DISPATCH_PATH = (
    "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
)
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_scoped_news_envelope_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainScopedNewsEnvelope:
    """Bind verified saved news to one domain as trigger evidence only.

    The adapter reuses one explicit artifact and its exact saved source and
    source-registry lineage.  It does not collect news, call an LLM, infer a
    directional thesis, confirm a hypothesis, accept a binding, learn, or trade.
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
        cutoff = _aware(as_of)
        source_file = Path(source_path).resolve()
        dispatch_file = Path(dispatch_path).resolve()
        source = _load_json(source_file)
        dispatch = _load_json(dispatch_file)
        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        overlay = profile.get("domain_overlay") or {}
        news_policy = overlay.get("news_binding_policy") or {}

        expected_lanes = sorted(
            {
                str(item)
                for item in overlay.get("required_evidence_types") or []
                if item
            }
        )
        ready_lanes = sorted(
            {str(item) for item in (source.get("summary") or {}).get(
                "ready_required_lanes", []
            )}
        )
        missing_lanes = sorted(
            {str(item) for item in (source.get("summary") or {}).get(
                "missing_required_lanes", []
            )}
        )
        declared_lanes = sorted(set(ready_lanes) | set(missing_lanes))

        registry_file = _declared_registry_path(source, source_file)
        registry = _load_yaml(registry_file) if registry_file.is_file() else {}
        blockers = _structural_blockers(
            source=source,
            dispatch=dispatch,
            source_file=source_file,
            domain_id=domain_id,
            cutoff=cutoff,
            profile=profile,
            news_policy=news_policy,
            registry_file=registry_file,
            expected_lanes=expected_lanes,
            declared_lanes=declared_lanes,
        )

        verified_fragment: dict[str, Any] = {}
        verification_error: str | None = None
        if not blockers:
            try:
                verified_fragment = (
                    load_verified_semiconductor_news_context_fragment(
                        source_file,
                        expected_as_of=cutoff.isoformat(),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                verification_error = f"{type(exc).__name__}: {exc}"
                blockers.append("news_source_recursive_verification_failed")

        blockers = sorted(set(blockers))
        lineage_verified = not blockers
        quality_gaps = _quality_gaps(
            source=source,
            registry=registry,
            expected_lanes=expected_lanes,
            ready_lanes=ready_lanes,
        ) if lineage_verified else []
        candidate_ready = lineage_verified and bool(
            (source.get("summary") or {}).get("accepted_news_record_count")
        )
        status = (
            "domain_news_candidate_ready_with_gaps"
            if candidate_ready and quality_gaps
            else "domain_news_candidate_ready"
            if candidate_ready
            else "domain_news_envelope_blocked"
        )
        source_sha = _sha256_file(source_file)
        records = list(verified_fragment.get("news") or []) if candidate_ready else []

        payload: dict[str, Any] = {
            "run_id": _run_id("domain_scoped_news_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_news_envelope",
            "contract": CONTRACT,
            "source_producer_contract": source.get("producer_contract"),
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "as_of": cutoff.isoformat(),
                "source_path": str(source_file),
                "source_sha256": source_sha,
                "source_run_id": source.get("run_id"),
                "dispatch_path": str(dispatch_file),
                "dispatch_sha256": _sha256_file(dispatch_file),
                "registry_path": str(registry_file),
                "registry_sha256": (
                    _sha256_file(registry_file) if registry_file.is_file() else None
                ),
                "profile_domain_overlay_sha256": profile.get(
                    "domain_overlay_sha256"
                ),
            },
            "domain_binding": {
                "profile_contract": profile.get("contract"),
                "fixed_contract_sha256": profile.get("fixed_contract_sha256"),
                "domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "expected_evidence_lanes": expected_lanes,
                "source_declared_lanes": declared_lanes,
                "ready_lanes": ready_lanes,
                "missing_lanes": missing_lanes,
                "context_role": "trigger_evidence_only",
                "news_is_hypothesis_confirmation": False,
                "news_is_directional_evidence_by_itself": False,
                "official_policy_confirmation_is_separate": True,
                "may_create_sector_thesis": False,
                "may_create_ticker_forecast": False,
            },
            "source_verification": {
                "verified": lineage_verified,
                "verification_error": verification_error,
                "producer_status": source.get("status"),
                "source_domain_id": (source.get("inputs") or {}).get("domain_id"),
                "accepted_news_record_count": len(records),
                "saved_source_lineage": source.get("source_provenance") or {},
                "saved_registry_lineage": source.get("registry") or {},
            },
            "news_context_fragment": {
                **verified_fragment,
                "domain_id": domain_id,
                "metadata": {
                    **dict(verified_fragment.get("metadata") or {}),
                    "domain_news_envelope_contract": CONTRACT,
                    "domain_id": domain_id,
                    "trigger_evidence_only": True,
                    "directional_claim_allowed": False,
                    "hypothesis_confirmation_allowed": False,
                },
            }
            if candidate_ready
            else {"as_of": cutoff.isoformat(), "domain_id": domain_id, "news": []},
            "quality": {
                "quality_gaps": quality_gaps,
                "source_registry_review_status": registry.get("review_status"),
                "accepted_news_record_count": len(records),
                "ready_lane_count": len(ready_lanes),
                "expected_lane_count": len(expected_lanes),
            },
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "structural_blockers": blockers,
                "quality_gaps": quality_gaps,
                "source_reuse_performed": candidate_ready,
                "source_lineage_verified": lineage_verified,
                "domain_identity_verified": lineage_verified,
                "trigger_semantics_preserved": lineage_verified,
                "candidate_ready_for_binding_review": candidate_ready,
                "producer_run_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "llm_call_performed": False,
                "binding_accepted": False,
                "hypothesis_confirmed": False,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_train": False,
                "can_trade": False,
            },
            "binding_gate": {
                "status": (
                    "candidate_ready_pending_explicit_binding_decision"
                    if candidate_ready
                    else "not_open"
                ),
                "allowed_decisions": ["accept_binding", "replace_candidate", "defer"],
                "candidate_sha256_binding_required": True,
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "producer_run_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "llm_call_performed": False,
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


def load_verified_domain_news_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Recursively verify a domain news envelope and its legacy lineage."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain news envelope contract")
    if payload.get("mode") != "domain_scoped_news_envelope":
        raise ValueError("unsupported domain news envelope mode")
    if payload.get("status") not in {
        "domain_news_candidate_ready",
        "domain_news_candidate_ready_with_gaps",
    }:
        raise ValueError("domain news envelope is not ready")

    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of = _aware(str(inputs.get("as_of") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain news envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain news envelope expected domain mismatch")
    if expected_as_of is not None and as_of != _aware(expected_as_of):
        raise ValueError("domain news envelope expected as_of mismatch")

    summary = payload.get("summary") or {}
    binding = payload.get("domain_binding") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain news envelope has structural blockers")
    for key in (
        "source_reuse_performed",
        "source_lineage_verified",
        "domain_identity_verified",
        "trigger_semantics_preserved",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(f"domain news envelope required flag invalid: {key}")
    for key in (
        "producer_run_performed",
        "collector_run_performed",
        "network_access_performed",
        "llm_call_performed",
        "binding_accepted",
        "hypothesis_confirmed",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_train",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"domain news envelope forbidden flag invalid: {key}")
    if (
        binding.get("context_role") != "trigger_evidence_only"
        or binding.get("news_is_hypothesis_confirmation") is not False
        or binding.get("news_is_directional_evidence_by_itself") is not False
        or binding.get("official_policy_confirmation_is_separate") is not True
        or safety.get("review_only") is not True
    ):
        raise ValueError("domain news envelope semantic boundary invalid")

    source_path = Path(str(inputs.get("source_path") or "")).resolve()
    registry_path = Path(str(inputs.get("registry_path") or "")).resolve()
    if (
        not source_path.is_file()
        or _sha256_file(source_path) != inputs.get("source_sha256")
    ):
        raise ValueError("domain news envelope source hash mismatch")
    if (
        not registry_path.is_file()
        or _sha256_file(registry_path) != inputs.get("registry_sha256")
    ):
        raise ValueError("domain news envelope registry hash mismatch")

    verified = load_verified_semiconductor_news_context_fragment(
        source_path,
        expected_as_of=as_of.isoformat(),
    )
    fragment = payload.get("news_context_fragment") or {}
    if (
        fragment.get("domain_id") != domain_id
        or _aware(str(fragment.get("as_of") or "")) != as_of
        or _sha256_json(fragment.get("news") or [])
        != _sha256_json(verified.get("news") or [])
    ):
        raise ValueError("domain news envelope fragment mismatch")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "news": list(verified.get("news") or []),
        "metadata": {
            **dict(fragment.get("metadata") or {}),
            "domain_news_envelope_verified": True,
            "domain_news_envelope_path": str(path),
            "domain_news_envelope_sha256": _sha256_file(path),
        },
    }


def _structural_blockers(
    *,
    source: dict[str, Any],
    dispatch: dict[str, Any],
    source_file: Path,
    domain_id: str,
    cutoff: datetime,
    profile: dict[str, Any],
    news_policy: dict[str, Any],
    registry_file: Path,
    expected_lanes: list[str],
    declared_lanes: list[str],
) -> list[str]:
    blockers: list[str] = []
    if source.get("producer_contract") != SAVED_SEMICONDUCTOR_NEWS_EVIDENCE_CONTRACT:
        blockers.append("unsupported_news_producer_contract")
    if source.get("status") not in {
        "semiconductor_news_evidence_ready",
        "semiconductor_news_evidence_ready_with_gaps",
    }:
        blockers.append("news_source_not_ready")
    inputs = source.get("inputs") or {}
    if inputs.get("domain_id") != domain_id:
        blockers.append("news_source_domain_mismatch")
    try:
        if _aware(str(inputs.get("as_of") or "")) != cutoff:
            blockers.append("news_as_of_mismatch")
    except ValueError:
        blockers.append("news_source_as_of_invalid")
    if profile.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    if not expected_lanes:
        blockers.append("domain_required_news_lanes_missing")
    if not set(expected_lanes).issubset(set(declared_lanes)):
        blockers.append("news_lane_contract_mismatch")
    configured_registry_value = str(news_policy.get("source_registry_path") or "").strip()
    configured_registry = Path(configured_registry_value)
    if not configured_registry_value:
        blockers.append("domain_news_registry_not_configured")
    elif configured_registry.resolve() != registry_file.resolve():
        blockers.append("news_registry_domain_binding_mismatch")
    if not registry_file.is_file():
        blockers.append("news_registry_unreadable")
    safety = source.get("safety") or {}
    boundary = source.get("integration_boundary") or {}
    if safety.get("review_only") is not True:
        blockers.append("news_review_only_boundary_missing")
    for key in (
        "network_access_performed",
        "collector_run_performed",
        "pipeline_run_performed",
        "training_run_performed",
        "learning_write_performed",
        "production_config_write_performed",
        "broker_access_performed",
        "live_execution_performed",
    ):
        if safety.get(key) is not False:
            blockers.append(f"news_safety_invalid:{key}")
    for key in (
        "keyword_hit_is_lane_completion",
        "plain_text_ticker_promotion_allowed",
        "pipeline_feature_promotion_allowed",
        "training_allowed",
        "automatic_trading_allowed",
    ):
        if boundary.get(key) is not False:
            blockers.append(f"news_integration_boundary_invalid:{key}")
    if boundary.get("independent_strong_sources_required") is not True:
        blockers.append("news_independent_source_boundary_missing")
    blockers.extend(_dispatch_blockers(dispatch, domain_id))
    if not source_file.is_file():
        blockers.append("news_source_unreadable")
    return sorted(set(blockers))


def _dispatch_blockers(dispatch: dict[str, Any], domain_id: str) -> list[str]:
    blockers: list[str] = []
    if dispatch.get("mode") != "domain_binding_task_dispatch":
        blockers.append("unsupported_binding_dispatch")
    if (dispatch.get("summary") or {}).get("domain_id") != domain_id:
        blockers.append("binding_dispatch_domain_mismatch")
    task = next(
        (
            item for item in dispatch.get("task_dispatches") or []
            if item.get("context_family") == "news"
        ),
        None,
    )
    if not task:
        blockers.append("news_dispatch_task_missing")
    elif task.get("recommended_action") not in {
        "domain_scoped_news_evidence_producer",
        "domain_scoped_news_envelope",
        "prepare_one_allowlisted_offline_adapter_run",
    }:
        blockers.append("news_envelope_not_dispatched")
    return blockers


def _quality_gaps(
    *,
    source: dict[str, Any],
    registry: dict[str, Any],
    expected_lanes: list[str],
    ready_lanes: list[str],
) -> list[str]:
    gaps = [
        f"required_news_lane_missing:{lane}"
        for lane in expected_lanes
        if lane not in ready_lanes
    ]
    review_status = str(registry.get("review_status") or "")
    if review_status != "accepted":
        gaps.append("source_registry_pending_operator_confirmation")
    if "policy_or_geopolitical" in ready_lanes:
        gaps.append("official_policy_source_confirmation_still_required")
    if source.get("status") == "semiconductor_news_evidence_ready_with_gaps" and not gaps:
        gaps.append("legacy_news_producer_reported_gaps")
    return sorted(set(gaps))


def _declared_registry_path(source: dict[str, Any], source_file: Path) -> Path:
    value = (source.get("registry") or {}).get("path") or (
        source.get("inputs") or {}
    ).get("registry_path")
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    workspace = Path.cwd() / path
    if workspace.exists():
        return workspace.resolve()
    return (source_file.parent / path).resolve()


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["as_of"],
        "actor": "domain_scoped_news_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": "bind_{}_news:{}".format(
            payload["domain_id"], payload["inputs"]["source_sha256"][:16]
        ),
        "source_artifact": artifact_binding(source_path),
        "context": {
            "context_family": "news",
            "context_role": "trigger_evidence_only",
            "review_only": True,
        },
        "payload": {
            "status": payload["summary"]["status"],
            "candidate_ready_for_binding_review": payload["summary"][
                "candidate_ready_for_binding_review"
            ],
            "source_lineage_verified": payload["summary"][
                "source_lineage_verified"
            ],
            "hypothesis_confirmed": False,
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
    return {"apply_requested": True, **result, **journal.status()}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    binding = payload["domain_binding"]
    quality = payload["quality"]
    lines = [
        "# DEAN-OS Domain-Scoped News Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Source lineage verified: {summary['source_lineage_verified']}",
        f"- Accepted news records: {quality['accepted_news_record_count']}",
        f"- Ready/expected lanes: {quality['ready_lane_count']}/{quality['expected_lane_count']}",
        f"- Context role: `{binding['context_role']}`",
        f"- Hypothesis confirmed: {summary['hypothesis_confirmed']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Structural blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["structural_blockers"] or ["none"])
    lines.extend(["", "## Quality gaps", ""])
    lines.extend(f"- {item}" for item in summary["quality_gaps"] or ["none"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- News is trigger evidence: it can open an investigation but cannot confirm or falsify a hypothesis by itself.",
            "- Policy/geopolitical news still requires separate official-policy source confirmation.",
            "- No producer, collector, network, LLM, binding, analyst, learning, training, or trading action ran.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "DomainScopedNewsEnvelope",
    "load_verified_domain_news_context_fragment",
    "render_markdown",
]
