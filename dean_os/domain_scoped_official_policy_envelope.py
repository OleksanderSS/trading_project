from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.analysts._producers.policy import (
    CONTRACT as LEGACY_POLICY_CONTRACT,
    SNAPSHOT_CONTRACT,
    load_verified_official_policy_context_fragment,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.domain_scoped_news_envelope import (
    CONTRACT as DOMAIN_NEWS_CONTRACT,
    load_verified_domain_news_context_fragment,
)
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


CONTRACT = "dean_domain_scoped_official_policy_envelope_v1"
DEFAULT_SOURCE_PATH = (
    "reports/dean_os/saved_official_policy_evidence_producer_current/latest.json"
)
DEFAULT_NEWS_ENVELOPE_PATH = (
    "reports/dean_os/domain_scoped_news_envelope_current/latest.json"
)
DEFAULT_DISPATCH_PATH = (
    "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
)
DEFAULT_OUTPUT_DIR = (
    "reports/dean_os/domain_scoped_official_policy_envelope_current"
)
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainScopedOfficialPolicyEnvelope:
    """Bind verified official policy evidence to one domain for review.

    The adapter reuses an explicit legacy policy artifact and a verified domain
    news envelope. It recursively verifies the snapshot, raw PDF, registry,
    news lineage, domain, cutoff, source identity, and safety boundaries. It
    performs no collection, network, LLM, binding, learning, or trading action.
    """

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str,
        as_of: str,
        source_path: str | Path = DEFAULT_SOURCE_PATH,
        news_envelope_path: str | Path = DEFAULT_NEWS_ENVELOPE_PATH,
        dispatch_path: str | Path = DEFAULT_DISPATCH_PATH,
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = _aware(as_of)
        source_file = Path(source_path).resolve()
        news_file = Path(news_envelope_path).resolve()
        dispatch_file = Path(dispatch_path).resolve()
        source = _load_json(source_file)
        news_envelope = _load_json(news_file)
        dispatch = _load_json(dispatch_file)
        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        overlay = profile.get("domain_overlay") or {}
        policy = overlay.get("official_policy_binding_policy") or {}

        inputs = source.get("inputs") or {}
        snapshot_file = _resolve_path(inputs.get("snapshot_artifact_path"), source_file)
        registry_file = _resolve_path(inputs.get("registry_path"), source_file)
        legacy_news_file = _resolve_path(
            inputs.get("corroborating_news_artifact_path"), source_file
        )
        snapshot = _load_json(snapshot_file) if snapshot_file.is_file() else {}
        registry = _load_yaml(registry_file) if registry_file.is_file() else {}
        raw_file = _resolve_path(
            (snapshot.get("source") or {}).get("immutable_path"), snapshot_file
        )

        blockers = _structural_blockers(
            source=source,
            source_file=source_file,
            news_envelope=news_envelope,
            news_file=news_file,
            dispatch=dispatch,
            domain_id=domain_id,
            cutoff=cutoff,
            profile=profile,
            policy=policy,
            snapshot=snapshot,
            snapshot_file=snapshot_file,
            registry=registry,
            registry_file=registry_file,
            legacy_news_file=legacy_news_file,
            raw_file=raw_file,
        )

        policy_fragment: dict[str, Any] = {}
        news_fragment: dict[str, Any] = {}
        verification_errors: list[str] = []
        if not blockers:
            try:
                policy_fragment = load_verified_official_policy_context_fragment(
                    source_file,
                    expected_as_of=cutoff.isoformat(),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                verification_errors.append(f"policy:{type(exc).__name__}:{exc}")
                blockers.append("official_policy_recursive_verification_failed")
            try:
                news_fragment = load_verified_domain_news_context_fragment(
                    news_file,
                    expected_domain_id=domain_id,
                    expected_as_of=cutoff.isoformat(),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                verification_errors.append(f"news:{type(exc).__name__}:{exc}")
                blockers.append("official_policy_news_envelope_verification_failed")

        blockers = sorted(set(blockers))
        lineage_verified = not blockers
        quality_gaps = _quality_gaps(registry=registry, policy=policy) if lineage_verified else []
        records = list(policy_fragment.get("news") or []) if lineage_verified else []
        candidate_ready = lineage_verified and bool(records)
        status = (
            "domain_official_policy_candidate_ready_with_gaps"
            if candidate_ready and quality_gaps
            else "domain_official_policy_candidate_ready"
            if candidate_ready
            else "domain_official_policy_envelope_blocked"
        )

        source_provenance = source.get("source_provenance") or {}
        corroboration = source.get("corroboration") or {}
        payload: dict[str, Any] = {
            "run_id": _run_id("domain_scoped_official_policy_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_official_policy_envelope",
            "contract": CONTRACT,
            "source_producer_contract": source.get("producer_contract"),
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "as_of": cutoff.isoformat(),
                "source_path": str(source_file),
                "source_sha256": _sha256_file(source_file),
                "source_run_id": source.get("run_id"),
                "news_envelope_path": str(news_file),
                "news_envelope_sha256": _sha256_file(news_file),
                "dispatch_path": str(dispatch_file),
                "dispatch_sha256": _sha256_file(dispatch_file),
                "snapshot_path": str(snapshot_file),
                "snapshot_sha256": (
                    _sha256_file(snapshot_file) if snapshot_file.is_file() else None
                ),
                "registry_path": str(registry_file),
                "registry_sha256": (
                    _sha256_file(registry_file) if registry_file.is_file() else None
                ),
                "raw_policy_source_path": str(raw_file),
                "raw_policy_source_sha256": (
                    _sha256_file(raw_file) if raw_file.is_file() else None
                ),
                "profile_domain_overlay_sha256": profile.get(
                    "domain_overlay_sha256"
                ),
            },
            "domain_binding": {
                "profile_contract": profile.get("contract"),
                "fixed_contract_sha256": profile.get("fixed_contract_sha256"),
                "domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "context_role": "official_policy_fact_evidence",
                "policy_fact_may_be_established": True,
                "directional_market_claim_allowed_by_itself": False,
                "news_corroboration_substitutes_for_official_source": False,
                "may_confirm_hypothesis": False,
                "may_create_sector_thesis": False,
                "may_create_ticker_forecast": False,
            },
            "source_verification": {
                "verified": lineage_verified,
                "verification_errors": verification_errors,
                "producer_status": source.get("status"),
                "source_identity": source_provenance.get("source_identity"),
                "source_tier": source_provenance.get("source_tier"),
                "source_url": source_provenance.get("final_url"),
                "published_at": source_provenance.get("published_at"),
                "snapshot_contract": snapshot.get("snapshot_contract"),
                "raw_pdf_verified": lineage_verified,
                "registry_review_status": registry.get("review_status"),
            },
            "corroboration": {
                **dict(corroboration),
                "domain_news_envelope_contract": DOMAIN_NEWS_CONTRACT,
                "domain_news_envelope_verified": lineage_verified,
                "legacy_news_artifact_matches_envelope_source": lineage_verified,
                "news_is_corroboration_not_official_source": True,
            },
            "policy_context_fragment": {
                **policy_fragment,
                "domain_id": domain_id,
                "metadata": {
                    **dict(policy_fragment.get("metadata") or {}),
                    "domain_official_policy_envelope_contract": CONTRACT,
                    "domain_id": domain_id,
                    "official_policy_fact_evidence": True,
                    "directional_market_claim_allowed": False,
                    "hypothesis_confirmation_allowed": False,
                    "domain_news_envelope_sha256": (
                        news_fragment.get("metadata") or {}
                    ).get("domain_news_envelope_sha256"),
                },
            }
            if candidate_ready
            else {"as_of": cutoff.isoformat(), "domain_id": domain_id, "news": []},
            "quality": {
                "quality_gaps": quality_gaps,
                "accepted_policy_record_count": len(records),
                "official_source_count": 1 if candidate_ready else 0,
                "independent_news_source_count": len(
                    corroboration.get("existing_independent_strong_sources") or []
                ),
                "registry_review_status": registry.get("review_status"),
            },
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "structural_blockers": blockers,
                "quality_gaps": quality_gaps,
                "source_reuse_performed": candidate_ready,
                "source_lineage_verified": lineage_verified,
                "official_source_identity_verified": lineage_verified,
                "news_corroboration_lineage_verified": lineage_verified,
                "policy_fact_established": candidate_ready,
                "candidate_ready_for_binding_review": candidate_ready,
                "producer_run_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "llm_call_performed": False,
                "binding_accepted": False,
                "hypothesis_confirmed": False,
                "directional_market_claim_created": False,
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


def load_verified_domain_official_policy_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Recursively verify a domain policy envelope and all bound inputs."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain official-policy envelope contract")
    if payload.get("mode") != "domain_scoped_official_policy_envelope":
        raise ValueError("unsupported domain official-policy envelope mode")
    if payload.get("status") not in {
        "domain_official_policy_candidate_ready",
        "domain_official_policy_candidate_ready_with_gaps",
    }:
        raise ValueError("domain official-policy envelope is not ready")

    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of = _aware(str(inputs.get("as_of") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain official-policy envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain official-policy envelope expected domain mismatch")
    if expected_as_of is not None and as_of != _aware(expected_as_of):
        raise ValueError("domain official-policy envelope expected as_of mismatch")

    summary = payload.get("summary") or {}
    binding = payload.get("domain_binding") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain official-policy envelope has structural blockers")
    for key in (
        "source_reuse_performed",
        "source_lineage_verified",
        "official_source_identity_verified",
        "news_corroboration_lineage_verified",
        "policy_fact_established",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(
                f"domain official-policy envelope required flag invalid: {key}"
            )
    for key in (
        "producer_run_performed",
        "collector_run_performed",
        "network_access_performed",
        "llm_call_performed",
        "binding_accepted",
        "hypothesis_confirmed",
        "directional_market_claim_created",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_train",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(
                f"domain official-policy envelope forbidden flag invalid: {key}"
            )
    if (
        binding.get("context_role") != "official_policy_fact_evidence"
        or binding.get("policy_fact_may_be_established") is not True
        or binding.get("directional_market_claim_allowed_by_itself") is not False
        or binding.get("news_corroboration_substitutes_for_official_source")
        is not False
        or binding.get("may_confirm_hypothesis") is not False
        or binding.get("may_create_sector_thesis") is not False
        or binding.get("may_create_ticker_forecast") is not False
        or safety.get("review_only") is not True
    ):
        raise ValueError("domain official-policy envelope semantic boundary invalid")

    input_bindings = (
        ("source_path", "source_sha256", "source"),
        ("news_envelope_path", "news_envelope_sha256", "news envelope"),
        ("dispatch_path", "dispatch_sha256", "dispatch"),
        ("snapshot_path", "snapshot_sha256", "snapshot"),
        ("registry_path", "registry_sha256", "registry"),
        (
            "raw_policy_source_path",
            "raw_policy_source_sha256",
            "raw policy source",
        ),
    )
    bound_paths: dict[str, Path] = {}
    for path_key, sha_key, label in input_bindings:
        bound = Path(str(inputs.get(path_key) or "")).resolve()
        if not bound.is_file() or _sha256_file(bound) != inputs.get(sha_key):
            raise ValueError(f"domain official-policy envelope {label} hash mismatch")
        bound_paths[path_key] = bound

    policy_fragment = load_verified_official_policy_context_fragment(
        bound_paths["source_path"],
        expected_as_of=as_of.isoformat(),
    )
    news_fragment = load_verified_domain_news_context_fragment(
        bound_paths["news_envelope_path"],
        expected_domain_id=domain_id,
        expected_as_of=as_of.isoformat(),
    )
    fragment = payload.get("policy_context_fragment") or {}
    metadata = fragment.get("metadata") or {}
    if (
        fragment.get("domain_id") != domain_id
        or _aware(str(fragment.get("as_of") or "")) != as_of
        or _sha256_json(fragment.get("news") or [])
        != _sha256_json(policy_fragment.get("news") or [])
        or metadata.get("domain_official_policy_envelope_contract") != CONTRACT
        or metadata.get("official_policy_fact_evidence") is not True
        or metadata.get("directional_market_claim_allowed") is not False
        or metadata.get("hypothesis_confirmation_allowed") is not False
        or metadata.get("domain_news_envelope_sha256")
        != (news_fragment.get("metadata") or {}).get(
            "domain_news_envelope_sha256"
        )
    ):
        raise ValueError("domain official-policy envelope fragment mismatch")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "news": list(policy_fragment.get("news") or []),
        "metadata": {
            **dict(metadata),
            "domain_official_policy_envelope_verified": True,
            "domain_official_policy_envelope_path": str(path),
            "domain_official_policy_envelope_sha256": _sha256_file(path),
        },
    }


def _structural_blockers(
    *,
    source: dict[str, Any],
    source_file: Path,
    news_envelope: dict[str, Any],
    news_file: Path,
    dispatch: dict[str, Any],
    domain_id: str,
    cutoff: datetime,
    profile: dict[str, Any],
    policy: dict[str, Any],
    snapshot: dict[str, Any],
    snapshot_file: Path,
    registry: dict[str, Any],
    registry_file: Path,
    legacy_news_file: Path,
    raw_file: Path,
) -> list[str]:
    blockers: list[str] = []
    if source.get("producer_contract") != LEGACY_POLICY_CONTRACT:
        blockers.append("unsupported_official_policy_producer_contract")
    if source.get("status") != "official_policy_evidence_ready":
        blockers.append("official_policy_source_not_ready")
    inputs = source.get("inputs") or {}
    try:
        if _aware(str(inputs.get("as_of") or "")) != cutoff:
            blockers.append("official_policy_as_of_mismatch")
    except ValueError:
        blockers.append("official_policy_source_as_of_invalid")
    if profile.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    if not policy:
        blockers.append("domain_official_policy_binding_policy_missing")

    configured_registry_value = str(policy.get("source_registry_path") or "").strip()
    if not configured_registry_value:
        blockers.append("domain_official_policy_registry_not_configured")
    elif Path(configured_registry_value).resolve() != registry_file.resolve():
        blockers.append("official_policy_registry_domain_binding_mismatch")

    for file, expected, label in (
        (snapshot_file, inputs.get("snapshot_artifact_sha256"), "snapshot"),
        (registry_file, inputs.get("registry_sha256"), "registry"),
        (
            legacy_news_file,
            inputs.get("corroborating_news_artifact_sha256"),
            "legacy_news",
        ),
    ):
        if not file.is_file() or _sha256_file(file) != expected:
            blockers.append(f"official_policy_{label}_sha256_mismatch")

    if (
        snapshot.get("snapshot_contract") != SNAPSHOT_CONTRACT
        or snapshot.get("status") != "official_policy_snapshot_ready"
    ):
        blockers.append("official_policy_snapshot_contract_invalid")
    snapshot_source = snapshot.get("source") or {}
    source_provenance = source.get("source_provenance") or {}
    if not raw_file.is_file():
        blockers.append("official_policy_raw_pdf_unreadable")
    else:
        if not raw_file.read_bytes()[:4] == b"%PDF":
            blockers.append("official_policy_raw_source_not_pdf")
        if _sha256_file(raw_file) != snapshot_source.get("sha256"):
            blockers.append("official_policy_raw_pdf_sha256_mismatch")
    for key in ("source_identity", "source_tier", "final_url", "sha256"):
        if snapshot_source.get(key) != source_provenance.get(key):
            blockers.append(f"official_policy_snapshot_provenance_mismatch:{key}")

    host = (urlsplit(str(snapshot_source.get("final_url") or "")).hostname or "").lower()
    allowed_hosts = {str(value).lower() for value in policy.get("allowed_official_hosts") or []}
    allowed_identities = {
        str(value) for value in policy.get("allowed_source_identities") or []
    }
    if not allowed_hosts or host not in allowed_hosts:
        blockers.append("official_policy_host_not_allowed_for_domain")
    if (
        not allowed_identities
        or snapshot_source.get("source_identity") not in allowed_identities
    ):
        blockers.append("official_policy_source_identity_not_allowed_for_domain")

    document = (registry.get("documents") or {}).get(snapshot_source.get("sha256"))
    if not isinstance(document, dict):
        blockers.append("official_policy_source_not_registered")
    else:
        for registry_key, source_key in (
            ("source_url", "final_url"),
            ("source_identity", "source_identity"),
            ("source_tier", "source_tier"),
        ):
            if document.get(registry_key) != snapshot_source.get(source_key):
                blockers.append(f"official_policy_registry_mismatch:{registry_key}")
        if document.get("evidence_type") != "policy_or_geopolitical":
            blockers.append("official_policy_registry_evidence_type_invalid")
        try:
            published_at = _aware(str(document.get("published_at") or ""))
            if published_at > cutoff:
                blockers.append("official_policy_publication_after_as_of")
            max_age = int(policy.get("max_source_age_days") or 0)
            if max_age <= 0 or (cutoff - published_at).total_seconds() / 86400 > max_age:
                blockers.append("official_policy_source_stale")
        except (TypeError, ValueError):
            blockers.append("official_policy_publication_timestamp_invalid")

    if news_envelope.get("contract") != DOMAIN_NEWS_CONTRACT:
        blockers.append("official_policy_domain_news_contract_invalid")
    if news_envelope.get("domain_id") != domain_id:
        blockers.append("official_policy_domain_news_identity_mismatch")
    news_inputs = news_envelope.get("inputs") or {}
    try:
        if _aware(str(news_inputs.get("as_of") or "")) != cutoff:
            blockers.append("official_policy_domain_news_as_of_mismatch")
    except ValueError:
        blockers.append("official_policy_domain_news_as_of_invalid")
    if (
        Path(str(news_inputs.get("source_path") or "")).resolve()
        != legacy_news_file.resolve()
        or news_inputs.get("source_sha256")
        != inputs.get("corroborating_news_artifact_sha256")
    ):
        blockers.append("official_policy_news_lineage_cross_binding_failed")
    news_summary = news_envelope.get("summary") or {}
    if (
        news_summary.get("source_lineage_verified") is not True
        or news_summary.get("trigger_semantics_preserved") is not True
        or news_summary.get("binding_accepted") is not False
        or news_summary.get("can_trade") is not False
    ):
        blockers.append("official_policy_domain_news_boundary_invalid")

    corroboration = source.get("corroboration") or {}
    independent_news = set(corroboration.get("existing_independent_strong_sources") or [])
    official_identity = str(corroboration.get("official_source_identity") or "")
    minimum_news = int(policy.get("minimum_independent_news_sources") or 0)
    if minimum_news <= 0 or len(independent_news) < minimum_news:
        blockers.append("official_policy_independent_news_corroboration_insufficient")
    if official_identity in independent_news:
        blockers.append("official_policy_source_counted_as_news_corroboration")
    combined = set(corroboration.get("combined_independent_sources") or [])
    if combined != independent_news | {official_identity}:
        blockers.append("official_policy_combined_source_set_invalid")

    summary = source.get("summary") or {}
    boundary = source.get("integration_boundary") or {}
    safety = source.get("safety") or {}
    if (
        summary.get("policy_lane_ready") is not True
        or summary.get("can_enter_market_context_review") is not True
        or summary.get("can_trade") is not False
    ):
        blockers.append("official_policy_source_summary_boundary_invalid")
    if (
        boundary.get("review_only") is not True
        or boundary.get("official_source_hash_bound") is not True
        or boundary.get("independent_corroboration_required") is not True
    ):
        blockers.append("official_policy_integration_boundary_invalid")
    for key in (
        "plain_text_ticker_promotion_allowed",
        "automatic_prediction_influence",
        "automatic_trading_allowed",
    ):
        if boundary.get(key) is not False:
            blockers.append(f"official_policy_integration_boundary_invalid:{key}")
    if safety.get("review_only") is not True:
        blockers.append("official_policy_review_only_boundary_missing")
    for key in (
        "network_access_performed",
        "pipeline_run_performed",
        "training_run_performed",
        "learning_write_performed",
        "live_execution_performed",
    ):
        if safety.get(key) is not False:
            blockers.append(f"official_policy_safety_invalid:{key}")
    blockers.extend(_dispatch_blockers(dispatch, domain_id))
    if not source_file.is_file() or not news_file.is_file():
        blockers.append("official_policy_explicit_input_unreadable")
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
            if item.get("context_family") == "official_policy"
        ),
        None,
    )
    if not task:
        blockers.append("official_policy_dispatch_task_missing")
    elif task.get("recommended_action") not in {
        "domain_scoped_official_policy_evidence_producer",
        "domain_scoped_official_policy_envelope",
        "prepare_one_allowlisted_offline_adapter_run",
    }:
        blockers.append("official_policy_envelope_not_dispatched")
    return blockers


def _quality_gaps(*, registry: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    accepted = {str(value) for value in policy.get("accepted_registry_review_statuses") or []}
    if str(registry.get("review_status") or "") not in accepted:
        gaps.append("official_policy_registry_pending_operator_acceptance")
    return gaps


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["as_of"],
        "actor": "domain_scoped_official_policy_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": "bind_{}_official_policy:{}".format(
            payload["domain_id"], payload["inputs"]["source_sha256"][:16]
        ),
        "source_artifact": artifact_binding(source_path),
        "context": {
            "context_family": "official_policy",
            "context_role": "official_policy_fact_evidence",
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
            "policy_fact_established": payload["summary"]["policy_fact_established"],
            "directional_market_claim_created": False,
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
    verification = payload["source_verification"]
    quality = payload["quality"]
    lines = [
        "# DEAN-OS Domain-Scoped Official Policy Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Official source: `{verification['source_identity']}`",
        f"- Source lineage verified: {summary['source_lineage_verified']}",
        f"- Raw PDF verified: {verification['raw_pdf_verified']}",
        f"- Independent news sources: {quality['independent_news_source_count']}",
        f"- Policy fact established: {summary['policy_fact_established']}",
        f"- Directional market claim created: {summary['directional_market_claim_created']}",
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
            "- The official document may establish that a policy exists and what it says; it does not establish market direction by itself.",
            "- News corroboration is independently verified context and never substitutes for the official source.",
            "- No producer, collector, network, LLM, binding, analyst, learning, training, or trading action ran.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _resolve_path(value: Any, declaring_file: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    workspace = Path.cwd() / path
    if workspace.exists():
        return workspace.resolve()
    return (declaring_file.parent / path).resolve()


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
    encoded = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "DomainScopedOfficialPolicyEnvelope",
    "load_verified_domain_official_policy_context_fragment",
    "render_markdown",
]
