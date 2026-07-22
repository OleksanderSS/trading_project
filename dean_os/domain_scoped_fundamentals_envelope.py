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
from dean_os.analysts._producers.sec.merger import (
    SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT,
    load_verified_merged_fundamental_context_fragment,
)
from dean_os.analysts._producers.sec.ratios import (
    SAVED_SEC_DERIVED_RATIO_CONTRACT,
    load_verified_derived_ratio_context_fragment,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal, artifact_binding
from dean_os.utils import json_ready


CONTRACT = "dean_domain_scoped_fundamentals_envelope_v1"
DEFAULT_SOURCE_PATH = (
    "reports/dean_os/saved_sec_derived_ratio_producer_current/latest.json"
)
DEFAULT_DISPATCH_PATH = (
    "reports/dean_os/domain_binding_task_dispatch_current/latest.json"
)
DEFAULT_OUTPUT_DIR = (
    "reports/dean_os/domain_scoped_fundamentals_envelope_current"
)
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"


class DomainScopedFundamentalsEnvelope:
    """Verify one terminal SEC artifact and bind it to one domain for review.

    The terminal derived-ratio artifact is SHA-bound to the merged fundamental
    artifact.  Existing producer loaders recursively verify the merger and its
    Company Facts / Inline XBRL sources.  This adapter runs no producer,
    collector, network request, valuation, prediction, learning, or trade.
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
        ratio = _load_json(source_file)
        dispatch = _load_json(dispatch_file)
        merger_file = _declared_merger_path(ratio)
        merger = _load_json(merger_file) if merger_file.exists() else {}

        profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        overlay = profile.get("domain_overlay") or {}
        policy = overlay.get("fundamentals_binding_policy") or {}
        expected_universe = _tickers(
            (overlay.get("market_measurement") or {}).get("primary_universe")
        )
        registry_file = _configured_registry_path(policy)
        identity = _load_yaml(registry_file) if registry_file else {}
        configured_issuers = _tickers((identity.get("issuers") or {}).keys())
        source_tickers = _tickers(
            (merger.get("inputs") or {}).get("requested_tickers")
        )
        accepted_tickers = _tickers(
            (merger.get("summary") or {}).get("accepted_fact_tickers")
        )
        ratio_tickers = _tickers(
            (ratio.get("summary") or {}).get("derived_tickers")
        )

        blockers = _structural_blockers(
            ratio=ratio,
            merger=merger,
            source_file=source_file,
            merger_file=merger_file,
            dispatch=dispatch,
            domain_id=domain_id,
            cutoff=cutoff,
            profile=profile,
            registry_file=registry_file,
            identity=identity,
            expected_universe=expected_universe,
            configured_issuers=configured_issuers,
            source_tickers=source_tickers,
            accepted_tickers=accepted_tickers,
            ratio_tickers=ratio_tickers,
        )
        fact_fragment: dict[str, Any] = {}
        ratio_fragment: dict[str, Any] = {}
        verification_error: str | None = None
        if not blockers:
            try:
                fact_fragment = load_verified_merged_fundamental_context_fragment(
                    merger_file,
                    expected_as_of=cutoff.isoformat(),
                )
                ratio_fragment = load_verified_derived_ratio_context_fragment(
                    source_file,
                    expected_as_of=cutoff.isoformat(),
                )
                if _tickers(fact_fragment.get("fundamentals", {}).keys()) != configured_issuers:
                    blockers.append(
                        "verified_fundamental_fragment_issuer_scope_mismatch"
                    )
                if _tickers(ratio_fragment.get("fundamentals", {}).keys()) != configured_issuers:
                    blockers.append(
                        "verified_ratio_fragment_issuer_scope_mismatch"
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                verification_error = f"{type(exc).__name__}: {exc}"
                blockers.append("fundamentals_recursive_lineage_verification_failed")

        blockers = sorted(set(blockers))
        coverage_gaps = _coverage_gaps(
            ratio=ratio,
            merger=merger,
            identity=identity,
            expected_universe=expected_universe,
            configured_issuers=configured_issuers,
            accepted_tickers=accepted_tickers,
        )
        candidate_ready = not blockers
        complete = candidate_ready and not coverage_gaps
        status = (
            "domain_fundamentals_candidate_ready"
            if complete
            else "domain_fundamentals_candidate_ready_with_gaps"
            if candidate_ready
            else "domain_fundamentals_envelope_blocked"
        )
        source_sha = _sha256_file(source_file)
        merger_sha = _sha256_file(merger_file) if merger_file.exists() else None
        identity_verified = candidate_ready
        lineage_verified = candidate_ready
        payload: dict[str, Any] = {
            "run_id": _run_id("domain_scoped_fundamentals_envelope"),
            "created_at": utc_now_iso(),
            "mode": "domain_scoped_fundamentals_envelope",
            "contract": CONTRACT,
            "source_producer_contract": SAVED_SEC_DERIVED_RATIO_CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "as_of": cutoff.isoformat(),
                "source_path": str(source_file),
                "source_sha256": source_sha,
                "source_run_id": ratio.get("run_id"),
                "merged_fundamental_path": str(merger_file),
                "merged_fundamental_sha256": merger_sha,
                "dispatch_path": str(dispatch_file),
                "dispatch_sha256": _sha256_file(dispatch_file),
                "identity_registry_path": (
                    str(registry_file) if registry_file else None
                ),
                "identity_registry_sha256": (
                    _sha256_file(registry_file) if registry_file else None
                ),
                "profile_domain_overlay_sha256": profile.get(
                    "domain_overlay_sha256"
                ),
            },
            "domain_binding": {
                "profile_contract": profile.get("contract"),
                "fixed_contract_sha256": profile.get("fixed_contract_sha256"),
                "domain_overlay_sha256": profile.get("domain_overlay_sha256"),
                "expected_primary_universe": expected_universe,
                "configured_issuer_scope": configured_issuers,
                "source_requested_tickers": source_tickers,
                "accepted_fact_tickers": accepted_tickers,
                "derived_ratio_tickers": ratio_tickers,
                "missing_profile_tickers": sorted(
                    set(expected_universe) - set(accepted_tickers)
                ),
                "context_role": "fundamental_context_only",
                "raw_fact_is_directional_evidence": False,
                "may_create_ticker_forecast": False,
            },
            "lineage_verification": {
                "verified": lineage_verified,
                "verification_error": verification_error,
                "terminal_contract": ratio.get("producer_contract"),
                "merged_contract": merger.get("producer_contract"),
                "declared_merged_sha256": (ratio.get("inputs") or {}).get(
                    "merged_fundamental_artifact_sha256"
                ),
                "actual_merged_sha256": merger_sha,
                "upstream_source_artifact_count": len(
                    merger.get("source_artifacts") or []
                ),
                "recursive_producer_verification_performed": candidate_ready,
            },
            "issuer_identity_verification": {
                "verified": identity_verified,
                "registry_id": identity.get("registry_id"),
                "registry_review_status": identity.get("review_status"),
                "configured_issuer_count": len(configured_issuers),
                "verified_fact_ticker_count": len(accepted_tickers),
                "ticker_cik_bindings": _ticker_cik_bindings(identity),
            },
            "coverage": {
                "status": "complete" if complete else "partial_with_gaps",
                "profile_universe_count": len(expected_universe),
                "configured_issuer_count": len(configured_issuers),
                "accepted_fact_ticker_count": len(accepted_tickers),
                "derived_ratio_ticker_count": len(ratio_tickers),
                "profile_ticker_coverage_ratio": _ratio(
                    len(set(accepted_tickers) & set(expected_universe)),
                    len(expected_universe),
                ),
                "configured_issuer_coverage_ratio": _ratio(
                    len(set(accepted_tickers) & set(configured_issuers)),
                    len(configured_issuers),
                ),
                "coverage_gaps": coverage_gaps,
                "full_cohort_comparability": bool(
                    (ratio.get("summary") or {}).get(
                        "can_claim_full_cohort_comparability"
                    )
                ),
            },
            "market_context_fragment": (
                {
                    "as_of": cutoff.isoformat(),
                    "domain_id": domain_id,
                    "fundamentals": fact_fragment.get("fundamentals", {}),
                    "derived_fundamental_ratios": ratio_fragment.get(
                        "fundamentals", {}
                    ),
                    "metadata": {
                        "domain_fundamentals_envelope_contract": CONTRACT,
                        "domain_id": domain_id,
                        "facts_metadata": fact_fragment.get("metadata", {}),
                        "ratios_metadata": ratio_fragment.get("metadata", {}),
                        "partial_coverage": bool(coverage_gaps),
                    },
                }
                if candidate_ready
                else {
                    "as_of": cutoff.isoformat(),
                    "domain_id": domain_id,
                    "fundamentals": {},
                    "derived_fundamental_ratios": {},
                }
            ),
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "structural_blockers": blockers,
                "coverage_gaps": coverage_gaps,
                "source_reuse_performed": candidate_ready,
                "source_lineage_verified": lineage_verified,
                "issuer_identity_verified": identity_verified,
                "candidate_ready_for_binding_review": candidate_ready,
                "producer_run_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "valuation_performed": False,
                "prediction_feature_created": False,
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
                "coverage_gaps_must_be_acknowledged": bool(coverage_gaps),
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "producer_run_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "valuation_performed": False,
                "prediction_feature_created": False,
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


def load_verified_domain_fundamentals_context_fragment(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    """Recursively verify a saved domain fundamentals envelope and lineage."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain fundamentals envelope contract")
    if payload.get("mode") != "domain_scoped_fundamentals_envelope":
        raise ValueError("unsupported domain fundamentals envelope mode")
    if payload.get("status") not in {
        "domain_fundamentals_candidate_ready",
        "domain_fundamentals_candidate_ready_with_gaps",
    }:
        raise ValueError("domain fundamentals envelope is not ready")

    inputs = payload.get("inputs") or {}
    domain_id = str(payload.get("domain_id") or "")
    as_of = _aware(str(inputs.get("as_of") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain fundamentals envelope identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain fundamentals envelope expected domain mismatch")
    if expected_as_of is not None and as_of != _aware(expected_as_of):
        raise ValueError("domain fundamentals envelope expected as_of mismatch")

    summary = payload.get("summary") or {}
    binding = payload.get("domain_binding") or {}
    safety = payload.get("safety") or {}
    if summary.get("structural_blockers"):
        raise ValueError("domain fundamentals envelope has structural blockers")
    for key in (
        "source_reuse_performed",
        "source_lineage_verified",
        "issuer_identity_verified",
        "candidate_ready_for_binding_review",
    ):
        if summary.get(key) is not True:
            raise ValueError(
                f"domain fundamentals envelope required flag invalid: {key}"
            )
    for key in (
        "producer_run_performed",
        "collector_run_performed",
        "network_access_performed",
        "valuation_performed",
        "prediction_feature_created",
        "binding_accepted",
        "can_update_profile_binding",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(
                f"domain fundamentals envelope forbidden flag invalid: {key}"
            )
    if (
        binding.get("context_role") != "fundamental_context_only"
        or binding.get("raw_fact_is_directional_evidence") is not False
        or binding.get("may_create_ticker_forecast") is not False
        or safety.get("review_only") is not True
    ):
        raise ValueError("domain fundamentals envelope semantic boundary invalid")

    source_file = Path(str(inputs.get("source_path") or "")).resolve()
    merger_file = Path(
        str(inputs.get("merged_fundamental_path") or "")
    ).resolve()
    dispatch_file = Path(str(inputs.get("dispatch_path") or "")).resolve()
    registry_file = Path(
        str(inputs.get("identity_registry_path") or "")
    ).resolve()
    for label, bound, expected_sha in (
        ("source", source_file, inputs.get("source_sha256")),
        ("merger", merger_file, inputs.get("merged_fundamental_sha256")),
        ("dispatch", dispatch_file, inputs.get("dispatch_sha256")),
        ("identity registry", registry_file, inputs.get("identity_registry_sha256")),
    ):
        if not bound.is_file() or _sha256_file(bound) != expected_sha:
            raise ValueError(
                f"domain fundamentals envelope {label} hash mismatch"
            )

    ratio = _load_json(source_file)
    merger = _load_json(merger_file)
    dispatch = _load_json(dispatch_file)
    identity = _load_yaml(registry_file)
    profile = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
    overlay = profile.get("domain_overlay") or {}
    policy = overlay.get("fundamentals_binding_policy") or {}
    expected_universe = _tickers(
        (overlay.get("market_measurement") or {}).get("primary_universe")
    )
    configured_registry = _configured_registry_path(policy)
    if configured_registry is None or configured_registry != registry_file:
        raise ValueError("domain fundamentals envelope registry binding mismatch")
    configured_issuers = _tickers((identity.get("issuers") or {}).keys())
    source_tickers = _tickers(
        (merger.get("inputs") or {}).get("requested_tickers")
    )
    accepted_tickers = _tickers(
        (merger.get("summary") or {}).get("accepted_fact_tickers")
    )
    ratio_tickers = _tickers(
        (ratio.get("summary") or {}).get("derived_tickers")
    )
    blockers = _structural_blockers(
        ratio=ratio,
        merger=merger,
        source_file=source_file,
        merger_file=merger_file,
        dispatch=dispatch,
        domain_id=domain_id,
        cutoff=as_of,
        profile=profile,
        registry_file=registry_file,
        identity=identity,
        expected_universe=expected_universe,
        configured_issuers=configured_issuers,
        source_tickers=source_tickers,
        accepted_tickers=accepted_tickers,
        ratio_tickers=ratio_tickers,
    )
    if blockers:
        raise ValueError(
            "domain fundamentals recursive verification failed: "
            + ",".join(blockers)
        )
    if (
        inputs.get("profile_domain_overlay_sha256")
        != profile.get("domain_overlay_sha256")
    ):
        raise ValueError("domain fundamentals profile binding mismatch")

    fact_fragment = load_verified_merged_fundamental_context_fragment(
        merger_file,
        expected_as_of=as_of.isoformat(),
    )
    ratio_fragment = load_verified_derived_ratio_context_fragment(
        source_file,
        expected_as_of=as_of.isoformat(),
    )
    fragment = payload.get("market_context_fragment") or {}
    if (
        fragment.get("domain_id") != domain_id
        or _aware(str(fragment.get("as_of") or "")) != as_of
        or _sha256_json(fragment.get("fundamentals") or {})
        != _sha256_json(fact_fragment.get("fundamentals") or {})
        or _sha256_json(fragment.get("derived_fundamental_ratios") or {})
        != _sha256_json(ratio_fragment.get("fundamentals") or {})
    ):
        raise ValueError("domain fundamentals envelope fragment mismatch")
    return {
        "as_of": as_of.isoformat(),
        "domain_id": domain_id,
        "fundamentals": dict(fact_fragment.get("fundamentals") or {}),
        "derived_fundamental_ratios": dict(
            ratio_fragment.get("fundamentals") or {}
        ),
        "metadata": {
            **dict(fragment.get("metadata") or {}),
            "domain_fundamentals_envelope_verified": True,
            "domain_fundamentals_envelope_path": str(path),
            "domain_fundamentals_envelope_sha256": _sha256_file(path),
        },
    }


def _structural_blockers(
    *,
    ratio: dict[str, Any],
    merger: dict[str, Any],
    source_file: Path,
    merger_file: Path,
    dispatch: dict[str, Any],
    domain_id: str,
    cutoff: datetime,
    profile: dict[str, Any],
    registry_file: Path | None,
    identity: dict[str, Any],
    expected_universe: list[str],
    configured_issuers: list[str],
    source_tickers: list[str],
    accepted_tickers: list[str],
    ratio_tickers: list[str],
) -> list[str]:
    blockers: list[str] = []
    if ratio.get("producer_contract") != SAVED_SEC_DERIVED_RATIO_CONTRACT:
        blockers.append("unsupported_derived_ratio_contract")
    if ratio.get("status") not in {
        "derived_ratio_evidence_ready",
        "derived_ratio_evidence_ready_with_gaps",
    }:
        blockers.append("derived_ratio_source_not_ready")
    if merger.get("producer_contract") != SAVED_SEC_FUNDAMENTAL_MERGER_CONTRACT:
        blockers.append("unsupported_merged_fundamental_contract")
    if merger.get("status") not in {
        "merged_fundamental_evidence_ready",
        "merged_fundamental_evidence_ready_with_gaps",
    }:
        blockers.append("merged_fundamental_source_not_ready")
    source_families = {
        str(item.get("family") or "")
        for item in merger.get("source_artifacts") or []
    }
    if "sec_companyfacts" not in source_families:
        blockers.append("merged_fundamental_companyfacts_lineage_missing")
    if not merger_file.exists():
        blockers.append("merged_fundamental_source_missing")
    else:
        declared = (ratio.get("inputs") or {}).get(
            "merged_fundamental_artifact_sha256"
        )
        if _sha256_file(merger_file) != declared:
            blockers.append("merged_fundamental_sha256_mismatch")
    for label, value in (
        ("derived_ratio", (ratio.get("inputs") or {}).get("as_of")),
        ("merged_fundamental", (merger.get("inputs") or {}).get("as_of")),
    ):
        try:
            if _aware(str(value or "")) != cutoff:
                blockers.append(f"{label}_as_of_mismatch")
        except ValueError:
            blockers.append(f"{label}_as_of_invalid")
    if profile.get("readiness", {}).get("schema_valid") is not True:
        blockers.append("domain_lifecycle_profile_invalid")
    if not expected_universe:
        blockers.append("domain_fundamentals_universe_missing")
    if registry_file is None or not registry_file.exists():
        blockers.append("domain_fundamentals_identity_registry_missing")
    if identity.get("domain_id") != domain_id:
        blockers.append("issuer_identity_registry_domain_mismatch")
    if not configured_issuers:
        blockers.append("issuer_identity_scope_missing")
    elif source_tickers != configured_issuers:
        blockers.append("fundamentals_requested_scope_mismatch")
    if set(configured_issuers) - set(expected_universe):
        blockers.append("configured_issuer_outside_domain_universe")
    if accepted_tickers != configured_issuers:
        blockers.append("fundamentals_configured_issuer_coverage_incomplete")
    if ratio_tickers != configured_issuers:
        blockers.append("fundamentals_ratio_issuer_coverage_incomplete")
    blockers.extend(_identity_blockers(identity, merger.get("facts") or []))
    blockers.extend(_producer_safety_blockers(ratio, merger))
    blockers.extend(_dispatch_blockers(dispatch, domain_id))
    if not source_file.exists():
        blockers.append("derived_ratio_source_missing")
    return sorted(set(blockers))


def _identity_blockers(
    identity: dict[str, Any], facts: list[dict[str, Any]]
) -> list[str]:
    issuers = identity.get("issuers") or {}
    blockers: list[str] = []
    seen_tickers: set[str] = set()
    for ticker, issuer in sorted(issuers.items()):
        cik = str((issuer or {}).get("cik") or "").zfill(10)
        if not cik.strip("0"):
            blockers.append(f"issuer_cik_missing:{ticker}")
    for fact in facts:
        ticker = str(fact.get("ticker") or "").strip().upper()
        seen_tickers.add(ticker)
        issuer = issuers.get(ticker) or {}
        expected_cik = str(issuer.get("cik") or "").zfill(10)
        actual_cik = str(fact.get("cik") or "").zfill(10)
        if not issuer:
            blockers.append(f"fact_ticker_not_in_identity_registry:{ticker}")
        elif expected_cik != actual_cik:
            blockers.append(f"fact_ticker_cik_mismatch:{ticker}")
    for ticker in sorted(set(issuers) - seen_tickers):
        blockers.append(f"configured_issuer_without_verified_fact:{ticker}")
    return blockers


def _producer_safety_blockers(
    ratio: dict[str, Any], merger: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    ratio_boundary = ratio.get("integration_boundary") or {}
    merger_safety = merger.get("safety") or {}
    if ratio_boundary.get("review_only") is not True:
        blockers.append("derived_ratio_review_only_boundary_missing")
    if merger_safety.get("review_only") is not True:
        blockers.append("merged_fundamental_review_only_boundary_missing")
    for label, payload, keys in (
        (
            "derived_ratio",
            ratio.get("safety") or {},
            (
                "network_access_performed",
                "pipeline_run_performed",
                "training_run_performed",
                "tuning_run_performed",
                "learning_write_performed",
                "live_execution_performed",
            ),
        ),
        (
            "merged_fundamental",
            merger_safety,
            (
                "network_access_performed",
                "valuation_performed",
                "pipeline_run_performed",
                "training_run_performed",
                "learning_write_performed",
                "production_config_write_performed",
                "live_execution_performed",
            ),
        ),
    ):
        for key in keys:
            if payload.get(key) is not False:
                blockers.append(f"{label}_safety_invalid:{key}")
    return blockers


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
            if item.get("context_family") == "fundamentals"
        ),
        None,
    )
    if not task:
        blockers.append("fundamentals_dispatch_task_missing")
    else:
        action = task.get("recommended_action")
        direct_actions = {
            "domain_scoped_fundamentals_evidence_envelope",
            "domain_scoped_fundamentals_envelope",
        }
        generic_adapter_dispatch = (
            action == "prepare_one_allowlisted_offline_adapter_run"
            and task.get("implementation") == "DomainScopedFundamentalsEnvelope"
            and task.get("adapter_actual_contract") == CONTRACT
        )
        if action not in direct_actions and not generic_adapter_dispatch:
            blockers.append("fundamentals_envelope_not_dispatched")
    return blockers


def _coverage_gaps(
    *,
    ratio: dict[str, Any],
    merger: dict[str, Any],
    identity: dict[str, Any],
    expected_universe: list[str],
    configured_issuers: list[str],
    accepted_tickers: list[str],
) -> list[str]:
    gaps: list[str] = []
    for ticker in sorted(set(expected_universe) - set(configured_issuers)):
        gaps.append(f"profile_issuer_not_configured:{ticker}")
    for ticker in sorted(set(configured_issuers) - set(accepted_tickers)):
        gaps.append(f"configured_issuer_without_facts:{ticker}")
    if (ratio.get("summary") or {}).get(
        "can_claim_full_cohort_comparability"
    ) is not True:
        gaps.append("full_cohort_ratio_comparability_unavailable")
    if (merger.get("summary") or {}).get(
        "can_claim_complete_sector_fundamentals"
    ) is not True:
        gaps.append("complete_sector_fundamentals_not_claimed")
    if "pending" in str(identity.get("review_status") or "") or "requires_manual" in str(
        identity.get("review_status") or ""
    ):
        gaps.append("issuer_identity_registry_manual_acceptance_pending")
    return sorted(set(gaps))


def _journal(
    *, payload: dict[str, Any], source_path: Path, journal_path: Path, apply: bool
) -> dict[str, Any]:
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["as_of"],
        "actor": "domain_scoped_fundamentals_envelope",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_binding_preview",
        "entity_id": "bind_{}_fundamentals:{}".format(
            payload["domain_id"], payload["inputs"]["source_sha256"][:16]
        ),
        "source_artifact": artifact_binding(source_path),
        "context": {"context_family": "fundamentals", "review_only": True},
        "payload": {
            "status": payload["summary"]["status"],
            "candidate_ready_for_binding_review": payload["summary"][
                "candidate_ready_for_binding_review"
            ],
            "source_lineage_verified": payload["summary"][
                "source_lineage_verified"
            ],
            "issuer_identity_verified": payload["summary"][
                "issuer_identity_verified"
            ],
            "coverage_gaps": payload["summary"]["coverage_gaps"],
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
    coverage = payload["coverage"]
    binding = payload["domain_binding"]
    lines = [
        "# DEAN-OS Domain-Scoped Fundamentals Envelope",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{summary['status']}`",
        f"- Recursive lineage verified: {summary['source_lineage_verified']}",
        f"- Issuer identity verified: {summary['issuer_identity_verified']}",
        (
            "- Profile/configured/accepted issuers: "
            f"{coverage['profile_universe_count']}/"
            f"{coverage['configured_issuer_count']}/"
            f"{coverage['accepted_fact_ticker_count']}"
        ),
        f"- Profile coverage: {coverage['profile_ticker_coverage_ratio']}",
        f"- Candidate ready: {summary['candidate_ready_for_binding_review']}",
        f"- Binding accepted: {summary['binding_accepted']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Structural blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["structural_blockers"] or ["none"])
    lines.extend(["", "## Coverage gaps", ""])
    lines.extend(f"- {item}" for item in summary["coverage_gaps"] or ["none"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- One explicit terminal ratio artifact and its complete saved SEC lineage were verified; no producer or network call ran.",
            "- Raw facts and ratios are context only. They do not create directional evidence, valuation, a prediction feature, or a ticker forecast.",
            "- Missing profile issuers and cross-cohort comparability remain explicit and must be acknowledged in any later binding decision.",
            "- Binding, analyst invocation, hypothesis approval, learning and trading remain disabled.",
            "",
            "## Covered issuers",
            "",
            "- " + (", ".join(binding["accepted_fact_tickers"]) or "none"),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _declared_merger_path(ratio: dict[str, Any]) -> Path:
    value = str(
        (ratio.get("inputs") or {}).get(
            "merged_fundamental_artifact_path", ""
        )
    ).strip()
    return Path(value).resolve() if value else Path("__missing_merger__").resolve()


def _configured_registry_path(policy: dict[str, Any]) -> Path | None:
    value = str(policy.get("issuer_identity_registry_path") or "").strip()
    return Path(value).resolve() if value else None


def _ticker_cik_bindings(identity: dict[str, Any]) -> dict[str, str]:
    return {
        str(ticker).upper(): str((payload or {}).get("cik") or "").zfill(10)
        for ticker, payload in sorted((identity.get("issuers") or {}).items())
    }


def _tickers(values: Any) -> list[str]:
    return sorted(
        {
            str(item).strip().upper()
            for item in (values or [])
            if str(item).strip()
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
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
    encoded = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "DomainScopedFundamentalsEnvelope",
    "load_verified_domain_fundamentals_context_fragment",
    "render_markdown",
]
