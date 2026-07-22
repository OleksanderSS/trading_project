from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.domain_scoped_fundamentals_envelope import (
    CONTRACT as FUNDAMENTALS_CONTRACT,
    load_verified_domain_fundamentals_context_fragment,
)
from dean_os.domain_scoped_macro_envelope import (
    DOMAIN_MACRO_ENVELOPE_CONTRACT,
    load_verified_domain_macro_context_fragment,
)
from dean_os.domain_scoped_news_envelope import (
    CONTRACT as NEWS_CONTRACT,
    load_verified_domain_news_context_fragment,
)
from dean_os.domain_scoped_official_policy_envelope import (
    CONTRACT as OFFICIAL_POLICY_CONTRACT,
    load_verified_domain_official_policy_context_fragment,
)
from dean_os.domain_scoped_pipeline_context_envelope import (
    CONTRACT as PIPELINE_CONTEXT_CONTRACT,
    load_verified_domain_pipeline_context_fragment,
)
from dean_os.domain_scoped_sector_market_envelope import (
    CONTRACT as SECTOR_MARKET_CONTRACT,
    load_verified_domain_sector_market_context_fragment,
)
from dean_os.schemas import utc_now_iso
from dean_os.system_journal import SystemJournal
from dean_os.utils import json_ready


CONTRACT = "dean_domain_context_set_v1"
DEFAULT_OUTPUT_DIR = "reports/dean_os/domain_context_set_current"
DEFAULT_JOURNAL_PATH = "data/dean_os/system_journal.jsonl"
FAMILY_ORDER = (
    "news",
    "official_policy",
    "macro",
    "fundamentals",
    "sector_market",
    "pipeline_context",
)
REQUIRED_CONTRACTS = {
    "news": NEWS_CONTRACT,
    "official_policy": OFFICIAL_POLICY_CONTRACT,
    "macro": DOMAIN_MACRO_ENVELOPE_CONTRACT,
    "fundamentals": FUNDAMENTALS_CONTRACT,
    "sector_market": SECTOR_MARKET_CONTRACT,
    "pipeline_context": PIPELINE_CONTEXT_CONTRACT,
}
_LOADERS: dict[str, Callable[..., dict[str, Any]]] = {
    "news": load_verified_domain_news_context_fragment,
    "official_policy": load_verified_domain_official_policy_context_fragment,
    "macro": load_verified_domain_macro_context_fragment,
    "fundamentals": load_verified_domain_fundamentals_context_fragment,
    "sector_market": load_verified_domain_sector_market_context_fragment,
    "pipeline_context": load_verified_domain_pipeline_context_fragment,
}


class DomainContextSetAssembler:
    """Verify six explicit domain envelopes without silently filling gaps.

    Each envelope is recursively reloaded through its family verifier. Different
    family timestamps are preserved and compared to one analysis cutoff; they
    are never rewritten to look contemporaneous. An incomplete set is useful
    for diagnosis and acquisition proposals, but cannot invoke an analyst.
    """

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str,
        analysis_cutoff: str,
        candidate_artifacts: dict[str, str | Path],
        journal_path: str | Path = DEFAULT_JOURNAL_PATH,
        apply_journal: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = _aware(analysis_cutoff)
        unknown = sorted(set(candidate_artifacts) - set(FAMILY_ORDER))
        if unknown:
            raise ValueError(
                "unsupported domain context families: " + ",".join(unknown)
            )

        receipts: list[dict[str, Any]] = []
        fragments: dict[str, dict[str, Any]] = {}
        for family in FAMILY_ORDER:
            candidate = candidate_artifacts.get(family)
            receipt, fragment = _verify_family(
                family=family,
                candidate=candidate,
                domain_id=domain_id,
                cutoff=cutoff,
            )
            receipts.append(receipt)
            if fragment is not None:
                fragments[family] = fragment

        verified = [
            item for item in receipts if item["verification_status"] == "verified"
        ]
        blocked = [
            item for item in receipts if item["verification_status"] != "verified"
        ]
        missing_families = [item["context_family"] for item in blocked]
        complete = len(verified) == len(FAMILY_ORDER)
        status = (
            "domain_context_set_candidate_ready"
            if complete
            else "domain_context_set_incomplete"
        )
        bindings = [
            {
                "context_family": item["context_family"],
                "path": item["artifact_path"],
                "sha256": item["artifact_sha256"],
                "contract": item["artifact_contract"],
                "effective_as_of": item["effective_as_of"],
            }
            for item in verified
        ]
        candidate_set_sha = _sha256_json(
            {
                "domain_id": domain_id,
                "analysis_cutoff": cutoff.isoformat(),
                "verified_artifact_bindings": bindings,
                "missing_families": missing_families,
            }
        )
        proposals = [_collection_proposal(item, domain_id) for item in blocked]
        payload: dict[str, Any] = {
            "run_id": _run_id("domain_context_set"),
            "created_at": utc_now_iso(),
            "mode": "domain_context_set_assembly",
            "contract": CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "analysis_cutoff": cutoff.isoformat(),
                "explicit_candidate_count": len(candidate_artifacts),
                "automatic_filesystem_discovery_performed": False,
            },
            "family_receipts": receipts,
            "verified_artifact_bindings": bindings,
            "verified_family_fragments": fragments,
            "candidate_set_sha256": candidate_set_sha,
            "collection_proposals": proposals,
            "summary": {
                "domain_id": domain_id,
                "status": status,
                "required_family_count": len(FAMILY_ORDER),
                "verified_family_count": len(verified),
                "blocked_family_count": len(blocked),
                "missing_families": missing_families,
                "family_timestamp_alignment_required": False,
                "all_family_timestamps_at_or_before_analysis_cutoff": all(
                    item.get("not_future_data") is True for item in verified
                ),
                "source_specific_freshness_reverified": all(
                    item.get("source_specific_freshness_reverified") is True
                    for item in verified
                ),
                "partial_context_preserved_for_review": bool(fragments),
                "partial_context_may_invoke_analyst": False,
                "binding_accepted": False,
                "can_submit_complete_set_for_binding_review": complete,
                "can_invoke_domain_analysis": False,
                "can_approve_hypothesis": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "binding_gate": {
                "status": (
                    "candidate_ready_pending_explicit_family_bindings"
                    if complete
                    else "not_open_incomplete_context_set"
                ),
                "candidate_set_sha256_binding_required": True,
                "all_six_verified_families_required": True,
                "family_bindings_accepted": [],
                "decision_recorded": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "producer_run_performed": False,
                "collector_run_performed": False,
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


def load_verified_domain_context_set(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
    expected_analysis_cutoff: str | None = None,
) -> dict[str, Any]:
    """Re-verify every referenced family and the saved set's stable content."""

    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain context set contract")
    if payload.get("mode") != "domain_context_set_assembly":
        raise ValueError("unsupported domain context set mode")
    if payload.get("status") not in {
        "domain_context_set_candidate_ready",
        "domain_context_set_incomplete",
    }:
        raise ValueError("domain context set status invalid")
    domain_id = str(payload.get("domain_id") or "")
    inputs = payload.get("inputs") or {}
    cutoff = _aware(str(inputs.get("analysis_cutoff") or ""))
    if not domain_id or inputs.get("domain_id") != domain_id:
        raise ValueError("domain context set identity invalid")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain context set expected domain mismatch")
    if (
        expected_analysis_cutoff is not None
        and cutoff != _aware(expected_analysis_cutoff)
    ):
        raise ValueError("domain context set expected analysis cutoff mismatch")

    safety = payload.get("safety") or {}
    summary = payload.get("summary") or {}
    for key in (
        "producer_run_performed",
        "collector_run_performed",
        "network_access_performed",
        "binding_write_performed",
        "analyst_invocation_performed",
        "hypothesis_approval_performed",
        "learning_write_performed",
        "production_config_write_performed",
        "broker_access_performed",
        "live_execution_performed",
    ):
        if safety.get(key) is not False:
            raise ValueError(f"domain context set forbidden safety flag: {key}")
    if safety.get("review_only") is not True:
        raise ValueError("domain context set review-only boundary invalid")
    for key in (
        "partial_context_may_invoke_analyst",
        "binding_accepted",
        "can_invoke_domain_analysis",
        "can_approve_hypothesis",
        "can_write_learning_memory",
        "can_trade",
    ):
        if summary.get(key) is not False:
            raise ValueError(f"domain context set authority flag invalid: {key}")

    saved_receipts = payload.get("family_receipts") or []
    if (
        len(saved_receipts) != len(FAMILY_ORDER)
        or [item.get("context_family") for item in saved_receipts]
        != list(FAMILY_ORDER)
    ):
        raise ValueError("domain context set family receipt topology invalid")
    rebuilt_receipts: list[dict[str, Any]] = []
    rebuilt_fragments: dict[str, dict[str, Any]] = {}
    for saved in saved_receipts:
        family = str(saved["context_family"])
        receipt, fragment = _verify_family(
            family=family,
            candidate=saved.get("artifact_path"),
            domain_id=domain_id,
            cutoff=cutoff,
        )
        rebuilt_receipts.append(receipt)
        if fragment is not None:
            rebuilt_fragments[family] = fragment

    verified = [
        item
        for item in rebuilt_receipts
        if item["verification_status"] == "verified"
    ]
    missing_families = [
        item["context_family"]
        for item in rebuilt_receipts
        if item["verification_status"] != "verified"
    ]
    bindings = [
        {
            "context_family": item["context_family"],
            "path": item["artifact_path"],
            "sha256": item["artifact_sha256"],
            "contract": item["artifact_contract"],
            "effective_as_of": item["effective_as_of"],
        }
        for item in verified
    ]
    rebuilt_sha = _sha256_json(
        {
            "domain_id": domain_id,
            "analysis_cutoff": cutoff.isoformat(),
            "verified_artifact_bindings": bindings,
            "missing_families": missing_families,
        }
    )
    rebuilt_status = (
        "domain_context_set_candidate_ready"
        if len(verified) == len(FAMILY_ORDER)
        else "domain_context_set_incomplete"
    )
    rebuilt_proposals = [
        _collection_proposal(item, domain_id)
        for item in rebuilt_receipts
        if item["verification_status"] != "verified"
    ]
    if (
        rebuilt_status != payload.get("status")
        or rebuilt_sha != payload.get("candidate_set_sha256")
        or _sha256_json(bindings)
        != _sha256_json(payload.get("verified_artifact_bindings") or [])
        or _sha256_json(rebuilt_receipts)
        != _sha256_json(saved_receipts)
        or _sha256_json(rebuilt_fragments)
        != _sha256_json(payload.get("verified_family_fragments") or {})
        or _sha256_json(rebuilt_proposals)
        != _sha256_json(payload.get("collection_proposals") or [])
    ):
        raise ValueError("domain context set saved content mismatch")
    return {
        "domain_id": domain_id,
        "analysis_cutoff": cutoff.isoformat(),
        "status": rebuilt_status,
        "candidate_set_sha256": rebuilt_sha,
        "family_receipts": rebuilt_receipts,
        "verified_family_fragments": rebuilt_fragments,
        "missing_families": missing_families,
        "collection_proposals": rebuilt_proposals,
        "complete": rebuilt_status == "domain_context_set_candidate_ready",
        "binding_accepted": False,
        "can_invoke_domain_analysis": False,
        "metadata": {
            "domain_context_set_verified": True,
            "domain_context_set_path": str(path),
            "domain_context_set_sha256": _sha256_file(path),
        },
    }


def _verify_family(
    *,
    family: str,
    candidate: str | Path | None,
    domain_id: str,
    cutoff: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base: dict[str, Any] = {
        "context_family": family,
        "required_contract": REQUIRED_CONTRACTS[family],
        "verification_status": "missing",
        "artifact_path": str(Path(candidate).resolve()) if candidate else None,
        "artifact_sha256": None,
        "artifact_contract": None,
        "effective_as_of": None,
        "age_at_analysis_cutoff_days": None,
        "not_future_data": False,
        "source_specific_freshness_reverified": False,
        "structural_blockers": ["explicit_candidate_missing"],
        "coverage_gaps": [],
    }
    if candidate is None:
        return base, None
    path = Path(candidate).resolve()
    if not path.is_file():
        base["structural_blockers"] = ["explicit_candidate_file_missing"]
        return base, None
    base["artifact_sha256"] = _sha256_file(path)
    try:
        artifact = _load_json(path)
        base["artifact_contract"] = artifact.get("contract")
        summary = artifact.get("summary") or {}
        base["coverage_gaps"] = list(summary.get("coverage_gaps") or [])
        declared_blockers = list(summary.get("structural_blockers") or [])
        if artifact.get("contract") != REQUIRED_CONTRACTS[family]:
            raise ValueError("domain context family contract mismatch")
        if artifact.get("domain_id") != domain_id:
            raise ValueError("domain context family identity mismatch")
        effective = _aware(str((artifact.get("inputs") or {}).get("as_of") or ""))
        base["effective_as_of"] = effective.isoformat()
        age_days = (cutoff - effective).total_seconds() / 86400
        base["age_at_analysis_cutoff_days"] = round(age_days, 6)
        if effective > cutoff:
            raise ValueError("domain context family contains future data")
        base["not_future_data"] = True
        fragment = _LOADERS[family](
            path,
            expected_domain_id=domain_id,
            expected_as_of=effective.isoformat(),
        )
        if fragment.get("domain_id") != domain_id:
            raise ValueError("verified domain context fragment identity mismatch")
        if _aware(str(fragment.get("as_of") or "")) != effective:
            raise ValueError("verified domain context fragment as_of mismatch")
        base["verification_status"] = "verified"
        base["source_specific_freshness_reverified"] = True
        base["structural_blockers"] = []
        return base, fragment
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        base["verification_status"] = "blocked"
        inferred = _error_code(str(exc))
        base["structural_blockers"] = sorted(
            set(declared_blockers if "declared_blockers" in locals() else [])
            | {inferred}
        )
        base["verification_error"] = f"{type(exc).__name__}: {exc}"
        return base, None


def _collection_proposal(receipt: dict[str, Any], domain_id: str) -> dict[str, Any]:
    family = str(receipt["context_family"])
    proposal: dict[str, Any] = {
        "proposal_id": f"prepare_{domain_id}_{family}",
        "domain_id": domain_id,
        "context_family": family,
        "task_type": "prepare_validated_domain_context_artifact",
        "required_contract": REQUIRED_CONTRACTS[family],
        "current_blockers": list(receipt.get("structural_blockers") or []),
        "execution_authorized": False,
        "synthetic_placeholder_allowed": False,
        "automatic_collection_allowed": False,
    }
    if family == "sector_market":
        proposal["required_upstream_chain"] = [
            "immutable_raw_market_snapshot",
            "domain_sector_market_coverage_bridge",
            "pipeline_control_saved_price_repair",
            "saved_sector_market_evidence_producer",
            "domain_scoped_sector_market_envelope",
        ]
        proposal["compatibility_warning"] = (
            "The domain coverage bridge must verify the exact 12-ticker universe "
            "plus benchmark before repair. The current clean snapshot covers only "
            "4 of 13 required tickers; a bounded network collection remains a "
            "separate, explicitly authorized action."
        )
    return proposal


def _journal(
    *, payload: dict[str, Any], journal_path: Path, apply: bool
) -> dict[str, Any]:
    journal = SystemJournal(journal_path)
    event = {
        "event_type": "action_reviewed",
        "effective_at": payload["inputs"]["analysis_cutoff"],
        "actor": "domain_context_set_assembler",
        "domain_id": payload["domain_id"],
        "entity_type": "domain_context_set_candidate",
        "entity_id": "context_set:{}:{}".format(
            payload["domain_id"], payload["candidate_set_sha256"][:16]
        ),
        "context": {
            "candidate_set_sha256": payload["candidate_set_sha256"],
            "verified_artifact_bindings": payload["verified_artifact_bindings"],
            "review_only": True,
        },
        "payload": {
            "status": payload["status"],
            "verified_family_count": payload["summary"]["verified_family_count"],
            "missing_families": payload["summary"]["missing_families"],
            "binding_accepted": False,
            "analyst_invoked": False,
            "hypothesis_approved": False,
            "learning_written": False,
            "trade_executed": False,
        },
    }
    if not apply:
        return {
            "apply_requested": False,
            "events_proposed": 1,
            "appended_count": 0,
            "existing_count": 0,
            **journal.status(),
        }
    result = journal.append_many([event])
    return {"apply_requested": True, **result, **journal.status()}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Domain Context Set",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Analysis cutoff: `{payload['inputs']['analysis_cutoff']}`",
        f"- Status: `{payload['status']}`",
        (
            "- Verified families: "
            f"{summary['verified_family_count']}/{summary['required_family_count']}"
        ),
        "- Binding accepted: False",
        "- Analyst invocation: disabled",
        "- Learning/trading: disabled",
        "",
        "## Family receipts",
        "",
        "| Family | Result | Effective as-of | Age (days) | Blockers |",
        "|---|---|---:|---:|---|",
    ]
    for item in payload["family_receipts"]:
        blockers = ", ".join(item["structural_blockers"]) or "none"
        lines.append(
            "| {family} | {status} | {as_of} | {age} | {blockers} |".format(
                family=item["context_family"],
                status=item["verification_status"],
                as_of=item["effective_as_of"] or "n/a",
                age=(
                    item["age_at_analysis_cutoff_days"]
                    if item["age_at_analysis_cutoff_days"] is not None
                    else "n/a"
                ),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "- Family timestamps remain distinct; each must be at or before the analysis cutoff.",
            "- Every accepted fragment was rebuilt through its family-specific recursive verifier.",
            "- Partial context is retained for inspection only and cannot invoke the analyst.",
            "- Missing families create proposals only; no collector or network operation is authorized.",
        ]
    )
    if payload["collection_proposals"]:
        lines.extend(["", "## Acquisition proposals", ""])
        for item in payload["collection_proposals"]:
            lines.append(
                f"- `{item['context_family']}`: "
                + ", ".join(item["current_blockers"])
            )
    return "\n".join(lines).strip() + "\n"


def _error_code(message: str) -> str:
    normalized = "_".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in message).split()
    )
    return (normalized or "family_verification_failed")[:160]


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
    encoded = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "FAMILY_ORDER",
    "REQUIRED_CONTRACTS",
    "DomainContextSetAssembler",
    "load_verified_domain_context_set",
    "render_markdown",
]
