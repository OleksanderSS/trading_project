from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.macro import load_verified_macro_context_fragment
from dean_os.analysts._producers.news import load_verified_semiconductor_news_context_fragment
from dean_os.analysts._producers.policy import load_verified_official_policy_context_fragment
from dean_os.analysts._producers.sec.merger import load_verified_merged_fundamental_context_fragment
from dean_os.analysts._producers.sector_market import load_verified_sector_market_context_fragment
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import MarketContext
from dean_os.world_model.world_model_event_learning import (
    WorldModelEventLearningPacket,
    render_world_model_event_learning_markdown,
)
from dean_os.world_model.world_model_pipeline_context import metadata_from_pipeline_context_bundle


class FullSystemCycleWorldModelBridge:
    contract = "dean_full_system_cycle_world_model_bridge_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/world_model_event_learning_cycle_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        cycle_path: str | Path,
        pipeline_context_bundle_path: str | Path,
        domain_id: str = "semiconductor_ai_infrastructure",
        max_events: int = 12,
        save: bool = True,
    ) -> dict[str, Any]:
        cycle_file = Path(cycle_path)
        cycle = _load(cycle_file)
        bindings = verify_cycle_bindings(cycle_file, cycle)
        if cycle.get("summary", {}).get("analysis_executed") is not True:
            raise ValueError("full-system cycle did not execute domain analysis")
        if cycle.get("summary", {}).get("pipeline_context_ready") is not True:
            raise ValueError("full-system cycle pipeline context is not ready")

        bundle_file = Path(pipeline_context_bundle_path)
        bundle = _load(bundle_file)
        artifacts = cycle.get("inputs", {}).get("artifacts") or {}
        manager_payload = _load(Path(bindings["manager_report"]["path"]))
        manager_metrics = (
            (manager_payload.get("agent_report") or {}).get("metrics_snapshot")
            or {}
        )
        cycle_as_of = str(
            manager_metrics.get("as_of")
            or cycle.get("created_at")
        )
        news = _verified_fragment(
            load_verified_semiconductor_news_context_fragment,
            _artifact_path(artifacts, "news"),
            cycle_as_of,
            name="news",
        )
        policy = _optional_fragment(
            artifacts,
            "policy",
            load_verified_official_policy_context_fragment,
            cycle_as_of,
            empty_field="news",
        )
        macro = _optional_fragment(
            artifacts,
            "macro",
            load_verified_macro_context_fragment,
            cycle_as_of,
            empty_field="macro",
        )
        sector_market = _optional_fragment(
            artifacts,
            "sector_market",
            load_verified_sector_market_context_fragment,
            cycle_as_of,
            empty_field="sector_data",
        )
        fundamental = _optional_fragment(
            artifacts,
            "fundamental",
            load_verified_merged_fundamental_context_fragment,
            cycle_as_of,
            empty_field="fundamentals",
        )
        metadata = metadata_from_pipeline_context_bundle(bundle)
        metadata["verified_fragments"] = {
            "news": dict(news.get("metadata") or {}),
            "policy": dict(policy.get("metadata") or {}),
            "macro": dict(macro.get("metadata") or {}),
            "sector_market": dict(sector_market.get("metadata") or {}),
            "fundamental": dict(fundamental.get("metadata") or {}),
        }
        metadata["full_system_review_cycle_binding"] = {
            "path": str(cycle_file),
            "sha256": _sha256(cycle_file),
            "run_id": cycle.get("run_id"),
        }
        metadata["manager_report_binding"] = bindings["manager_report"]
        metadata["pipeline_context_bundle_binding"] = {
            "path": str(bundle_file),
            "sha256": _sha256(bundle_file),
        }
        metadata["pipeline_context"] = manager_payload
        context = MarketContext(
            as_of=cycle_as_of,
            news=[
                *list(news.get("news") or []),
                *list(policy.get("news") or []),
            ],
            fundamentals=dict(fundamental.get("fundamentals") or {}),
            macro=dict(macro.get("macro") or {}),
            sector_data=dict(sector_market.get("sector_data") or {}),
            metadata=metadata,
            pipeline_result=bundle.get("pipeline_context") or manager_payload,
            timeframes=["15m", "60m", "1d"],
        )
        payload = WorldModelEventLearningPacket(output_dir=self.output_dir).build(
            context,
            domain_id=domain_id,
            as_of=cycle_as_of,
            max_events=max_events,
            save=False,
        )
        payload["cycle_binding_contract"] = self.contract
        payload["upstream_bindings"] = {
            **bindings,
            "pipeline_context_bundle": {
                "path": str(bundle_file),
                "sha256": _sha256(bundle_file),
            },
        }
        payload["upstream_domain_analysis"] = {
            "hypotheses": [
                _sector_hypothesis_payload(item)
                for item in list(manager_metrics.get("hypotheses") or [])
            ],
            "evidence_gaps": list(manager_metrics.get("evidence_gaps") or []),
            "transmission_channels": list(
                manager_metrics.get("transmission_channels") or []
            ),
            "regime_context": manager_metrics.get("regime_context"),
            "expectation_gap": manager_metrics.get("expectation_gap"),
        }
        payload["summary"]["upstream_domain_hypothesis_count"] = len(
            payload["upstream_domain_analysis"]["hypotheses"]
        )
        payload["summary"]["upstream_domain_evidence_gap_count"] = len(
            payload["upstream_domain_analysis"]["evidence_gaps"]
        )
        alignment = _hypothesis_alignment(payload)
        payload["hypothesis_alignment_review"] = alignment
        payload["summary"]["aligned_upstream_hypothesis_count"] = alignment[
            "summary"
        ]["aligned_upstream_hypothesis_count"]
        payload["summary"]["unaligned_upstream_hypothesis_count"] = alignment[
            "summary"
        ]["unaligned_upstream_hypothesis_count"]
        payload["summary"]["hypothesis_alignment_status"] = alignment["summary"][
            "status"
        ]
        _bind_replay_tasks_to_upstream(payload, alignment)
        payload["summary"]["downstream_hash_binding_ready"] = True
        payload["safety"]["upstream_cycle_hash_verified"] = True
        payload["safety"]["manager_report_hash_verified"] = True
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_world_model_event_learning_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def verify_cycle_bindings(cycle_path: Path, cycle: dict[str, Any]) -> dict[str, Any]:
    if cycle.get("contract") != "dean_full_system_review_cycle_v1":
        raise ValueError("unsupported full-system review cycle contract")
    manager = cycle.get("manager_report") or {}
    manager_path = Path(str(manager.get("path") or ""))
    if not manager_path.is_file():
        raise ValueError("cycle manager report is missing")
    actual_manager_sha = _sha256(manager_path)
    if manager.get("sha256") != actual_manager_sha:
        raise ValueError("cycle manager report SHA-256 mismatch")
    for name, binding in (cycle.get("inputs", {}).get("artifacts") or {}).items():
        path = Path(str(binding.get("path") or ""))
        if not path.is_file() or binding.get("sha256") != _sha256(path):
            raise ValueError(f"cycle input artifact binding mismatch: {name}")
    readiness = cycle.get("inputs", {}).get("timeframe_lane_readiness") or {}
    readiness_path = Path(str(readiness.get("path") or ""))
    if not readiness_path.is_file() or readiness.get("sha256") != _sha256(readiness_path):
        raise ValueError("cycle timeframe readiness binding mismatch")
    return {
        "full_system_review_cycle": {
            "path": str(cycle_path),
            "sha256": _sha256(cycle_path),
            "run_id": cycle.get("run_id"),
        },
        "manager_report": {
            "path": str(manager_path),
            "sha256": actual_manager_sha,
        },
        "timeframe_lane_readiness": {
            "path": str(readiness_path),
            "sha256": _sha256(readiness_path),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(artifacts: dict[str, Any], name: str) -> Path:
    path = Path(str((artifacts.get(name) or {}).get("path") or ""))
    if not path.is_file():
        raise ValueError(f"cycle artifact is missing: {name}")
    return path


def _optional_fragment(
    artifacts: dict[str, Any],
    name: str,
    loader: Any,
    as_of: str,
    *,
    empty_field: str,
) -> dict[str, Any]:
    if name not in artifacts:
        return {
            "as_of": as_of,
            empty_field: {} if empty_field != "news" else [],
            "metadata": {"status": "not_supplied"},
        }
    return _verified_fragment(
        loader,
        _artifact_path(artifacts, name),
        as_of,
        name=name,
    )


def _verified_fragment(
    loader: Any,
    path: Path,
    analysis_as_of: str,
    *,
    name: str,
) -> dict[str, Any]:
    fragment = loader(path)
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    cycle_as_of = parse_timezone_aware(analysis_as_of)
    if fragment_as_of is None or cycle_as_of is None:
        raise ValueError(f"{name} fragment or cycle as_of is invalid")
    if fragment_as_of > cycle_as_of:
        raise ValueError(f"{name} fragment is future relative to cycle as_of")
    return fragment


def _sector_hypothesis_payload(item: Any) -> dict[str, Any]:
    payload = dict(item) if isinstance(item, dict) else {}
    payload.update(
        {
            "hypothesis_scope": "sector_thesis",
            "horizon_family": "sector_thesis_monitoring_v1",
        }
    )
    return payload


def _mechanism_from_hypothesis(text: Any) -> str:
    value = str(text or "").lower()
    if "capex cycle" in value:
        return "capex_cycle"
    if "ai demand" in value:
        return "sector_demand"
    if "supply constraint" in value:
        return "supply_chain"
    if "sanction" in value or "market access" in value:
        return "policy_or_geopolitical"
    if "tariff" in value:
        return "policy_or_geopolitical"
    return "unmapped"


def _hypothesis_alignment(payload: dict[str, Any]) -> dict[str, Any]:
    world = list(payload.get("hypotheses") or [])
    upstream = list((payload.get("upstream_domain_analysis") or {}).get("hypotheses") or [])
    world_by_mechanism: dict[str, list[dict[str, Any]]] = {}
    for item in world:
        world_by_mechanism.setdefault(
            _mechanism_from_hypothesis(item.get("hypothesis")), []
        ).append(item)
    rows: list[dict[str, Any]] = []
    aligned = 0
    for item in upstream:
        mechanism = _mechanism_from_hypothesis(item.get("hypothesis"))
        matches = world_by_mechanism.get(mechanism, [])
        if matches:
            aligned += 1
        rows.append(
            {
                "upstream_hypothesis_id": item.get("hypothesis_id"),
                "upstream_hypothesis": item.get("hypothesis"),
                "mechanism": mechanism,
                "upstream_horizon_family": item.get("horizon_family"),
                "upstream_horizons_days": list(item.get("horizons_to_check") or []),
                "world_hypothesis_ids": [match.get("hypothesis_id") for match in matches],
                "world_horizon_family": (
                    matches[0].get("horizon_family") if matches else None
                ),
                "world_horizons_days": (
                    list(matches[0].get("horizons_to_check") or []) if matches else []
                ),
                "alignment_status": (
                    "event_response_candidate_mapped"
                    if matches
                    else "no_event_response_candidate_in_bounded_sample"
                ),
                "manual_review_required": True,
            }
        )
    unaligned = len(upstream) - aligned
    return {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "status": (
                "all_upstream_mechanisms_mapped_pending_manual_review"
                if upstream and unaligned == 0
                else "upstream_mechanism_disposition_required"
                if unaligned
                else "no_upstream_hypotheses"
            ),
            "upstream_hypothesis_count": len(upstream),
            "world_hypothesis_count": len(world),
            "aligned_upstream_hypothesis_count": aligned,
            "unaligned_upstream_hypothesis_count": unaligned,
            "manual_review_required": True,
            "horizon_substitution_allowed": False,
        },
        "horizon_contract": {
            "sector_thesis": {
                "family": "sector_thesis_monitoring_v1",
                "purpose": "validate the slower domain thesis",
            },
            "event_response": {
                "family": "event_response_fixed_v1",
                "purpose": "observe the market/fundamental response to a dated event",
            },
            "rule": (
                "Event-response horizons complement sector-thesis horizons; "
                "neither family replaces the other."
            ),
        },
        "alignments": rows,
    }


def _bind_replay_tasks_to_upstream(
    payload: dict[str, Any], alignment: dict[str, Any]
) -> None:
    reverse: dict[str, list[str]] = {}
    for row in alignment.get("alignments", []):
        upstream_id = str(row.get("upstream_hypothesis_id") or "")
        for world_id in row.get("world_hypothesis_ids", []):
            reverse.setdefault(str(world_id), []).append(upstream_id)
    for task in payload.get("replay_tasks", []):
        task["related_upstream_hypothesis_ids"] = sorted(
            item for item in reverse.get(str(task.get("hypothesis_id")), []) if item
        )


__all__ = ["FullSystemCycleWorldModelBridge", "verify_cycle_bindings"]
