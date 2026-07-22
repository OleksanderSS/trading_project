from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


class ReplayOutcomeEvidencePlanBuilder:
    """Turn routed hypothesis replays into source-specific collection plans."""

    contract = "dean_replay_outcome_evidence_plan_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_outcome_evidence_plan_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_path: str | Path,
        routing_path: str | Path,
        *,
        save: bool = True,
    ) -> dict[str, Any]:
        packet_path = Path(packet_path)
        routing_path = Path(routing_path)
        packet = _load(packet_path)
        routing = _load(routing_path)
        if packet.get("contract") != "dean_world_model_event_learning_packet_v1":
            raise ValueError("unsupported replay packet contract")
        if routing.get("contract") != "dean_replay_evaluation_routing_v1":
            raise ValueError("unsupported replay routing contract")
        if (routing.get("source_packet") or {}).get("run_id") != packet.get("run_id"):
            raise ValueError("routing source packet run_id mismatch")

        gap_ref = (packet.get("source_lineage") or {}).get("gap_review") or {}
        gap_path = Path(str(gap_ref.get("path") or ""))
        if not gap_path.is_file():
            raise FileNotFoundError(gap_path)
        if _sha256(gap_path) != gap_ref.get("sha256"):
            raise ValueError("gap review hash mismatch")
        gap_review = _load(gap_path)
        gaps = {
            str(item.get("gap_id")): item
            for item in gap_review.get("gap_reviews") or []
        }
        tasks = {
            str(item.get("task_id")): item
            for item in packet.get("replay_tasks") or []
        }
        plans = []
        for route in routing.get("routes") or []:
            if route.get("route") not in {
                "hypothesis_outcome_replay",
                "event_and_hypothesis_replay",
            }:
                continue
            task = tasks.get(str(route.get("task_id")))
            if task is None:
                raise ValueError(f"routing references unknown task {route.get('task_id')}")
            plans.append(_task_plan(task, route, gaps))

        created_at = utc_now_iso()
        run_id = "replay_outcome_evidence_plan_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        lane_references = [
            lane for plan in plans for lane in plan["evidence_lanes"]
        ]
        unresolved = sum(
            lane["resolution_status"] != "resolved"
            for lane in lane_references
        )
        unique_lanes = {
            lane["gap_id"]: lane for lane in lane_references
        }
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "replay_outcome_evidence_plan",
            "contract": self.contract,
            "inputs": {
                "packet": {"path": str(packet_path), "sha256": _sha256(packet_path)},
                "routing": {"path": str(routing_path), "sha256": _sha256(routing_path)},
                "gap_review": {"path": str(gap_path), "sha256": _sha256(gap_path)},
            },
            "summary": {
                "task_plan_count": len(plans),
                "waiting_task_count": sum(
                    plan["evaluation_status"] == "waiting" for plan in plans
                ),
                "lane_reference_count": len(lane_references),
                "unique_gap_count": len(unique_lanes),
                "unresolved_lane_reference_count": unresolved,
                "unique_gap_status_counts": dict(
                    sorted(
                        Counter(
                            lane["resolution_status"]
                            for lane in unique_lanes.values()
                        ).items()
                    )
                ),
                "expected_source_type_counts": dict(
                    sorted(
                        Counter(
                            lane["expected_source_type"]
                            for lane in unique_lanes.values()
                        ).items()
                    )
                ),
                "collection_route_status_counts": dict(
                    sorted(
                        Counter(
                            lane["collection_route"]["status"]
                            for lane in unique_lanes.values()
                        ).items()
                    )
                ),
                "voi_status_counts": dict(
                    sorted(
                        Counter(
                            lane["value_of_information"]["status"]
                            for lane in unique_lanes.values()
                        ).items()
                    )
                ),
                "voi_scored_gap_count": sum(
                    lane["value_of_information"]["triage_score"] is not None
                    for lane in unique_lanes.values()
                ),
                "collection_can_start": bool(plans),
                "outcome_evaluation_can_run": any(
                    plan["can_evaluate_outcome"] for plan in plans
                ),
                "registration_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
            "task_plans": plans,
            "collection_policy": {
                "start": "Collect evidence prospectively from task as_of.",
                "pre_due_review": "Review source coverage seven days before due_at.",
                "due_review": "Evaluate expected observations and invalidation signals at due_at.",
                "market_prices": (
                    "Price response is secondary context for hypothesis tasks; "
                    "event-study attribution requires a separate verified event timestamp."
                ),
                "closure": "No evidence gap closes automatically; human review is required.",
                "prioritization": (
                    "Do not rank collectors by missing/high labels alone. Validate decision relevance, "
                    "scenario/confidence change potential, wrong-conclusion blocking value, feasibility, "
                    "cost, assessor, and evidence basis before a VoI triage score exists."
                ),
            },
            "safety": {
                "review_only": True,
                "task_registration_performed": False,
                "outcome_write_performed": False,
                "learning_memory_write_performed": False,
                "model_promotion_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_render_markdown(payload),
                run_id=run_id,
            )
        return payload


def _task_plan(
    task: dict[str, Any],
    route: dict[str, Any],
    gaps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    as_of = parse_timezone_aware(str(task.get("as_of") or ""))
    due_at = parse_timezone_aware(str(task.get("due_at") or ""))
    if as_of is None or due_at is None:
        raise ValueError("hypothesis replay task requires timezone-aware as_of/due_at")
    lanes = []
    for gap_id in task.get("linked_gap_ids") or []:
        gap = gaps.get(str(gap_id))
        if gap is None:
            lanes.append(
                {
                    "gap_id": str(gap_id),
                    "resolution_status": "missing_gap_definition",
                    "expected_source_type": "unknown",
                    "description": "Routing task references a gap absent from gap review.",
                    "supporting_evidence_count": 0,
                    "limitations": ["gap_definition_missing"],
                    "manual_review_required": True,
                    "collection_route": _collection_route("unknown"),
                    "value_of_information": _voi_intake(),
                }
            )
            continue
        lanes.append(
            {
                "gap_id": str(gap_id),
                "resolution_status": gap.get("resolution_status"),
                "expected_source_type": gap.get("expected_source_type"),
                "description": gap.get("description"),
                "supporting_evidence_count": len(gap.get("supporting_evidence") or []),
                "limitations": list(gap.get("limitations") or []),
                "manual_review_required": True,
                "collection_route": _collection_route(
                    str(gap.get("expected_source_type") or "unknown")
                ),
                "value_of_information": _voi_intake(
                    linked_hypothesis_ids=list(gap.get("linked_hypothesis_ids") or [])
                ),
            }
        )
    pre_due = max(as_of, due_at - timedelta(days=7))
    return {
        "task_id": task.get("task_id"),
        "hypothesis_id": task.get("hypothesis_id"),
        "horizon_days": task.get("horizon_days"),
        "as_of": as_of.isoformat(),
        "due_at": due_at.isoformat(),
        "evaluation_status": route.get("evaluation_status"),
        "can_evaluate_outcome": route.get("evaluation_status")
        == "ready_for_outcome_evidence_review",
        "checkpoints": {
            "collection_start": as_of.isoformat(),
            "pre_due_source_review": pre_due.isoformat(),
            "due_outcome_review": due_at.isoformat(),
        },
        "expected_observations": list(task.get("expected_observations") or []),
        "invalidation_signals": list(task.get("invalidation_signals") or []),
        "evidence_lanes": lanes,
        "secondary_market_context": {
            "role": "context_only",
            "required_timeframes": ["15m", "60m", "1d"],
            "benchmark_required_for_abnormal_return": True,
            "event_study_allowed": False,
            "reason": "Task has no verified discrete event release timestamp.",
        },
    }


def _collection_route(source_type: str) -> dict[str, Any]:
    routes = {
        "company_filing": {
            "status": "route_available_metric_gap_open",
            "modules": [
                "SavedSECCompanyFactsProducer",
                "SavedSECInlineXBRLProducer",
                "ResearchCorpusEvidenceLoader",
            ],
            "next_action": "refresh filings and extract the requested metric with accession binding",
        },
        "company_data": {
            "status": "route_available_metric_gap_open",
            "modules": [
                "SavedSECCompanyFactsProducer",
                "SavedSECFundamentalEvidenceMerger",
            ],
            "next_action": "refresh issuer facts and retain period/unit/accession lineage",
        },
        "market_or_company_data": {
            "status": "route_available_outcome_not_matured",
            "modules": [
                "clean_market_snapshots",
                "SavedSECFundamentalEvidenceMerger",
            ],
            "next_action": "continue clean 15m/60m/1d accumulation and issuer refresh through due_at",
        },
        "earnings_call": {
            "status": "intake_path_available_source_refresh_required",
            "modules": [
                "SourceRoutingAgent",
                "SourceEvidenceValidationGate",
                "ResearchCorpusEvidenceLoader",
            ],
            "next_action": "ingest timestamped transcript/presentation and validate source provenance",
        },
        "industry_report": {
            "status": "intake_path_available_source_refresh_required",
            "modules": [
                "SourceRoutingAgent",
                "SourceEvidenceValidationGate",
                "ResearchCorpusEvidenceLoader",
            ],
            "next_action": "ingest dated industry-body or methodology-backed report",
        },
        "industry_data": {
            "status": "structured_adapter_ready_source_feed_missing",
            "modules": [
                "SourceRoutingAgent",
                "IndustryOperationalMetricsBuilder",
                "unknown_graph",
            ],
            "next_action": "connect a reviewed capacity/utilization/orders/lead-time source feed to the existing structured adapter",
        },
    }
    return routes.get(
        source_type,
        {
            "status": "manual_source_route_required",
            "modules": ["SourceRoutingAgent"],
            "next_action": "define and review a source-specific intake route",
        },
    )


def _voi_intake(*, linked_hypothesis_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "contract": "dean_unknown_value_of_information_v1",
        "status": "unassessed",
        "uncertainty_type": "unknown",
        "linked_hypothesis_ids": linked_hypothesis_ids or [],
        "scenario_change_potential": None,
        "confidence_change_potential": None,
        "wrong_conclusion_blocking_value": None,
        "decision_relevance": None,
        "collection_feasibility": None,
        "normalized_collection_cost": None,
        "evidence_basis": [],
        "assessor": None,
        "assessed_at": None,
        "triage_score": None,
        "rule": "No validated inputs means no VoI triage score and no automatic collector ordering.",
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Replay Outcome Evidence Plan\n\n"
        f"- Task plans: `{summary['task_plan_count']}`\n"
        f"- Waiting: `{summary['waiting_task_count']}`\n"
        f"- Unique gaps: `{summary['unique_gap_count']}`\n"
        f"- Unresolved lane references: `{summary['unresolved_lane_reference_count']}`\n"
        f"- Collection can start: `{str(summary['collection_can_start']).lower()}`\n"
        f"- Outcome evaluation can run: `{str(summary['outcome_evaluation_can_run']).lower()}`\n"
        "- Learning write performed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["ReplayOutcomeEvidencePlanBuilder"]
