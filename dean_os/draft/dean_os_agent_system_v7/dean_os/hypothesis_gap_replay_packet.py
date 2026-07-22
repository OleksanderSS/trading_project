from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_event_learning import WORLD_MODEL_EVENT_LEARNING_CONTRACT


class HypothesisGapReplayPacketBridge:
    """Adapt a reviewed hypothesis-gap artifact into the existing replay gate contract."""

    adapter_contract = "dean_hypothesis_gap_replay_packet_bridge_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/hypothesis_gap_replay_packet_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        gap_review_path: str | Path,
        *,
        save: bool = True,
    ) -> dict[str, Any]:
        gap_path = _latest(Path(gap_review_path))
        gap_review = _load_json(gap_path)
        _validate_gap_review(gap_review)
        analyst_ref = (gap_review.get("inputs") or {}).get("analyst_review") or {}
        analyst_path = Path(str(analyst_ref.get("path") or ""))
        if not analyst_path.is_file():
            raise FileNotFoundError(analyst_path)
        if _sha256(analyst_path) != analyst_ref.get("sha256"):
            raise ValueError("linked analyst review hash mismatch")
        analyst = _load_json(analyst_path)
        if analyst.get("contract") != "dean_domain_analyst_review_run_v1":
            raise ValueError("unsupported linked analyst review contract")
        if (analyst.get("safety") or {}).get("review_only") is not True:
            raise ValueError("linked analyst review is not review-only")

        metrics = (analyst.get("agent_report") or {}).get("metrics_snapshot") or {}
        hypotheses = list(metrics.get("hypotheses") or [])
        gap_candidates = {
            str(item.get("hypothesis_id")): item
            for item in gap_review.get("replay_task_candidates") or []
        }
        as_of = str((analyst.get("inputs") or {}).get("as_of") or "")
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError("linked analyst as_of must be timezone-aware")
        context_snapshot = _pipeline_context_snapshot(analyst)
        tasks: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            candidate = gap_candidates.get(hypothesis_id)
            if not candidate:
                continue
            for horizon in hypothesis.get("horizons_to_check") or []:
                horizon_days = int(horizon)
                tasks.append(
                    {
                        "task_id": f"replay_{hypothesis_id}_{horizon_days}d",
                        "hypothesis_id": hypothesis_id,
                        "scenario_graph_id": None,
                        "as_of": as_of,
                        "horizon_days": horizon_days,
                        "due_at": (as_of_dt + timedelta(days=horizon_days)).isoformat(),
                        "registration_status": "candidate_pending_manual_review",
                        "manual_review_gate_required": True,
                        "pipeline_context_snapshot": context_snapshot,
                        "linked_gap_ids": candidate.get("linked_gap_ids", []),
                        "expected_observations": hypothesis.get(
                            "expected_observations", []
                        ),
                        "invalidation_signals": hypothesis.get(
                            "invalidation_signals", []
                        ),
                        "review_action": "observe_outcome_and_score_hypothesis",
                        "allowed_update_after_review": [
                            "mark_hypothesis_confirmed_weakened_falsified_or_unresolved",
                            "record_false_analog_risk",
                            "propose_collector_or_template_improvement",
                        ],
                        "forbidden_update": [
                            "trade_signal",
                            "position_sizing",
                            "model_promotion_without_review",
                            "learning_memory_write_without_review",
                        ],
                    }
                )

        created_at = utc_now_iso()
        run_id = "hypothesis_gap_replay_packet_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        domain_id = str(analyst.get("domain_id") or "")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "world_model_event_learning",
            "contract": WORLD_MODEL_EVENT_LEARNING_CONTRACT,
            "adapter_contract": self.adapter_contract,
            "source_lineage": {
                "gap_review": {"path": str(gap_path), "sha256": _sha256(gap_path)},
                "analyst_review": {"path": str(analyst_path), "sha256": _sha256(analyst_path)},
            },
            "summary": {
                "packet_status": "world_model_event_learning_ready_pending_replay",
                "domain_id": domain_id,
                "as_of": as_of,
                "accepted_evidence_count": int(metrics.get("evidence_count") or 0),
                "event_record_count": 0,
                "classified_event_count": int(metrics.get("classified_event_count") or 0),
                "historical_analog_candidate_count": 0,
                "hypothesis_count": len(hypotheses),
                "evidence_gap_count": len(gap_review.get("gap_reviews") or []),
                "scenario_graph_available": False,
                "scenario_probability_mass_valid": None,
                "pipeline_indicator_context_status": context_snapshot.get(
                    "pipeline_indicator_context_status"
                ),
                "indicator_metric_count": context_snapshot.get(
                    "indicator_metric_count", 0
                ),
                "regime_label": context_snapshot.get("regime_label"),
                "expectation_context_available": context_snapshot.get(
                    "expectation_context_available", False
                ),
                "pipeline_context_tags": context_snapshot.get("context_tags", []),
                "watch_metric_count": len(context_snapshot.get("watch_metrics", [])),
                "watch_metrics": context_snapshot.get("watch_metrics", []),
                "replay_task_count": len(tasks),
                "manual_review_required": True,
                "manual_review_gate": "world_model_replay_review_gate_required",
                "can_register_replay_after_manual_review": bool(tasks),
                "can_write_learning_memory": False,
                "can_promote_model": False,
                "can_write_config": False,
                "can_trade": False,
            },
            "hypotheses": hypotheses,
            "evidence_gaps": gap_review.get("gap_reviews") or [],
            "replay_tasks": tasks,
            "operator_next_steps": [
                "Review each hypothesis, linked gap and invalidation signal.",
                "Run the replay review gate without approval first.",
                "Only an identified human reviewer may approve registration.",
            ],
            "safety": {
                "review_only": True,
                "adapter_only": True,
                "replay_task_registration_performed": False,
                "outcome_registration_performed": False,
                "learning_memory_write_performed": False,
                "model_promotion_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
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


def _pipeline_context_snapshot(analyst: dict[str, Any]) -> dict[str, Any]:
    ref = (analyst.get("inputs") or {}).get("pipeline_context_artifact") or {}
    metrics = (analyst.get("agent_report") or {}).get("metrics_snapshot") or {}
    regime = metrics.get("regime_context") or {}
    dimensions = regime.get("dimensions") or {}
    market_state = dimensions.get("market_state") or {}
    expectation_gap = metrics.get("expectation_gap")
    path_value = str(ref.get("path") or "").strip()
    if not path_value:
        return {
            "pipeline_indicator_context_status": "pipeline_context_unavailable",
            "indicator_metric_count": 0,
            "regime_label": market_state.get("state"),
            "regime_confidence": regime.get("confidence"),
            "expectation_context_available": bool(expectation_gap),
            "context_tags": [],
            "watch_metrics": ["hypothesis_outcome", "invalidation_signal"],
            "pipeline_context_artifact_sha256": None,
            "pipeline_metrics": {},
            "timeframe_lane_status": {},
        }

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(path)
    expected_hash = str(ref.get("sha256") or "")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("linked pipeline context hash mismatch")
    artifact = _load_json(path)
    if artifact.get("contract") != "dean_world_model_pipeline_context_v1":
        raise ValueError("unsupported linked pipeline context contract")
    indicator_grid = artifact.get("indicator_state_grid") or {}
    pipeline_context = artifact.get("pipeline_context") or {}
    pipeline_metrics = indicator_grid.get("metrics") or pipeline_context.get("metrics") or {}
    context_tags = indicator_grid.get("context_tags") or pipeline_context.get("context_tags") or []
    lane_status = pipeline_context.get("timeframe_lane_status") or {
        str(lane.get("timeframe")): lane.get("status")
        for lane in indicator_grid.get("timeframe_lanes") or []
        if lane.get("timeframe")
    }
    watch_metrics = sorted(
        {str(key) for key in pipeline_metrics}
        | {"hypothesis_outcome", "invalidation_signal"}
    )
    return {
        "pipeline_indicator_context_status": indicator_grid.get("status")
        or pipeline_context.get("status")
        or "pipeline_context_available",
        "indicator_metric_count": len(pipeline_metrics),
        "regime_label": market_state.get("state"),
        "regime_confidence": regime.get("confidence"),
        "expectation_context_available": bool(expectation_gap),
        "expectation_gap": expectation_gap,
        "context_tags": list(context_tags),
        "watch_metrics": watch_metrics,
        "pipeline_context_artifact_sha256": actual_hash,
        "pipeline_metrics": pipeline_metrics,
        "timeframe_lane_status": lane_status,
    }


def _validate_gap_review(payload: dict[str, Any]) -> None:
    if payload.get("contract") != "dean_hypothesis_evidence_gap_review_v1":
        raise ValueError("unsupported hypothesis gap review contract")
    if payload.get("status") != "hypothesis_gaps_reviewed_manual_action_required":
        raise ValueError("hypothesis gap review is not review-routable")
    safety = payload.get("safety") or {}
    if safety.get("review_only") is not True:
        raise ValueError("hypothesis gap review is not review-only")
    if safety.get("replay_task_registration_performed") is True:
        raise ValueError("source gap review already registered replay tasks")


def _latest(path: Path) -> Path:
    return path if path.is_file() else path / "latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
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
        "# Hypothesis Gap Replay Packet\n\n"
        f"- Hypotheses: `{summary['hypothesis_count']}`\n"
        f"- Replay tasks: `{summary['replay_task_count']}`\n"
        "- Manual review required: `true`\n"
        "- Registration performed: `false`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["HypothesisGapReplayPacketBridge"]
