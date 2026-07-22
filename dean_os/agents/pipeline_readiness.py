from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport

_EXPECTED_MODES = {
    "timeframe_lane_readiness": "pipeline_timeframe_lane_readiness",
    "feature_timeframe_audit": "pipeline_feature_timeframe_audit",
    "target_readiness": "pipeline_target_readiness_audit",
    "stage4_review": "pipeline_stage4_exact_context_review",
    "prediction_review": "pipeline_prediction_review_packet",
    "sector_to_ticker_review": "sector_to_ticker_review_packet",
}


def load_pipeline_readiness(
    paths: dict[str, str | Path | None],
) -> dict[str, Any]:
    bindings: dict[str, dict[str, Any]] = {}
    errors = []
    for name, expected_mode in _EXPECTED_MODES.items():
        value = paths.get(name)
        if not value:
            continue
        try:
            bindings[name] = _load_binding(
                name,
                Path(value),
                expected_mode=expected_mode,
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    blockers = []
    timeframe = bindings.get("timeframe_lane_readiness") or {}
    timeframe_summary = timeframe.get("summary") or {}
    if timeframe and (
        timeframe_summary.get("can_condition_world_model") is not True
        or int(timeframe_summary.get("source_valid_lane_count") or 0)
        < int(timeframe_summary.get("requested_lane_count") or 0)
    ):
        blockers.append("multitimeframe_context_not_ready")
    feature = bindings.get("feature_timeframe_audit") or {}
    feature_summary = feature.get("summary") or {}
    mismatch_count = int(
        feature_summary.get("timeframe_mismatch_ticker_count") or 0
    )
    if mismatch_count:
        blockers.append(
            "feature_timeframe_cadence_mismatch"
        )

    target_readiness = bindings.get("target_readiness") or {}
    target_summary = target_readiness.get("summary") or {}
    target_count = int(target_summary.get("target_count") or 0)
    ready_target_count = int(
        target_summary.get("ready_target_count") or 0
    )
    if (
        target_readiness
        and (
            target_summary.get("can_use_for_stage4") is not True
            or ready_target_count < target_count
        )
    ):
        blockers.append("stage4_targets_not_ready")

    stage4_review = bindings.get("stage4_review") or {}
    stage4_summary = stage4_review.get("summary") or {}
    if (
        stage4_review
        and stage4_summary.get("contract_passed") is not True
    ):
        blockers.append("stage4_validation_contract_failed")

    prediction = bindings.get("prediction_review") or {}
    prediction_summary = prediction.get("summary") or {}
    prediction_count = int(
        prediction_summary.get("context_count")
        or prediction.get("context_count")
        or 0
    )
    complete_prediction_count = int(
        prediction_summary.get("complete_context_count")
        or prediction.get("complete_context_count")
        or 0
    )
    if prediction and complete_prediction_count < prediction_count:
        blockers.append(
            "stage5_prediction_contexts_quarantined_or_incomplete"
        )

    sector_review = bindings.get("sector_to_ticker_review") or {}
    sector_summary = sector_review.get("summary") or {}
    if sector_review and int(sector_summary.get("review_ready_count") or 0) == 0:
        blockers.append("zero_review_ready_ticker_candidates")
    elif sector_review and sector_summary.get("can_create_ticker_forecast") is not True:
        blockers.append("sector_to_ticker_blocked_missing_ticker_evidence")

    model_contract_bound = any(
        name in bindings
        for name in (
            "feature_timeframe_audit",
            "target_readiness",
            "stage4_review",
            "prediction_review",
            "sector_to_ticker_review",
        )
    )
    if model_contract_bound and "prediction_review" not in bindings:
        blockers.append("stage5_prediction_review_not_supplied")

    feature_sha = ((feature.get("inputs") or {}).get("features_sha256"))
    target_feature_sha = (
        ((target_readiness.get("lineage_bindings") or {}).get("feature_artifact") or {}).get("sha256")
    )
    if feature_sha and target_feature_sha and feature_sha != target_feature_sha:
        errors.append("target_readiness: feature artifact SHA does not match feature audit")

    all_errors = (
        errors
        + [
            s
            for s in (
                feature_summary.get("error_message"),
                target_summary.get("error_message"),
                stage4_summary.get("error_message"),
                prediction_summary.get("error_message"),
                sector_summary.get("error_message"),
            )
            if s
        ]
    )

    all_errors = list(dict.fromkeys(errors + [item for item in all_errors if item not in errors]))
    is_ready = len(blockers) == 0 and len(all_errors) == 0
    can_use_for_stage4 = bool(
        target_readiness
        and target_summary.get("can_use_for_stage4") is True
        and not all_errors
    )
    can_use_ticker_pipeline = bool(
        is_ready
        and prediction
        and prediction_count > 0
        and complete_prediction_count == prediction_count
        and sector_review
        and sector_summary.get("can_create_ticker_forecast") is True
    )
    return {
        "status": (
            "pipeline_readiness_no_checks_bound"
            if not bindings and not all_errors
            else "pipeline_readiness_ready"
            if is_ready
            else "pipeline_readiness_blocked"
        ),
        "bindings": {
            k: _summarize_binding(k, v) for k, v in bindings.items()
        },
        "blockers": blockers,
        "blocking_reasons": list(blockers),
        "errors": all_errors,
        "is_ready": is_ready,
        "target_count": target_count,
        "ready_target_count": ready_target_count,
        "can_use_for_stage4": can_use_for_stage4,
        "stage4_contract_passed": (
            stage4_summary.get("contract_passed") if stage4_review else None
        ),
        "stage4_failed_contract_checks": list(
            stage4_summary.get("failed_contract_checks") or []
        ),
        "prediction_context_count": prediction_count,
        "complete_prediction_context_count": complete_prediction_count,
        "can_use_ticker_pipeline": can_use_ticker_pipeline,
        "decision_influence": False,
        "can_trade": False,
        "bound_count": len(bindings),
        "expected_count": len(
            [v for v in paths.values() if v]
        ),
    }


def _load_binding(
    name: str,
    path: Path,
    expected_mode: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Not a JSON object: {path}")
    mode = raw.get("mode") or raw.get("status") or raw.get("pipeline_mode") or ""
    if expected_mode not in mode:
        raise ValueError(
            f"Expected mode '{expected_mode}' but got '{mode}'"
        )
    return raw


def _summarize_binding(name: str, data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary") or {}
    return {
        "status": data.get("status") or data.get("mode") or "unknown",
        "summary_keys": list(summary.keys()) if summary else [],
        "error": summary.get("error_message"),
    }


class PipelineReadinessAgent(BaseAgent):
    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        artifact_paths = self.config.get("artifact_paths", {})
        if not artifact_paths:
            return PipelineReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="needs_more_data",
                confidence=0.5,
                data_quality_score=0.5,
                signal_strength=0.0,
                reasons=["No pipeline artifact paths configured"],
                evidence=[self.evidence("audit_finding", "pipeline_readiness", "artifact_paths", "none")],
                input_hash=self.context_hash(context),
                metrics_snapshot={},
            )

        result = load_pipeline_readiness(artifact_paths)

        if result["is_ready"]:
            return PipelineReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="clear",
                confidence=0.9,
                data_quality_score=0.9,
                signal_strength=0.8,
                reasons=[f"Pipeline readiness checks passed: {result['bound_count']} artifacts verified"],
                evidence=[
                    self.evidence("audit_finding", "pipeline_readiness", "bound_count", result["bound_count"]),
                    self.evidence("audit_finding", "pipeline_readiness", "blockers", result["blockers"]),
                    self.evidence("audit_finding", "pipeline_readiness", "errors", result["errors"]),
                ],
                input_hash=self.context_hash(context),
                metrics_snapshot={
                    "bound_count": result["bound_count"],
                    "is_ready": True,
                    "blockers": result["blockers"],
                },
            )
        else:
            reasons = result["blockers"] + result["errors"] if result["blockers"] else result["errors"]
            return PipelineReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="blocked",
                confidence=0.8,
                data_quality_score=0.3,
                signal_strength=-0.5,
                reasons=reasons[:5],
                risks=["Pipeline stages 4/5 are not ready for forecast creation"],
                blind_spots=["Artifact paths may be stale or point to outdated runs"],
                evidence=[
                    self.evidence("audit_finding", "pipeline_readiness", "blockers", result["blockers"]),
                    self.evidence("audit_finding", "pipeline_readiness", "errors", result["errors"]),
                    self.evidence("audit_finding", "pipeline_readiness", "bound_count", result["bound_count"]),
                ],
                input_hash=self.context_hash(context),
                metrics_snapshot={
                    "blockers": result["blockers"],
                    "errors": result["errors"],
                    "bound_count": result["bound_count"],
                    "is_ready": False,
                },
            )
