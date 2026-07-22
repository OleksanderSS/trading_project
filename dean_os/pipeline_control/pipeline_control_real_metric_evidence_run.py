from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_caution_review_packet import PipelineControlCautionReviewPacket
from dean_os.pipeline_control.pipeline_control_evidence_inventory import (
    verify_locked_feature_stability,
    verify_locked_model_evaluation,
)
from dean_os.pipeline_control.pipeline_control_instance_contract import PipelineControlInstanceContract
from dean_os.pipeline_control.pipeline_control_surface import PipelineControlSurface
from dean_os.pipeline_metric_input_readiness_gate import PipelineMetricInputReadinessGate
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REPLAY_BATCH_JSON = "reports/dean_os/historical_replay_batch_repaired_expanded/latest.json"
DEFAULT_DATA_QUALITY_JSON = "diagnostic_reports/feature_lineage_report_current_cache.json"
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"
DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON = "reports/dean_os/domain_analyst_instance_contract_current/latest.json"

MODEL_REQUIRED_METRICS = {
    "max_drawdown": ["max_drawdown", "maximum_drawdown", "mdd", "drawdown"],
    "train_score": ["train_score", "training_score", "in_sample_score"],
    "validation_score": ["validation_score", "val_score", "test_score", "out_of_sample_score"],
    "sample_count": ["sample_count", "n_samples", "test_samples", "validation_samples", "observations"],
}
MODEL_OPTIONAL_PROFITABILITY = {
    "profitability": ["total_return", "return", "realized_return", "pnl_pct", "pnl", "profit", "net_profit", "sharpe", "sharpe_ratio"],
}
METRIC_PAIR_LINEAGE_FIELDS = ("ticker", "model", "target_name", "timeframe", "context_fingerprint")


class PipelineControlRealMetricEvidenceRun:
    """Runs pipeline-control gates from real locked metric artifacts.

    This is the evidence counterpart to PipelineControlMetricFixtureValidation.
    Synthetic fixtures may prove control flow, but only this runner can accept
    saved model evaluation and feature-stability artifacts as metric evidence.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/pipeline_control_real_metric_evidence_run"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        model_evaluation_json: str | Path | None = None,
        feature_stability_report: str | Path | None = None,
        replay_batch_json: str | Path | None = DEFAULT_REPLAY_BATCH_JSON,
        data_quality_json: str | Path | None = DEFAULT_DATA_QUALITY_JSON,
        constraints_path: str | Path | None = None,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        domain_instance_contract_json: str | Path | None = DEFAULT_DOMAIN_ANALYST_INSTANCE_CONTRACT_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        run_id = _run_id("pipeline_control_real_metric_evidence_run")
        work_dir = self.output_dir / run_id
        evidence_checks = _evidence_checks(
            model_evaluation_json=model_evaluation_json,
            feature_stability_report=feature_stability_report,
            replay_batch_json=replay_batch_json,
            data_quality_json=data_quality_json,
        )

        readiness = PipelineMetricInputReadinessGate(output_dir=work_dir / "readiness").build(
            model_performance_path=model_evaluation_json,
            replay_batch_path=replay_batch_json,
            feature_report_path=feature_stability_report,
            data_quality_path=data_quality_json,
            constraints_path=constraints_path,
        )
        surface = PipelineControlSurface(output_dir=work_dir / "surface").run(
            model_performance_path=model_evaluation_json,
            replay_batch_path=replay_batch_json,
            feature_report_path=feature_stability_report,
            data_quality_path=data_quality_json,
            constraints_path=constraints_path,
        )
        instance = PipelineControlInstanceContract(output_dir=work_dir / "instance").build(
            pipeline_surface_json=surface["saved_paths"]["latest_json"],
            architecture_map_json=architecture_map_json,
            domain_instance_contract_json=domain_instance_contract_json,
        )
        caution_review = PipelineControlCautionReviewPacket(output_dir=work_dir / "caution_review").build(
            pipeline_metric_input_readiness_json=readiness["saved_paths"]["latest_json"],
            pipeline_control_instance_json=instance["saved_paths"]["latest_json"],
            model_performance_report_json=model_evaluation_json,
            feature_report_json=feature_stability_report,
            data_quality_json=data_quality_json,
        )

        chain = _chain_results(readiness, surface, instance, caution_review)
        status = _real_metric_status(evidence_checks, chain)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_real_metric_evidence_run",
            "inputs": {
                "model_evaluation_json": str(model_evaluation_json) if model_evaluation_json else None,
                "model_evaluation_sha256": _file_sha256(
                    model_evaluation_json
                ),
                "feature_stability_report": str(feature_stability_report) if feature_stability_report else None,
                "feature_stability_sha256": _file_sha256(
                    feature_stability_report
                ),
                "replay_batch_json": str(replay_batch_json) if replay_batch_json else None,
                "data_quality_json": str(data_quality_json) if data_quality_json else None,
                "constraints_path": str(constraints_path) if constraints_path else None,
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
                "domain_instance_contract_json": str(domain_instance_contract_json) if domain_instance_contract_json else None,
            },
            "summary": _summary(status, evidence_checks, chain),
            "real_evidence_contract": _real_evidence_contract(),
            "input_evidence_checks": evidence_checks,
            "chain_results": chain,
            "explicit_non_actions": _explicit_non_actions(),
            "operator_next_steps": _operator_next_steps(status, evidence_checks, chain),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_control_real_metric_evidence_run_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_pipeline_control_real_metric_evidence_run_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Pipeline Control Real Metric Evidence Run",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Real metric evidence status: `{summary.get('real_metric_evidence_status')}`",
        f"- Can use as metric evidence: {summary.get('can_use_as_metric_evidence')}",
        f"- Can clear current real cautions: {summary.get('can_clear_current_real_cautions')}",
        f"- Readiness status: `{summary.get('readiness_status')}`",
        f"- Surface status: `{summary.get('surface_status')}`",
        f"- Instance status: `{summary.get('instance_status')}`",
        f"- Caution review status: `{summary.get('caution_review_status')}`",
        f"- Blocked planes: {', '.join(summary.get('blocked_metric_planes', [])) or 'none'}",
        f"- Caution planes: {', '.join(summary.get('caution_metric_planes', [])) or 'none'}",
        f"- Can write production config: {summary.get('can_write_production_config')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Input Evidence Checks",
        "",
    ]
    for check in payload.get("input_evidence_checks", []):
        lines.append(f"- {check.get('status').upper()}: `{check.get('code')}` - {check.get('message')}")

    lines.extend(["", "## Chain Results", ""])
    for item in payload.get("chain_results", []):
        lines.append(f"- `{item.get('step_id')}`: {item.get('status')} path=`{item.get('latest_json')}`")

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Operator Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    return "\n".join(lines).strip() + "\n"


def _evidence_checks(
    *,
    model_evaluation_json: str | Path | None,
    feature_stability_report: str | Path | None,
    replay_batch_json: str | Path | None,
    data_quality_json: str | Path | None,
) -> list[dict[str, str]]:
    checks = []
    model = _load_evidence_json(model_evaluation_json)
    feature = _load_evidence_json(feature_stability_report)
    replay = _load_evidence_json(replay_batch_json)
    data_quality = _load_evidence_json(data_quality_json)
    checks.extend(_model_evaluation_checks(model))
    checks.extend(_feature_stability_checks(feature))
    checks.extend(_metric_pair_lineage_checks(model, feature))
    checks.append(_supporting_artifact_check("replay_batch_json", replay, required=False))
    checks.append(_supporting_artifact_check("data_quality_json", data_quality, required=False))
    return checks


def _model_evaluation_checks(artifact: dict[str, Any]) -> list[dict[str, str]]:
    if not artifact["available"]:
        return [_check("fail", "model_evaluation_json_available", artifact["message"])]
    payload = artifact["payload"]
    checks = [
        _check("pass", "model_evaluation_json_available", f"Loaded {artifact['path']}."),
        _not_synthetic_check("model_evaluation_json", payload),
    ]
    provenance = verify_locked_model_evaluation(payload)
    checks.append(
        _check(
            "pass" if provenance["valid"] else "fail",
            "model_evaluation_locked_provenance_verified",
            (
                f"Locked model provenance verified via {provenance['proof']}."
                if provenance["valid"]
                else "Invalid locked model provenance: "
                + ", ".join(provenance["failures"])
                + "."
            ),
        )
    )
    metrics = _flatten(payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload)
    missing = [
        canonical
        for canonical, aliases in MODEL_REQUIRED_METRICS.items()
        if _first_number(metrics, aliases) is None
    ]
    checks.append(
        _check(
            "pass" if not missing else "fail",
            "model_evaluation_required_metrics_present",
            "Required model evaluation metrics present." if not missing else "Missing: " + ", ".join(missing) + ".",
        )
    )
    has_profitability = any(_first_number(metrics, aliases) is not None for aliases in MODEL_OPTIONAL_PROFITABILITY.values())
    checks.append(
        _check(
            "pass" if has_profitability else "warn",
            "model_evaluation_profitability_metric_present",
            "Profitability metric present." if has_profitability else "No total_return, pnl, or sharpe metric found; replay may only provide a proxy.",
        )
    )
    return checks


def _feature_stability_checks(artifact: dict[str, Any]) -> list[dict[str, str]]:
    if not artifact["available"]:
        return [_check("fail", "feature_stability_report_available", artifact["message"])]
    payload = artifact["payload"]
    checks = [
        _check("pass", "feature_stability_report_available", f"Loaded {artifact['path']}."),
        _not_synthetic_check("feature_stability_report", payload),
    ]
    provenance = verify_locked_feature_stability(payload)
    checks.append(
        _check(
            "pass" if provenance["valid"] else "fail",
            "feature_stability_locked_provenance_verified",
            (
                f"Locked feature provenance verified via {provenance['proof']}."
                if provenance["valid"]
                else "Invalid locked feature provenance: "
                + ", ".join(provenance["failures"])
                + "."
            ),
        )
    )
    importances = _extract_feature_importances(payload)
    flattened = _flatten(payload)
    has_stability_score = _first_number(flattened, ["feature_stability_score", "stability_score"]) is not None
    has_unstable_count = _first_number(flattened, ["unstable_feature_count", "unstable_features_count"]) is not None
    has_unstable_list = isinstance(_find_key(payload, "unstable_features"), list)
    has_stability_signal = has_stability_score or has_unstable_count or has_unstable_list
    if not importances:
        checks.append(
            _check(
                "fail",
                "feature_stability_importances_present",
                "Feature importances or weights are missing.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "feature_stability_importances_present",
                f"Recognized {len(importances)} feature weights.",
            )
        )
    checks.append(
        _check(
            "pass" if has_stability_signal else "fail",
            "feature_stability_signal_present",
            (
                "Feature stability score or unstable-feature signal present."
                if has_stability_signal
                else "Missing feature_stability_score, unstable_feature_count, or unstable_features."
            ),
        )
    )
    return checks


def _metric_pair_lineage_checks(
    model_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
) -> list[dict[str, str]]:
    if not model_artifact["available"] or not feature_artifact["available"]:
        return []
    model_lineage = _extract_metric_lineage(model_artifact["payload"], lineage_container="joined_lineage")
    feature_lineage = _extract_metric_lineage(feature_artifact["payload"], lineage_container="training_lineage")
    checks = []
    for field in METRIC_PAIR_LINEAGE_FIELDS:
        model_value = model_lineage.get(field)
        feature_value = feature_lineage.get(field)
        if _lineage_value_missing(model_value) or _lineage_value_missing(feature_value):
            checks.append(
                _check(
                    "fail",
                    f"metric_pair_{field}_present",
                    f"Both locked metric artifacts must declare {field} lineage.",
                )
            )
            continue
        matches = _normalize_lineage_value(model_value) == _normalize_lineage_value(feature_value)
        checks.append(
            _check(
                "pass" if matches else "fail",
                f"metric_pair_{field}_matches",
                (
                    f"Metric-pair {field} lineage matches."
                    if matches
                    else f"Metric-pair {field} mismatch: model={model_value!r}, feature={feature_value!r}."
                ),
            )
        )
    return checks


def _extract_metric_lineage(payload: dict[str, Any], *, lineage_container: str) -> dict[str, Any]:
    nested = payload.get(lineage_container)
    nested = nested if isinstance(nested, dict) else {}
    return {
        "ticker": nested.get("ticker") or _find_key(payload, "ticker") or _find_key(payload, "symbol"),
        "model": nested.get("model")
        or _find_key(payload, "model_id")
        or _find_key(payload, "model_name")
        or _find_key(payload, "model_type")
        or _find_key(payload, "selected_primary_model"),
        "target_name": nested.get("target_name")
        or _find_key(payload, "target_name")
        or _find_key(payload, "target")
        or _find_key(payload, "target_column"),
        "timeframe": nested.get("timeframe")
        or _find_key(payload, "timeframe")
        or _find_key(payload, "horizon")
        or _find_key(payload, "prediction_horizon"),
        "context_fingerprint": nested.get("context_fingerprint")
        or _find_key(payload, "context_fingerprint")
        or _find_key(payload, "context_key")
        or _find_key(payload, "regime_fingerprint"),
    }


def _lineage_value_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_lineage_value(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _supporting_artifact_check(artifact_id: str, artifact: dict[str, Any], *, required: bool) -> dict[str, str]:
    if not artifact["available"]:
        return _check("fail" if required else "warn", f"{artifact_id}_available", artifact["message"])
    synthetic = _is_synthetic_or_fixture(artifact["payload"])
    return _check(
        "fail" if synthetic else "pass",
        f"{artifact_id}_not_synthetic",
        "Synthetic or fixture supporting artifact rejected." if synthetic else f"Loaded {artifact['path']}.",
    )


def _not_synthetic_check(artifact_id: str, payload: dict[str, Any]) -> dict[str, str]:
    synthetic = _is_synthetic_or_fixture(payload)
    return _check(
        "fail" if synthetic else "pass",
        f"{artifact_id}_not_synthetic",
        "Synthetic or fixture artifact rejected." if synthetic else "Artifact is not marked synthetic or fixture-only.",
    )


def _chain_results(
    readiness: dict[str, Any],
    surface: dict[str, Any],
    instance: dict[str, Any],
    caution_review: dict[str, Any],
) -> list[dict[str, Any]]:
    readiness_path = readiness.get("saved_paths", {}).get("latest_json")
    surface_path = surface.get("saved_paths", {}).get("latest_json")
    instance_path = instance.get("saved_paths", {}).get("latest_json")
    caution_review_path = caution_review.get("saved_paths", {}).get(
        "latest_json"
    )
    return [
        {
            "step_id": "pipeline_metric_input_readiness",
            "status": readiness.get("summary", {}).get("readiness_status"),
            "latest_json": readiness_path,
            "latest_json_sha256": _file_sha256(readiness_path),
            "blocked_metric_planes": readiness.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": readiness.get("summary", {}).get("caution_metric_planes", []),
        },
        {
            "step_id": "pipeline_control_surface",
            "status": surface.get("surface", {}).get("status"),
            "latest_json": surface_path,
            "latest_json_sha256": _file_sha256(surface_path),
            "blocked_metric_planes": surface.get("surface", {}).get("allowed_variation", {}).get("blocked_axes", []),
            "caution_metric_planes": surface.get("surface", {}).get("allowed_variation", {}).get("caution_axes", []),
        },
        {
            "step_id": "pipeline_control_instance_contract",
            "status": instance.get("summary", {}).get("instance_status"),
            "latest_json": instance_path,
            "latest_json_sha256": _file_sha256(instance_path),
            "blocked_metric_planes": instance.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": instance.get("summary", {}).get("caution_metric_planes", []),
        },
        {
            "step_id": "pipeline_control_caution_review",
            "status": caution_review.get("summary", {}).get("caution_review_status"),
            "latest_json": caution_review_path,
            "latest_json_sha256": _file_sha256(caution_review_path),
            "blocked_metric_planes": caution_review.get("summary", {}).get("blocked_metric_planes", []),
            "caution_metric_planes": caution_review.get("summary", {}).get("caution_metric_planes", []),
        },
    ]


def _summary(status: str, checks: list[dict[str, str]], chain: list[dict[str, Any]]) -> dict[str, Any]:
    by_step = {item["step_id"]: item for item in chain}
    blocked = _unique([plane for item in chain for plane in item.get("blocked_metric_planes", [])])
    caution = _unique([plane for item in chain for plane in item.get("caution_metric_planes", [])])
    failed_checks = [check["code"] for check in checks if check["status"] == "fail"]
    warning_checks = [check["code"] for check in checks if check["status"] == "warn"]
    return {
        "real_metric_evidence_status": status,
        "failed_input_evidence_checks": failed_checks,
        "warning_input_evidence_checks": warning_checks,
        "can_use_as_metric_evidence": status != "real_metric_evidence_rejected",
        "can_clear_current_real_cautions": status == "real_metric_evidence_chain_ready",
        "readiness_status": by_step["pipeline_metric_input_readiness"]["status"],
        "surface_status": by_step["pipeline_control_surface"]["status"],
        "instance_status": by_step["pipeline_control_instance_contract"]["status"],
        "caution_review_status": by_step["pipeline_control_caution_review"]["status"],
        "blocked_metric_planes": blocked,
        "caution_metric_planes": caution,
        "current_artifacts_overwritten": False,
        "can_run_autonomous_tuning_now": False,
        "can_write_production_config": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _real_metric_status(checks: list[dict[str, str]], chain: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "real_metric_evidence_rejected"
    blocked_statuses = {
        "blocked_metric_inputs",
        "blocked",
        "blocked_pipeline_control_instance",
        "pipeline_caution_review_blocked_by_hard_planes",
    }
    if any(item.get("status") in blocked_statuses for item in chain):
        return "real_metric_evidence_blocked_by_metric_planes"
    caution_statuses = {
        "metric_inputs_ready_with_cautions",
        "caution",
        "pipeline_control_instance_review_ready_with_cautions",
        "pipeline_cautions_need_reviewed_inputs",
    }
    if any(item.get("status") in caution_statuses for item in chain):
        return "real_metric_evidence_ready_with_cautions"
    return "real_metric_evidence_chain_ready"


def _real_evidence_contract() -> dict[str, Any]:
    return {
        "model_evaluation_json_required_metrics": MODEL_REQUIRED_METRICS,
        "feature_stability_report_required_fields": [
            "feature_importance or feature_importances or feature_weights",
            "feature_stability_score or unstable_feature_count or unstable_features",
        ],
        "metric_pair_required_matching_lineage": list(METRIC_PAIR_LINEAGE_FIELDS),
        "accepted_artifact_class": "saved past or locked evaluation artifact",
        "rejected_artifact_class": "synthetic fixture, control-flow proof, live collector output, code audit, clean lineage alone",
        "fixed_chain": [
            "model_evaluation_json + feature_stability_report",
            "PipelineMetricInputReadinessGate",
            "PipelineControlSurface",
            "PipelineControlInstanceContract",
            "PipelineControlCautionReviewPacket",
        ],
    }


def _operator_next_steps(status: str, checks: list[dict[str, str]], chain: list[dict[str, Any]]) -> list[str]:
    failed = [check["code"] for check in checks if check["status"] == "fail"]
    if status == "real_metric_evidence_rejected":
        return [
            "Supply non-synthetic saved evaluation artifacts before treating this run as metric evidence.",
            "Failed input evidence checks: " + ", ".join(failed) + ".",
            "Keep the synthetic fixture validation separate; it proves control flow only.",
        ]
    blocked = _unique([plane for item in chain for plane in item.get("blocked_metric_planes", [])])
    if status == "real_metric_evidence_blocked_by_metric_planes":
        return [
            "Do not hand this surface to tuning proposals until blocked metric planes are repaired: " + ", ".join(blocked) + ".",
            "Refresh the same real-evidence chain after the underlying metric artifact is corrected.",
        ]
    caution = _unique([plane for item in chain for plane in item.get("caution_metric_planes", [])])
    if status == "real_metric_evidence_ready_with_cautions":
        return [
            "Review the remaining caution planes before widening any experiment bounds: " + ", ".join(caution) + ".",
            "A human may accept one tiny bounded proposal, but autonomous tuning, config writes, learning writes, and trading stay closed.",
        ]
    return [
        "Real metric evidence clears the current pipeline-control chain for manual proposal review.",
        "Use this only as a reviewed experiment input; production config, learning promotion, recommendations, and trading remain separate gates.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No replay is rerun on market data.",
        "No model training or hyperparameter search is executed.",
        "No current pipeline-control artifact is overwritten.",
        "No production config is written.",
        "No learning memory or analyst-weight update is written.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
    ]


def _load_evidence_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"available": False, "path": None, "payload": {}, "message": "No path was supplied."}
    resolved = Path(path)
    if not resolved.exists():
        return {"available": False, "path": str(path), "payload": {}, "message": f"Missing file: {path}"}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"available": False, "path": str(path), "payload": {}, "message": f"Invalid JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"available": False, "path": str(path), "payload": {}, "message": "Expected JSON object."}
    return {"available": True, "path": str(path), "payload": payload, "message": f"Loaded {path}."}


def _is_synthetic_or_fixture(payload: dict[str, Any]) -> bool:
    if payload.get("fixture_not_evidence") is True or payload.get("synthetic") is True:
        return True
    for key in ["mode", "artifact_type", "source_type", "evidence_class"]:
        value = str(payload.get(key, "")).lower()
        if "synthetic" in value or "fixture" in value:
            return True
    return False


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = _normalize_key(str(key))
        full_key = f"{prefix}.{normalized_key}" if prefix else normalized_key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[normalized_key] = value
            flattened[full_key] = value
    return flattened


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _first_number(payload: dict[str, Any], keys: list[str]) -> float | None:
    lowered = {_normalize_key(str(key)): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(_normalize_key(key))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_feature_importances(payload: dict[str, Any]) -> dict[str, float]:
    candidates = [
        payload.get("feature_importance"),
        payload.get("feature_importances"),
        payload.get("feature_weights"),
        payload.get("weights"),
    ]
    for candidate in candidates:
        parsed = _parse_feature_importance(candidate)
        if parsed:
            return parsed
    return {}


def _parse_feature_importance(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        parsed = {}
        for key, item in value.items():
            try:
                parsed[str(key)] = float(item)
            except (TypeError, ValueError):
                continue
        return parsed
    if isinstance(value, list):
        parsed = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("feature") or item.get("name") or item.get("column")
            raw_weight = item.get("importance", item.get("weight", item.get("value")))
            if not name:
                continue
            try:
                parsed[str(name)] = float(raw_weight)
            except (TypeError, ValueError):
                continue
        return parsed
    return {}


def _find_key(value: Any, target_key: str) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if _normalize_key(str(key)) == _normalize_key(target_key):
                return item
            found = _find_key(item, target_key)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_key(item, target_key)
            if found is not None:
                return found
    return None


def _unique(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def _check(status: str, code: str, message: str) -> dict[str, str]:
    return {"status": status, "code": code, "message": message}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
