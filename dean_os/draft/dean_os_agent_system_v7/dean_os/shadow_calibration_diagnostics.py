from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.shadow_calibration_case_index import (
    validate_shadow_calibration_case,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.shadow_component_case_producer import (
    validate_component_case,
)
from dean_os.utils import json_ready, sha256_json

COMPONENTS = (
    "prediction",
    "regime",
    "specialist",
    "context_synthesis",
)


class ShadowCalibrationDiagnostics:
    """Compute review-only metrics from aligned, validated outcome episodes."""

    def __init__(
        self,
        *,
        case_index_path: str | Path,
        policy_path: str | Path = (
            "dean_os/config/shadow_calibration_policy.yaml"
        ),
        output_dir: str | Path = (
            "reports/dean_os/shadow_calibration_diagnostics_current"
        ),
    ):
        self.case_index_path = Path(case_index_path)
        self.policy_path = Path(policy_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        index_state = _json_state(self.case_index_path)
        policy_state = _yaml_state(self.policy_path)
        blockers = []
        if not index_state["available"]:
            blockers.append("case_index_unavailable")
        if not policy_state["available"]:
            blockers.append("diagnostic_policy_unavailable")

        records: list[dict[str, Any]] = []
        invalid_record_count = 0
        if not blockers:
            index_payload = index_state["payload"]
            if (
                index_payload.get("mode")
                != "shadow_calibration_case_index"
                or index_payload.get("schema_version")
                != "dean_shadow_calibration_case_index_v1"
            ):
                blockers.append("case_index_schema_mismatch")
            else:
                for record in index_payload.get("records", []):
                    issues = validate_shadow_calibration_case(record)
                    if (
                        isinstance(record, dict)
                        and record.get("component") != "prediction"
                    ):
                        issues.extend(validate_component_case(record))
                    if issues:
                        invalid_record_count += 1
                    else:
                        records.append(record)
                if invalid_record_count:
                    blockers.append(
                        "case_index_contains_invalid_records"
                    )

        policy = policy_state.get("payload", {})
        requirements = _mapping(policy.get("case_requirements"))
        minimum = int(
            requirements.get("diagnostic_min_cases_per_context", 30)
        )
        context_sets, alignment_issues = _aligned_context_sets(
            records
        )
        ready_contexts = {
            key: value
            for key, value in context_sets.items()
            if value["common_episode_count"] >= minimum
            and not value["issues"]
        }
        if not ready_contexts:
            blockers.append(
                "no_exact_context_with_minimum_aligned_episodes"
            )
        blockers.extend(alignment_issues)
        blockers = sorted(set(blockers))

        diagnostics = []
        for context_key, context_data in sorted(
            ready_contexts.items()
        ):
            diagnostics.append(
                _context_diagnostics(
                    context_key=context_key,
                    context_data=context_data,
                    policy=policy,
                )
            )

        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "shadow_calibration_diagnostics",
            "schema_version": (
                "dean_shadow_calibration_diagnostics_v1"
            ),
            "status": (
                "shadow_diagnostics_ready_for_review"
                if diagnostics and not blockers
                else "shadow_diagnostics_blocked"
            ),
            "source_inventory": {
                "case_index": _public_state(index_state),
                "policy": _public_state(policy_state),
            },
            "policy_snapshot": {
                "diagnostic_min_cases_per_context": minimum,
                "component_metrics": policy.get(
                    "component_metrics",
                    {},
                ),
                "safety_thresholds": policy.get(
                    "safety_thresholds",
                    {},
                ),
            },
            "valid_record_count": len(records),
            "invalid_record_count": invalid_record_count,
            "context_coverage": {
                key: {
                    "component_counts": value["component_counts"],
                    "common_episode_count": value[
                        "common_episode_count"
                    ],
                    "issues": value["issues"],
                }
                for key, value in context_sets.items()
            },
            "diagnostic_context_count": len(diagnostics),
            "diagnostics": diagnostics,
            "blocking_gaps": blockers,
            "template_alignment": {
                "classification_probability_metrics": (
                    "only_when_positive_class_probability_is_explicit"
                ),
                "classification_label_metrics": (
                    "only_from_verified_raw_class_label"
                ),
                "numeric_metadata_completeness": "required",
                "time_leakage_rate": "required_zero",
                "unsafe_output_rate": "required_zero",
                "human_review_disagreement_rate": (
                    "unavailable_until_explicit_human_labels"
                ),
            },
            "safety": {
                "review_only": True,
                "diagnostics_computed": bool(diagnostics),
                "calibration_executed": False,
                "consensus_weight_eligible": False,
                "automatic_weight_change_allowed": False,
                "decision_influence": False,
                "can_write_learning_memory": False,
                "can_write_production_config": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
        }
        payload["diagnostics_fingerprint"] = sha256_json({
            "case_index_sha256": index_state.get("sha256"),
            "policy_sha256": policy_state.get("sha256"),
            "context_coverage": payload["context_coverage"],
            "diagnostics": diagnostics,
            "blocking_gaps": blockers,
        })
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_shadow_diagnostics_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def _aligned_context_sets(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    grouped: dict[
        str,
        dict[str, dict[str, list[dict[str, Any]]]],
    ] = {}
    for record in records:
        context_key = _context_key(record)
        component = str(record.get("component") or "")
        episode_id = _episode_id(record)
        if component not in COMPONENTS or not episode_id:
            continue
        grouped.setdefault(
            context_key,
            {name: {} for name in COMPONENTS},
        )[component].setdefault(episode_id, []).append(record)

    result = {}
    global_issues = []
    for context_key, component_map in sorted(grouped.items()):
        issues = []
        for component, episode_map in component_map.items():
            duplicates = [
                episode_id
                for episode_id, values in episode_map.items()
                if len(values) != 1
            ]
            if duplicates:
                issues.append(
                    f"duplicate_{component}_records_per_episode"
                )
        common_ids = set.intersection(
            *[
                set(episode_map)
                for episode_map in component_map.values()
            ]
        )
        aligned = {
            episode_id: {
                component: component_map[component][episode_id][0]
                for component in COMPONENTS
            }
            for episode_id in sorted(common_ids)
            if all(
                len(component_map[component][episode_id]) == 1
                for component in COMPONENTS
            )
        }
        result[context_key] = {
            "component_counts": {
                component: sum(
                    len(values)
                    for values in episode_map.values()
                )
                for component, episode_map in component_map.items()
            },
            "common_episode_count": len(aligned),
            "episodes": aligned,
            "issues": sorted(set(issues)),
        }
        global_issues.extend(
            f"{context_key}:{issue}" for issue in issues
        )
    return result, sorted(set(global_issues))


def _context_diagnostics(
    *,
    context_key: str,
    context_data: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    episodes = list(context_data["episodes"].values())
    component_records = {
        component: [episode[component] for episode in episodes]
        for component in COMPONENTS
    }
    return {
        "context_key": context_key,
        "episode_count": len(episodes),
        "prediction": _prediction_metrics(
            component_records["prediction"]
        ),
        "regime": _regime_metrics(
            component_records["regime"]
        ),
        "specialist": _specialist_metrics(
            component_records["specialist"]
        ),
        "context_synthesis": _synthesis_metrics(
            component_records["context_synthesis"]
        ),
        "safety_metrics": _safety_metrics(
            [
                record
                for records in component_records.values()
                for record in records
            ]
        ),
        "policy_metric_contract": policy.get(
            "component_metrics",
            {},
        ),
        "diagnostic_only": True,
        "consensus_weight_eligible": False,
        "can_trade": False,
    }


def _prediction_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    target_types = {
        str(
            _mapping(record.get("prediction")).get("target_type")
            or _mapping(record.get("realization")).get("target_type")
            or ""
        )
        for record in records
    }
    target_types.discard("")
    available = {}
    unavailable = {}
    if len(target_types) != 1:
        unavailable["all_prediction_metrics"] = (
            "target_type_missing_or_mixed"
        )
        return {
            "target_types": sorted(target_types),
            "available": available,
            "unavailable": unavailable,
        }

    target_type = next(iter(target_types))
    if target_type.startswith("classification"):
        actual = [
            _integer_or_none(
                _mapping(record.get("realization")).get(
                    "realized_target"
                )
            )
            for record in records
        ]
        raw_predictions = [
            _integer_or_none(
                _mapping(record.get("prediction")).get("raw_value")
            )
            for record in records
        ]
        raw_scales = {
            _mapping(record.get("prediction")).get(
                "raw_output_scale"
            )
            for record in records
        }
        if (
            all(value is not None for value in actual)
            and all(value is not None for value in raw_predictions)
            and raw_scales == {"class_label_from_predict"}
        ):
            available["classification_label"] = (
                _classification_label_metrics(
                    [int(value) for value in raw_predictions],
                    [int(value) for value in actual],
                )
            )
        else:
            unavailable["classification_label"] = (
                "verified_raw_class_labels_not_available"
            )

        probabilities = [
            _finite_float(
                _mapping(record.get("prediction")).get("value")
            )
            for record in records
        ]
        probability_flags = [
            _mapping(record.get("prediction")).get(
                "positive_class_probability"
            )
            is True
            for record in records
        ]
        if (
            all(probability_flags)
            and all(
                value is not None and 0.0 <= value <= 1.0
                for value in probabilities
            )
            and all(value is not None for value in actual)
        ):
            available["classification_probability"] = (
                _classification_probability_metrics(
                    [float(value) for value in probabilities],
                    [int(value) for value in actual],
                )
            )
        else:
            unavailable["classification_probability"] = (
                "final_output_is_not_validated_positive_class_probability"
            )
        unavailable["adjusted_score_directional_accuracy"] = (
            "no_reviewed_score_to_class_threshold"
        )
    elif target_type == "regression":
        predicted = [
            _finite_float(
                _mapping(record.get("prediction")).get("value")
            )
            for record in records
        ]
        actual = [
            _finite_float(
                _mapping(record.get("realization")).get(
                    "realized_target"
                )
            )
            for record in records
        ]
        if all(value is not None for value in predicted + actual):
            available["regression"] = _regression_metrics(
                [float(value) for value in predicted],
                [float(value) for value in actual],
            )
        else:
            unavailable["regression"] = (
                "finite_prediction_or_realized_target_missing"
            )
    else:
        unavailable["all_prediction_metrics"] = (
            f"unsupported_target_type:{target_type}"
        )
    return {
        "target_types": sorted(target_types),
        "available": available,
        "unavailable": unavailable,
    }


def _classification_label_metrics(
    predicted: list[int],
    actual: list[int],
) -> dict[str, float | int]:
    classes = sorted(set(actual))
    recalls = []
    for label in classes:
        positives = sum(value == label for value in actual)
        correct = sum(
            observed == label and forecast == label
            for forecast, observed in zip(predicted, actual, strict=True)
        )
        recalls.append(correct / positives if positives else 0.0)
    true_positive = sum(
        forecast == 1 and observed == 1
        for forecast, observed in zip(predicted, actual, strict=True)
    )
    predicted_positive = sum(value == 1 for value in predicted)
    actual_positive = sum(value == 1 for value in actual)
    return {
        "sample_count": len(actual),
        "accuracy": _mean(
            [
                float(forecast == observed)
                for forecast, observed in zip(predicted, actual, strict=True)
            ]
        ),
        "balanced_accuracy": _mean(recalls),
        "precision": (
            true_positive / predicted_positive
            if predicted_positive
            else 0.0
        ),
        "recall": (
            true_positive / actual_positive
            if actual_positive
            else 0.0
        ),
    }


def _classification_probability_metrics(
    probabilities: list[float],
    actual: list[int],
) -> dict[str, float | int]:
    epsilon = 1e-12
    brier = _mean([
        (probability - observed) ** 2
        for probability, observed in zip(probabilities, actual, strict=True)
    ])
    log_loss = -_mean([
        observed * math.log(max(probability, epsilon))
        + (1 - observed)
        * math.log(max(1.0 - probability, epsilon))
        for probability, observed in zip(probabilities, actual, strict=True)
    ])
    return {
        "sample_count": len(actual),
        "brier_score": brier,
        "log_loss": log_loss,
        "calibration_error": abs(
            _mean(probabilities) - _mean(actual)
        ),
    }


def _regression_metrics(
    predicted: list[float],
    actual: list[float],
) -> dict[str, float | int]:
    errors = [
        forecast - observed
        for forecast, observed in zip(predicted, actual, strict=True)
    ]
    return {
        "sample_count": len(actual),
        "mae": _mean([abs(value) for value in errors]),
        "rmse": math.sqrt(_mean([value * value for value in errors])),
        "directional_accuracy": _mean([
            float(_sign(forecast) == _sign(observed))
            for forecast, observed in zip(predicted, actual, strict=True)
        ]),
    }


def _regime_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        regime = str(
            _mapping(record.get("assessment")).get("regime")
            or "UNKNOWN"
        )
        realized_return = _finite_float(
            _mapping(record.get("realization")).get(
                "realized_return"
            )
        )
        if regime != "UNKNOWN" and realized_return is not None:
            grouped.setdefault(regime, []).append(realized_return)
    conditional = {
        regime: {
            "sample_count": len(values),
            "mean_forward_return": _mean(values),
            "median_forward_return": statistics.median(values),
            "cross_episode_return_std": (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0
            ),
        }
        for regime, values in sorted(grouped.items())
    }
    return {
        "available": {
            "conditional_forward_return": conditional,
            "conditional_volatility": {
                regime: item["cross_episode_return_std"]
                for regime, item in conditional.items()
            },
        },
        "unavailable": {
            "conditional_drawdown": (
                "outcome_path_inside_realization_window_not_stored"
            ),
            "transition_stability": (
                "ordered_non_overlapping_regime_sequence_not_proven"
            ),
        },
    }


def _specialist_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    assessments = [
        _mapping(record.get("assessment")) for record in records
    ]
    return {
        "available": {
            "accepted_case_count": len(records),
            "point_in_time_valid_rate": _mean([
                float(
                    item.get("point_in_time_status")
                    == "point_in_time_compatible"
                )
                for item in assessments
            ]),
            "timeframe_alignment_rate": _mean([
                float(
                    item.get("timeframe_alignment_status")
                    == "aligned"
                )
                for item in assessments
            ]),
            "manual_review_complete_rate": _mean([
                float(item.get("manual_review_required") is False)
                for item in assessments
            ]),
        },
        "unavailable": {
            "direct_ticker_scope_precision": (
                "specialist_assessment_has_no_directional_hypothesis_label"
            ),
            "human_review_disagreement_rate": (
                "explicit_human_disagreement_labels_not_supplied"
            ),
        },
        "selection_note": (
            "Rates describe accepted gated cases, not all submitted "
            "specialist assessments."
        ),
    }


def _synthesis_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    assessments = [
        _mapping(record.get("assessment")) for record in records
    ]
    return {
        "available": {
            "accepted_case_count": len(records),
            "context_match_rate": 1.0 if records else 0.0,
            "freshness_compatibility_rate": _mean([
                float(item.get("freshness_status") == "compatible")
                for item in assessments
            ]),
        },
        "unavailable": {
            "conflict_precision": (
                "no_reviewed_conflict_truth_label"
            ),
            "human_review_disagreement_rate": (
                "explicit_human_disagreement_labels_not_supplied"
            ),
        },
        "selection_note": (
            "Context match is guaranteed by the case gate and is not "
            "an ungated production-rate estimate."
        ),
    }


def _safety_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = {
        "unsafe_output_rate": "unsafe_output_detected",
        "time_leakage_rate": "time_leakage_detected",
        "sector_to_ticker_leakage_rate": (
            "sector_to_ticker_leakage_detected"
        ),
        "future_evidence_rate": "future_evidence_used",
        "context_mismatch_rate": None,
    }
    metrics = {}
    for metric, field in fields.items():
        if field is None:
            count = sum(
                _mapping(record.get("safety")).get(
                    "exact_context_match"
                )
                is not True
                for record in records
            )
        else:
            count = sum(
                _mapping(record.get("safety")).get(field) is True
                for record in records
            )
        metrics[metric] = {
            "count": count,
            "sample_count": len(records),
            "rate": count / len(records) if records else 0.0,
        }
    return metrics


def render_shadow_diagnostics_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# DEAN-OS Shadow Calibration Diagnostics",
        "",
        f"- Status: `{payload.get('status')}`",
        (
            "- Diagnostic contexts: "
            f"{payload.get('diagnostic_context_count')}"
        ),
        f"- Valid records: {payload.get('valid_record_count')}",
        f"- Invalid records: {payload.get('invalid_record_count')}",
        "- Consensus weight eligible: False",
        "- Can trade: False",
    ]
    if payload.get("blocking_gaps"):
        lines.extend(["", "## Blocking Gaps", ""])
        lines.extend(
            f"- `{item}`" for item in payload["blocking_gaps"]
        )
    for item in payload.get("diagnostics", []):
        lines.extend([
            "",
            f"## {item.get('context_key')}",
            "",
            f"- Episodes: {item.get('episode_count')}",
            "- Diagnostic only: True",
        ])
        unavailable = _mapping(
            _mapping(item.get("prediction")).get("unavailable")
        )
        if unavailable:
            lines.append(
                "- Unavailable prediction metrics: "
                + ", ".join(sorted(unavailable))
            )
    lines.extend([
        "",
        "Diagnostics never change agent weights, learning memory, "
        "production config, recommendations, or trades.",
    ])
    return "\n".join(lines) + "\n"


def _context_key(record: dict[str, Any]) -> str:
    identity = _mapping(record.get("identity"))
    return "|".join(
        str(identity.get(field) or "")
        for field in (
            "ticker",
            "timeframe",
            "target_name",
            "context_fingerprint",
        )
    )


def _episode_id(record: dict[str, Any]) -> str | None:
    value = (
        record.get("case_id")
        if record.get("component") == "prediction"
        else record.get("base_prediction_case_id")
    )
    return str(value) if value else None


def _json_state(path: Path) -> dict[str, Any]:
    return _structured_state(path, loader="json")


def _yaml_state(path: Path) -> dict[str, Any]:
    return _structured_state(path, loader="yaml")


def _structured_state(
    path: Path,
    *,
    loader: str,
) -> dict[str, Any]:
    state = {
        "path": str(path),
        "available": False,
        "sha256": None,
        "payload": {},
    }
    if not path.is_file():
        return state
    try:
        text = path.read_text(encoding="utf-8")
        raw = (
            json.loads(text)
            if loader == "json"
            else yaml.safe_load(text)
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return state
    if not isinstance(raw, dict):
        return state
    state.update({
        "available": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload": raw,
    })
    return state


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key != "payload"
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer_or_none(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or parsed != int(parsed):
        return None
    return int(parsed)


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"shadow_calibration_diagnostics_{stamp}"
