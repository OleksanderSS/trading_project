from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.prediction_target_semantics import (
    PredictionTargetSemanticsRegistry,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready, sha256_json

REQUIRED_LINEAGE_FIELDS = (
    "ticker",
    "model_context_id",
    "target_name",
    "model_type",
    "timeframe",
    "context_fingerprint",
    "selected_primary_model",
)


class PipelinePredictionReviewPacket:
    """Normalize Stage5 forecasts into supporting, per-context review data."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_prediction_review_packet_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        pipeline_result: dict[str, Any],
        *,
        requested_tickers: list[str] | None = None,
        requested_timeframes: list[str] | None = None,
        filter_to_requested_scope: bool = False,
        source_artifact_path: str | Path | None = None,
        sector_to_ticker_review_path: str | Path | None = None,
        save: bool = False,
    ) -> dict[str, Any]:
        raw_predictions, source_path = _prediction_results(
            pipeline_result
        )
        tickers = _normalized_upper_list(
            requested_tickers
            if requested_tickers is not None
            else pipeline_result.get("tickers")
        )
        timeframes = _normalized_lower_list(
            requested_timeframes
            if requested_timeframes is not None
            else (
                pipeline_result.get("timeframes")
                or pipeline_result.get("timeframe")
                )
        )
        source_context_count = len(raw_predictions)
        selected_predictions = raw_predictions
        if filter_to_requested_scope:
            selected_predictions = {
                key: value
                for key, value in raw_predictions.items()
                if isinstance(value, dict)
                and (
                    not tickers
                    or _upper_or_none(value.get("ticker")) in tickers
                )
                and (
                    not timeframes
                    or _lower_or_none(value.get("timeframe"))
                    in timeframes
                )
            }
        contexts = [
            _normalize_prediction(
                context_key=str(context_key),
                value=value,
                requested_tickers=tickers,
                requested_timeframes=timeframes,
            )
            for context_key, value in sorted(
                selected_predictions.items(),
                key=lambda item: str(item[0]),
            )
            if isinstance(value, dict)
        ]
        semantics_registry = PredictionTargetSemanticsRegistry()
        for item in contexts:
            semantics = semantics_registry.resolve(
                target_name=item.get("target_name"),
                timeframe=item.get("timeframe"),
                prediction_as_of=_mapping(
                    item.get("prediction")
                ).get("as_of"),
                model_output_contract=_mapping(
                    item.get("model_output_contract")
                ),
            )
            item["target_semantics"] = semantics
            if semantics.get("status") != "target_semantics_ready":
                item["review_issues"].append(
                    "target_semantics_incomplete"
                )
            if not _mapping(
                semantics.get("calibration")
            ).get("model_output_scale_known"):
                item["review_issues"].append(
                    "model_output_contract_incomplete"
                )
        sector_context_review = _sector_context_review_binding(
            sector_to_ticker_review_path
        )
        for item in contexts:
            item["supporting_sector_ticker_context"] = (
                _supporting_sector_ticker_context(
                    item,
                    sector_context_review,
                )
            )
        _mark_duplicate_lineage(contexts)
        complete_count = sum(
            item["lineage_status"] == "complete"
            and not item["review_issues"]
            for item in contexts
        )
        issue_counts = Counter(
            issue
            for item in contexts
            for issue in item.get("review_issues", [])
        )
        missing_lineage_counts = Counter(
            field
            for item in contexts
            for field in item.get("missing_lineage_fields", [])
        )
        if not raw_predictions:
            status = "stage5_predictions_not_available"
        elif not contexts:
            status = "stage5_predictions_invalid"
        elif complete_count == len(contexts):
            status = "stage5_prediction_review_ready"
        else:
            status = "stage5_prediction_review_partial"
        source_artifact = _source_artifact(
            source_artifact_path
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "pipeline_prediction_review_packet",
            "schema_version": "dean_stage5_prediction_review_v1",
            "status": status,
            "source_path": source_path,
            "source_contract": "pipeline_stage5_prediction_results",
            "source_artifact": source_artifact,
            "sector_context_review": sector_context_review,
            "sector_context_overlay_summary": (
                _sector_context_overlay_summary(contexts)
            ),
            "requested_tickers": tickers,
            "requested_timeframes": timeframes,
            "filter_to_requested_scope": bool(
                filter_to_requested_scope
            ),
            "source_context_count": source_context_count,
            "excluded_by_scope_count": (
                source_context_count - len(selected_predictions)
            ),
            "context_count": len(contexts),
            "complete_context_count": complete_count,
            "review_issue_counts": dict(sorted(issue_counts.items())),
            "missing_lineage_field_counts": dict(
                sorted(missing_lineage_counts.items())
            ),
            "contexts": contexts,
            "packet_fingerprint": sha256_json(
                {
                    "requested_tickers": tickers,
                    "requested_timeframes": timeframes,
                    "source_artifact": source_artifact,
                    "sector_context_review": (
                        sector_context_review
                    ),
                    "contexts": contexts,
                }
            ),
            "evidence_class": (
                "supporting_prediction_review_not_locked_evidence"
            ),
            "target_semantics_contract": {
                "schema_version": (
                    "dean_prediction_target_semantics_v1"
                ),
                "config_path": str(
                    semantics_registry.config_path
                ),
                "config_sha256": (
                    semantics_registry.config_sha256
                ),
                "directional_inference_allowed": False,
            },
            "safety": {
                "supporting_review_only": True,
                "is_model_evaluation": False,
                "is_realized_outcome": False,
                "can_clear_locked_evidence": False,
                "can_promote_model": False,
                "can_write_learning_memory": False,
                "can_create_recommendation": False,
                "decision_influence": False,
                "sector_context_decision_influence": False,
                "ticker_evidence_decision_influence": False,
                "can_trade": False,
            },
        }
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_prediction_review_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)


def _sector_context_review_binding(
    value: str | Path | None,
) -> dict[str, Any]:
    if value is None:
        return {
            "path": None,
            "available": False,
            "sha256": None,
            "status": "not_attached",
            "sector_stance": None,
            "ticker_review_map": [],
            "decision_influence": False,
        }
    path = Path(value)
    if not path.is_file():
        raise ValueError(
            f"sector-to-ticker review artifact missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("mode") != (
        "sector_to_ticker_review_packet"
    ):
        raise ValueError(
            "unsupported sector-to-ticker review artifact"
        )
    summary = _mapping(payload.get("summary"))
    if summary.get("packet_status") not in {
        "review_ready",
        "review_ready_with_limitations",
    }:
        raise ValueError(
            "sector-to-ticker review is not review-ready"
        )
    if (
        summary.get("can_create_ticker_forecast") is not False
        or summary.get("can_write_learning_memory") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError(
            "sector-to-ticker review safety boundary invalid"
        )
    ticker_map = payload.get("ticker_review_map")
    if not isinstance(ticker_map, list):
        raise ValueError(
            "sector-to-ticker ticker_review_map missing"
        )
    for item in ticker_map:
        if not isinstance(item, dict):
            continue
        reasoning = _mapping(
            _mapping(item.get("sector_context")).get(
                "verified_reasoning"
            )
        )
        if reasoning.get("available") is True and (
            reasoning.get("runtime_hash_bound") is not True
            or int(reasoning.get("directional_ticker_event_count") or 0)
            != 0
        ):
            raise ValueError(
                "sector reasoning context violates hash/ticker boundary"
            )
    return {
        "path": str(path),
        "available": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_id": payload.get("run_id"),
        "status": summary.get("packet_status"),
        "sector": summary.get("sector"),
        "domain_profile": summary.get("domain_profile"),
        "sector_stance": summary.get("sector_stance"),
        "ticker_review_map": ticker_map,
        "decision_influence": False,
        "can_create_ticker_forecast": False,
    }


def _supporting_sector_ticker_context(
    prediction_context: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    if binding.get("available") is not True:
        return {
            "status": "not_attached",
            "ticker": prediction_context.get("ticker"),
            "decision_influence": False,
            "can_change_prediction": False,
        }
    ticker = str(prediction_context.get("ticker") or "").upper()
    matches = [
        item
        for item in binding.get("ticker_review_map", [])
        if str(item.get("ticker") or "").upper() == ticker
    ]
    if len(matches) != 1:
        return {
            "status": (
                "ticker_context_missing"
                if not matches
                else "ticker_context_ambiguous"
            ),
            "ticker": ticker,
            "sector_stance": binding.get("sector_stance"),
            "decision_influence": False,
            "can_change_prediction": False,
        }
    review = matches[0]
    exact_cases = review.get("exact_pipeline_contexts") or []
    aligned_cases = [
        case
        for case in exact_cases
        if _pipeline_case_matches_prediction(
            case,
            prediction_context,
        )
    ]
    ticker_evidence = _mapping(
        review.get("ticker_specific_evidence")
    )
    sector_reasoning = _mapping(
        _mapping(review.get("sector_context")).get(
            "verified_reasoning"
        )
    )
    feature_timeframe_audit = _mapping(
        review.get("feature_timeframe_audit")
    )
    flags = []
    if ticker_evidence.get("eligible_record_count", 0):
        flags.append(
            "ticker_company_mechanism_evidence_supporting_only"
        )
    if sector_reasoning.get("available") is True:
        flags.append("verified_sector_reasoning_supporting_only")
    if feature_timeframe_audit.get("status") in {
        "timeframe_cadence_mismatch",
        "timeframe_cadence_ambiguous",
    }:
        flags.append(
            "candidate_feature_timeframe_cadence_mismatch"
        )
    if (
        feature_timeframe_audit.get(
            "can_assert_feature_parentage"
        )
        is False
    ):
        flags.append(
            "legacy_stage5_feature_parentage_unverified"
        )
    if exact_cases and not aligned_cases:
        flags.append(
            "attached_pipeline_cases_do_not_match_prediction_identity"
        )
    for case in aligned_cases:
        if case.get("case_classification") == (
            "negative_evaluation_block_case"
        ):
            flags.append(
                "negative_pipeline_evaluation_case_aligned"
            )
    return {
        "status": "supporting_context_attached",
        "ticker": ticker,
        "sector": binding.get("sector"),
        "domain_profile": binding.get("domain_profile"),
        "sector_stance": binding.get("sector_stance"),
        "sector_reasoning_context": sector_reasoning,
        "feature_timeframe_audit": feature_timeframe_audit,
        "ticker_review_status": review.get("review_status"),
        "allowed_use": review.get("allowed_use"),
        "ticker_evidence_status": ticker_evidence.get("status"),
        "ticker_evidence_eligible_record_count": (
            ticker_evidence.get("eligible_record_count", 0)
        ),
        "ticker_evidence_corroborated_lane_count": (
            ticker_evidence.get("corroborated_lane_count", 0)
        ),
        "exact_pipeline_case_count": len(exact_cases),
        "aligned_pipeline_case_count": len(aligned_cases),
        "aligned_pipeline_cases": aligned_cases,
        "required_next_inputs": review.get(
            "required_next_inputs", []
        ),
        "context_flags": sorted(set(flags)),
        "decision_influence": False,
        "can_change_prediction": False,
        "can_fill_missing_lineage": False,
        "can_clear_model_evaluation": False,
        "can_create_ticker_forecast": False,
    }


def _pipeline_case_matches_prediction(
    case: dict[str, Any],
    prediction: dict[str, Any],
) -> bool:
    return all(
        (
            str(case.get(case_field) or "").lower()
            == str(prediction.get(prediction_field) or "").lower()
        )
        for case_field, prediction_field in (
            ("ticker", "ticker"),
            ("model", "selected_primary_model"),
            ("target_name", "target_name"),
            ("timeframe", "timeframe"),
            ("context_fingerprint", "context_fingerprint"),
        )
    )


def _sector_context_overlay_summary(
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    overlays = [
        _mapping(item.get("supporting_sector_ticker_context"))
        for item in contexts
    ]
    return {
        "context_count": len(overlays),
        "attached_count": sum(
            item.get("status") == "supporting_context_attached"
            for item in overlays
        ),
        "ticker_evidence_context_count": sum(
            int(
                item.get(
                    "ticker_evidence_eligible_record_count", 0
                )
                or 0
            )
            > 0
            for item in overlays
        ),
        "aligned_pipeline_case_count": sum(
            int(item.get("aligned_pipeline_case_count", 0) or 0)
            for item in overlays
        ),
        "decision_influence": False,
        "can_change_prediction": False,
    }


def _prediction_results(
    pipeline_result: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    containers = [
        ("results.prediction_results", pipeline_result.get("results")),
        ("prediction_results", pipeline_result),
        ("summary.prediction_results", pipeline_result.get("summary")),
    ]
    for source_path, container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get("prediction_results")
        if isinstance(value, dict):
            return value, source_path
    return {}, None


def _normalize_prediction(
    *,
    context_key: str,
    value: dict[str, Any],
    requested_tickers: list[str],
    requested_timeframes: list[str],
) -> dict[str, Any]:
    lineage = {
        "ticker": _upper_or_none(value.get("ticker")),
        "model_context_id": _text_or_none(
            value.get("model_context_id")
        ),
        "target_name": _text_or_none(value.get("target_name")),
        "model_type": _text_or_none(value.get("model_type")),
        "timeframe": _lower_or_none(value.get("timeframe")),
        "context_fingerprint": _text_or_none(
            value.get("context_fingerprint")
        ),
        "selected_primary_model": _text_or_none(
            value.get("selected_primary_model")
        ),
    }
    missing = [
        field
        for field in REQUIRED_LINEAGE_FIELDS
        if not lineage.get(field)
    ]
    issues: list[str] = []
    if (
        requested_tickers
        and lineage["ticker"] not in requested_tickers
    ):
        issues.append("ticker_outside_requested_context")
    if (
        requested_timeframes
        and lineage["timeframe"] not in requested_timeframes
    ):
        issues.append("timeframe_outside_requested_context")
    confidence = _unit_interval_or_none(value.get("confidence"))
    anomaly_score = _unit_interval_or_none(
        value.get("anomaly_score")
    )
    if value.get("confidence") is not None and confidence is None:
        issues.append("invalid_confidence")
    if (
        value.get("anomaly_score") is not None
        and anomaly_score is None
    ):
        issues.append("invalid_anomaly_score")
    prediction_value, prediction_shape = _scalar_summary(
        value.get("predictions")
    )
    raw_forecast, raw_shape = _scalar_summary(
        value.get("raw_forecast")
    )
    if prediction_value is None and value.get("predictions") is not None:
        issues.append("prediction_not_single_scalar")
    prediction_as_of = _text_or_none(value.get("timestamp"))
    if prediction_as_of is None:
        issues.append("prediction_as_of_missing")
    elif not _timezone_aware_timestamp(prediction_as_of):
        issues.append("prediction_as_of_not_timezone_aware")
    context_fingerprint = lineage.get("context_fingerprint")
    if context_fingerprint and _placeholder_context_fingerprint(
        context_fingerprint
    ):
        issues.append("context_fingerprint_placeholder_or_pattern")
    timeframe_lineage = _mapping(value.get("timeframe_lineage"))
    timeframe_lineage_status = timeframe_lineage.get("status")
    if timeframe_lineage_status in {
        "timeframe_cadence_mismatch",
        "timeframe_cadence_ambiguous",
    }:
        issues.append(str(timeframe_lineage_status))
    return {
        "context_key": context_key,
        **lineage,
        "lineage_status": "complete" if not missing else "incomplete",
        "missing_lineage_fields": missing,
        "review_issues": issues,
        "timeframe_lineage": timeframe_lineage,
        "prediction": {
            "value": prediction_value,
            "shape": prediction_shape,
            "raw_forecast": raw_forecast,
            "raw_forecast_shape": raw_shape,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "last_price": _finite_float_or_none(
                value.get("last_price")
            ),
            "as_of": prediction_as_of,
        },
        "model_contribution_count": _safe_len(
            value.get("predictions_by_model")
        ),
        "model_output_contract": _mapping(
            value.get("model_output_contract")
        ),
        "supporting_review_only": True,
        "is_model_evaluation": False,
        "is_realized_outcome": False,
        "decision_influence": False,
        "can_promote_model": False,
        "can_trade": False,
    }


def _mark_duplicate_lineage(
    contexts: list[dict[str, Any]],
) -> None:
    identities: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in contexts:
        if item["lineage_status"] != "complete":
            continue
        identity = tuple(
            item.get(field) for field in REQUIRED_LINEAGE_FIELDS
        )
        identities.setdefault(identity, []).append(item)
    for duplicates in identities.values():
        if len(duplicates) < 2:
            continue
        for item in duplicates:
            item["review_issues"].append(
                "duplicate_prediction_lineage"
            )


def _scalar_summary(value: Any) -> tuple[float | None, dict[str, Any]]:
    if value is None:
        return None, {"kind": "missing", "count": 0}
    scalar = _finite_float_or_none(value)
    if scalar is not None:
        return scalar, {"kind": "scalar", "count": 1}
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            scalar = _finite_float_or_none(value[0])
            if scalar is not None:
                return scalar, {"kind": "single_item_sequence", "count": 1}
        return None, {"kind": "sequence", "count": len(value)}
    shape = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    if size == 1:
        try:
            scalar = _finite_float_or_none(value.item())
        except Exception:
            scalar = None
        if scalar is not None:
            return scalar, {
                "kind": "single_item_array",
                "count": 1,
                "shape": list(shape) if shape is not None else None,
            }
    return None, {
        "kind": type(value).__name__,
        "count": int(size) if isinstance(size, int) else None,
        "shape": list(shape) if shape is not None else None,
    }


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _unit_interval_or_none(value: Any) -> float | None:
    parsed = _finite_float_or_none(value)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _safe_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _source_artifact(
    value: str | Path | None,
) -> dict[str, Any]:
    if value is None:
        return {
            "path": None,
            "available": False,
            "sha256": None,
            "immutable_binding_ready": False,
        }
    path = Path(value)
    if not path.is_file():
        return {
            "path": str(path),
            "available": False,
            "sha256": None,
            "immutable_binding_ready": False,
        }
    return {
        "path": str(path),
        "available": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "immutable_binding_ready": True,
    }


def _timezone_aware_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _placeholder_context_fingerprint(value: str) -> bool:
    normalized = str(value).strip().lower()
    return (
        normalized
        in {
            "default",
            "normal",
            "unknown",
            "unknown_context",
            "batch_training",
        }
        or normalized.startswith("legacy_")
    )


def _normalized_upper_list(value: Any) -> list[str]:
    return sorted(
        {
            str(item).strip().upper()
            for item in _as_list(value)
            if str(item).strip()
        }
    )


def _normalized_lower_list(value: Any) -> list[str]:
    return sorted(
        {
            str(item).strip().lower()
            for item in _as_list(value)
            if str(item).strip()
        }
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _upper_or_none(value: Any) -> str | None:
    text = _text_or_none(value)
    return text.upper() if text else None


def _lower_or_none(value: Any) -> str | None:
    text = _text_or_none(value)
    return text.lower() if text else None


def render_pipeline_prediction_review_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# DEAN-OS Stage5 Prediction Review Packet",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Contexts: {payload.get('context_count')}",
        f"- Complete contexts: {payload.get('complete_context_count')}",
        "- Evidence class: "
        f"`{payload.get('evidence_class')}`",
        "",
        "| Context | Ticker | Timeframe | Target | Model | Confidence | Anomaly | Lineage | Issues |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in payload.get("contexts", []):
        prediction = item.get("prediction", {})
        lines.append(
            "| {key} | {ticker} | {timeframe} | {target} | "
            "{model} | {confidence} | {anomaly} | {lineage} | "
            "{issues} |".format(
                key=item.get("context_key"),
                ticker=item.get("ticker"),
                timeframe=item.get("timeframe"),
                target=item.get("target_name"),
                model=item.get("selected_primary_model"),
                confidence=prediction.get("confidence"),
                anomaly=prediction.get("anomaly_score"),
                lineage=item.get("lineage_status"),
                issues=", ".join(item.get("review_issues", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Supporting Sector/Ticker Context",
            "",
            f"- Attached contexts: {payload.get('sector_context_overlay_summary', {}).get('attached_count')}",
            f"- Contexts with eligible company evidence: {payload.get('sector_context_overlay_summary', {}).get('ticker_evidence_context_count')}",
            f"- Exact aligned pipeline cases: {payload.get('sector_context_overlay_summary', {}).get('aligned_pipeline_case_count')}",
            "- Decision influence: False",
            "",
            "Predictions are supporting review context only. This packet "
            "is not model evaluation, a realized outcome, locked evidence, "
            "a recommendation, or trading authority.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"pipeline_prediction_review_packet_{stamp}"
