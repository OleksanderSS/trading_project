from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready, sha256_json


class ShadowCalibrationCaseIndexBuilder:
    """Bind reviewed Stage5 predictions to exact, immutable outcome rows."""

    def __init__(
        self,
        *,
        prediction_review_path: str | Path,
        outcome_source_path: str | Path,
        output_dir: str | Path = (
            "reports/dean_os/shadow_calibration_case_index_current"
        ),
    ):
        self.prediction_review_path = Path(prediction_review_path)
        self.outcome_source_path = Path(outcome_source_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        prediction_state = _json_artifact_state(
            self.prediction_review_path
        )
        outcome_state, outcome_rows = _outcome_source_state(
            self.outcome_source_path
        )
        source_issues = []
        if not prediction_state["available"]:
            source_issues.append("prediction_review_unavailable")
        if not outcome_state["available"]:
            source_issues.extend(outcome_state["issues"])

        records: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        if not source_issues:
            payload = prediction_state["payload"]
            packet_issues = _prediction_packet_issues(payload)
            if packet_issues:
                source_issues.extend(packet_issues)
            else:
                for context in payload.get("contexts", []):
                    if not isinstance(context, dict):
                        continue
                    record, issues = _build_prediction_case(
                        context=context,
                        prediction_review_state=prediction_state,
                        outcome_state=outcome_state,
                        outcome_rows=outcome_rows,
                    )
                    if issues:
                        rejected.append({
                            "context_key": context.get("context_key"),
                            "issues": issues,
                        })
                    elif record is not None:
                        records.append(record)

        records = _deduplicated_records(records, rejected)
        status = (
            "shadow_calibration_case_index_ready"
            if records and not source_issues
            else "shadow_calibration_case_index_blocked"
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "shadow_calibration_case_index",
            "schema_version": "dean_shadow_calibration_case_index_v1",
            "status": status,
            "source_inventory": {
                "prediction_review": _public_state(
                    prediction_state
                ),
                "outcome_source": _public_state(outcome_state),
            },
            "source_issues": sorted(set(source_issues)),
            "record_count": len(records),
            "rejected_context_count": len(rejected),
            "records": records,
            "rejected_contexts": rejected,
            "component_counts": {
                "prediction": len(records),
                "regime": 0,
                "specialist": 0,
                "context_synthesis": 0,
            },
            "case_contract": {
                "exact_ticker_required": True,
                "exact_timeframe_required": True,
                "exact_target_required": True,
                "exact_realization_timestamp_required": True,
                "immutable_prediction_source_hash_required": True,
                "immutable_outcome_source_hash_required": True,
                "future_rows_may_exist_but_are_never_selected": True,
            },
            "next_steps": [
                (
                    "Accumulate at least 30 accepted cases for the same "
                    "ticker/timeframe/target/context before diagnostics."
                ),
                (
                    "Join regime, specialist, and synthesis assessments "
                    "only through separate exact-context case producers."
                ),
                (
                    "Keep consensus weights unchanged until reviewed "
                    "diagnostics and human feedback exist."
                ),
            ],
            "safety": {
                "review_only": True,
                "calibration_executed": False,
                "automatic_weight_change_allowed": False,
                "decision_influence": False,
                "can_write_learning_memory": False,
                "can_write_production_config": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
        }
        payload["index_fingerprint"] = sha256_json({
            "source_inventory": payload["source_inventory"],
            "source_issues": payload["source_issues"],
            "records": records,
            "rejected_contexts": rejected,
        })
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_shadow_calibration_case_index_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def _build_prediction_case(
    *,
    context: dict[str, Any],
    prediction_review_state: dict[str, Any],
    outcome_state: dict[str, Any],
    outcome_rows: pd.DataFrame,
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    if context.get("lineage_status") != "complete":
        issues.append("prediction_lineage_incomplete")
    if context.get("review_issues"):
        issues.append("prediction_review_has_issues")

    ticker = _text(context.get("ticker"), upper=True)
    timeframe = _text(context.get("timeframe"), lower=True)
    target_name = _text(context.get("target_name"))
    model_context_id = _text(context.get("model_context_id"))
    context_fingerprint = _text(context.get("context_fingerprint"))
    if not all((
        ticker,
        timeframe,
        target_name,
        model_context_id,
        context_fingerprint,
    )):
        issues.append("exact_context_identity_incomplete")

    prediction = _mapping(context.get("prediction"))
    target_semantics = _mapping(context.get("target_semantics"))
    calibration = _mapping(target_semantics.get("calibration"))
    if target_semantics.get("status") != "target_semantics_ready":
        issues.append("target_semantics_not_ready")
    if calibration.get("model_output_scale_known") is not True:
        issues.append("model_output_scale_not_validated")

    prediction_as_of = _timestamp(prediction.get("as_of"))
    realization = _mapping(
        target_semantics.get("realization_window")
    )
    expected_end = _timestamp(realization.get("expected_end"))
    if prediction_as_of is None:
        issues.append("prediction_as_of_invalid")
    if expected_end is None:
        issues.append("realization_expected_end_invalid")
    if (
        prediction_as_of is not None
        and expected_end is not None
        and expected_end <= prediction_as_of
    ):
        issues.append("realization_window_not_forward")

    prediction_value = _finite_float(prediction.get("value"))
    start_price = _finite_float(prediction.get("last_price"))
    if prediction_value is None:
        issues.append("prediction_value_invalid")
    if start_price is None or start_price <= 0:
        issues.append("prediction_start_price_invalid")

    pipeline_source = _mapping(
        _mapping(
            prediction_review_state.get("payload")
        ).get("source_artifact")
    )
    pipeline_source_issues = _bound_source_issues(pipeline_source)
    issues.extend(
        f"pipeline_{issue}" for issue in pipeline_source_issues
    )

    selected = pd.DataFrame()
    if ticker and timeframe and expected_end is not None:
        selected = outcome_rows[
            (outcome_rows["ticker"] == ticker)
            & (outcome_rows["timeframe"] == timeframe)
            & (outcome_rows["observed_at"] == expected_end)
        ]
        if len(selected) != 1:
            issues.append(
                "outcome_row_missing"
                if selected.empty
                else "outcome_row_not_unique"
            )

    end_price = None
    if len(selected) == 1:
        end_price = _finite_float(selected.iloc[0]["close"])
        if end_price is None or end_price <= 0:
            issues.append("outcome_end_price_invalid")

    realized_return = None
    if (
        start_price is not None
        and start_price > 0
        and end_price is not None
    ):
        realized_return = end_price / start_price - 1.0
    realized_target = None
    if realized_return is not None:
        realized_target, target_issue = _realized_target(
            target_semantics,
            realized_return,
        )
        if target_issue:
            issues.append(target_issue)

    if issues:
        return None, sorted(set(issues))

    output_scale = target_semantics.get(
        "stage5_scalar_semantics"
    )
    model_output_contract = _mapping(
        context.get("model_output_contract")
    )
    raw_output_contract = _mapping(
        model_output_contract.get("raw_output")
    )
    identity = {
        "ticker": ticker,
        "timeframe": timeframe,
        "target_name": target_name,
        "model_context_id": model_context_id,
        "context_fingerprint": context_fingerprint,
    }
    source_max = _timestamp(
        outcome_state.get("max_observed_at")
    )
    source_contains_later_rows = bool(
        source_max is not None
        and expected_end is not None
        and source_max > expected_end
    )
    case_id = "shadow_case:" + sha256_json({
        "component": "prediction",
        "identity": identity,
        "prediction_as_of": prediction_as_of.isoformat(),
        "expected_end": expected_end.isoformat(),
        "prediction_review_sha256": prediction_review_state["sha256"],
        "outcome_source_sha256": outcome_state["sha256"],
    })[:24]
    return {
        "schema_version": "dean_shadow_calibration_case_v1",
        "case_id": case_id,
        "case_validation_status": "accepted",
        "component": "prediction",
        "identity": identity,
        "market_regime": context.get("market_regime") or "unknown",
        "prediction": {
            "as_of": prediction_as_of.isoformat(),
            "value": prediction_value,
            "output_scale": output_scale,
            "raw_value": _finite_float(
                prediction.get("raw_forecast")
            ),
            "raw_output_scale": raw_output_contract.get("scale"),
            "positive_class_probability": bool(
                _mapping(
                    model_output_contract.get("final_output")
                ).get("positive_class_probability", False)
            ),
            "target_type": target_semantics.get("target_type"),
            "confidence": _finite_float(
                prediction.get("confidence")
            ),
            "anomaly_score": _finite_float(
                prediction.get("anomaly_score")
            ),
        },
        "realization": {
            "expected_end": expected_end.isoformat(),
            "observed_at": expected_end.isoformat(),
            "timestamp_match": "exact",
            "start_price": start_price,
            "end_price": end_price,
            "realized_return": realized_return,
            "realized_target": realized_target,
            "target_type": target_semantics.get("target_type"),
            "target_unit": target_semantics.get("target_unit"),
        },
        "source_provenance": {
            "prediction_review": {
                "path": prediction_review_state["path"],
                "sha256": prediction_review_state["sha256"],
            },
            "pipeline_result": {
                "path": pipeline_source.get("path"),
                "sha256": pipeline_source.get("sha256"),
            },
            "outcome_source": {
                "path": outcome_state["path"],
                "sha256": outcome_state["sha256"],
                "selected_timestamp": expected_end.isoformat(),
                "source_contains_later_rows": (
                    source_contains_later_rows
                ),
                "later_rows_used": False,
            },
        },
        "safety": {
            "exact_context_match": True,
            "exact_realization_timestamp_match": True,
            "time_leakage_detected": False,
            "future_evidence_used": False,
            "sector_to_ticker_leakage_detected": False,
            "unsafe_output_detected": False,
            "decision_influence": False,
            "can_trade": False,
        },
    }, []


def validate_shadow_calibration_case(
    record: Any,
) -> list[str]:
    if not isinstance(record, dict):
        return ["case_not_mapping"]
    issues = []
    if record.get("schema_version") != (
        "dean_shadow_calibration_case_v1"
    ):
        issues.append("case_schema_mismatch")
    if record.get("case_validation_status") != "accepted":
        issues.append("case_not_accepted")
    if not _text(record.get("case_id")):
        issues.append("case_id_missing")
    if record.get("component") not in {
        "prediction",
        "regime",
        "specialist",
        "context_synthesis",
    }:
        issues.append("case_component_invalid")
    identity = _mapping(record.get("identity"))
    for field in (
        "ticker",
        "timeframe",
        "target_name",
        "context_fingerprint",
    ):
        if not _text(identity.get(field)):
            issues.append(f"case_identity_{field}_missing")
    realization = _mapping(record.get("realization"))
    prediction = _mapping(record.get("prediction"))
    prediction_as_of = _timestamp(prediction.get("as_of"))
    expected_end = _timestamp(realization.get("expected_end"))
    observed_at = _timestamp(realization.get("observed_at"))
    if prediction_as_of is None:
        issues.append("case_prediction_as_of_invalid")
    if expected_end is None or observed_at is None:
        issues.append("case_realization_timestamp_invalid")
    elif expected_end != observed_at:
        issues.append("case_realization_timestamp_not_exact")
    if (
        prediction_as_of is not None
        and expected_end is not None
        and expected_end <= prediction_as_of
    ):
        issues.append("case_realization_window_invalid")
    if _finite_float(realization.get("realized_return")) is None:
        issues.append("case_realized_return_invalid")
    if record.get("component") == "prediction":
        output_scale = _text(prediction.get("output_scale"))
        if (
            not output_scale
            or "unknown" in output_scale
            or "not_declared" in output_scale
        ):
            issues.append("case_prediction_output_scale_missing")
        if _finite_float(prediction.get("value")) is None:
            issues.append("case_prediction_value_invalid")
        if realization.get("realized_target") is None:
            issues.append("case_realized_target_missing")
    provenance = _mapping(record.get("source_provenance"))
    for source_name in (
        "prediction_review",
        "pipeline_result",
        "outcome_source",
    ):
        source = _mapping(provenance.get(source_name))
        if not _text(source.get("path")):
            issues.append(f"case_{source_name}_path_missing")
        if not _sha_text(source.get("sha256")):
            issues.append(f"case_{source_name}_sha256_invalid")
    safety = _mapping(record.get("safety"))
    if safety.get("exact_context_match") is not True:
        issues.append("case_exact_context_not_proven")
    if safety.get("exact_realization_timestamp_match") is not True:
        issues.append("case_exact_realization_not_proven")
    for field in (
        "time_leakage_detected",
        "future_evidence_used",
        "sector_to_ticker_leakage_detected",
        "unsafe_output_detected",
        "decision_influence",
        "can_trade",
    ):
        if safety.get(field) is not False:
            issues.append(f"case_unsafe_{field}")
    return sorted(set(issues))


def _prediction_packet_issues(
    payload: dict[str, Any],
) -> list[str]:
    issues = []
    if payload.get("mode") != "pipeline_prediction_review_packet":
        issues.append("prediction_review_mode_mismatch")
    if payload.get("schema_version") != (
        "dean_stage5_prediction_review_v1"
    ):
        issues.append("prediction_review_schema_mismatch")
    if payload.get("status") != "stage5_prediction_review_ready":
        issues.append("prediction_review_not_ready")
    if not payload.get("contexts"):
        issues.append("prediction_review_has_no_contexts")
    return issues


def _realized_target(
    semantics: dict[str, Any],
    realized_return: float | None,
) -> tuple[float | int | None, str | None]:
    if realized_return is None:
        return None, "realized_return_unavailable"
    target_type = str(semantics.get("target_type") or "")
    if target_type == "classification_binary":
        threshold = _finite_float(
            _mapping(semantics.get("threshold")).get("value")
        )
        classes = _mapping(semantics.get("class_semantics"))
        positive = classes.get("positive_class")
        negative = classes.get("negative_class")
        if threshold is None or positive is None or negative is None:
            return None, "binary_target_semantics_incomplete"
        return (
            int(positive)
            if realized_return > threshold
            else int(negative)
        ), None
    if (
        target_type == "regression"
        and semantics.get("target_unit") == "return_ratio"
    ):
        return realized_return, None
    return None, "target_outcome_binding_not_supported"


def _outcome_source_state(
    path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base = {
        "path": str(path),
        "available": False,
        "sha256": None,
        "row_count": 0,
        "max_observed_at": None,
        "issues": [],
    }
    if not path.is_file():
        base["issues"] = ["outcome_source_missing"]
        return base, _empty_outcomes()
    try:
        frame = _read_outcome_frame(path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        base["issues"] = ["outcome_source_unreadable"]
        return base, _empty_outcomes()
    normalized, issues = _normalize_outcome_frame(frame)
    base.update({
        "available": not issues,
        "sha256": _sha256(path),
        "row_count": len(normalized),
        "max_observed_at": (
            normalized["observed_at"].max().isoformat()
            if not normalized.empty
            else None
        ),
        "issues": issues,
    })
    return base, normalized


def _read_outcome_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if isinstance(raw, dict):
            for key in ("rows", "records", "data", "observations"):
                if isinstance(raw.get(key), list):
                    return pd.DataFrame(raw[key])
        raise ValueError("JSON outcome source has no row list.")
    raise ValueError("Unsupported outcome source format.")


def _normalize_outcome_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    aliases = {
        "ticker": ("ticker", "symbol"),
        "timeframe": ("timeframe", "interval"),
        "observed_at": ("observed_at", "datetime", "timestamp", "date"),
        "close": ("close", "Close"),
    }
    selected = {}
    issues = []
    for canonical, candidates in aliases.items():
        match = next(
            (name for name in candidates if name in frame.columns),
            None,
        )
        if match is None:
            issues.append(f"outcome_{canonical}_column_missing")
        else:
            selected[canonical] = frame[match]
    if issues:
        return _empty_outcomes(), issues
    normalized = pd.DataFrame(selected)
    normalized["ticker"] = (
        normalized["ticker"].astype(str).str.strip().str.upper()
    )
    normalized["timeframe"] = (
        normalized["timeframe"].astype(str).str.strip().str.lower()
    )
    normalized["observed_at"] = pd.to_datetime(
        normalized["observed_at"],
        utc=True,
        errors="coerce",
    )
    normalized["close"] = pd.to_numeric(
        normalized["close"],
        errors="coerce",
    )
    if normalized["observed_at"].isna().any():
        issues.append("outcome_timestamp_invalid")
    if normalized["close"].isna().any():
        issues.append("outcome_close_invalid")
    if (normalized["ticker"] == "").any():
        issues.append("outcome_ticker_invalid")
    if (normalized["timeframe"] == "").any():
        issues.append("outcome_timeframe_invalid")
    return normalized, sorted(set(issues))


def _json_artifact_state(path: Path) -> dict[str, Any]:
    state = {
        "path": str(path),
        "available": False,
        "sha256": None,
        "payload": {},
        "issues": [],
    }
    if not path.is_file():
        state["issues"] = ["artifact_missing"]
        return state
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state["issues"] = ["artifact_unreadable"]
        return state
    if not isinstance(payload, dict):
        state["issues"] = ["artifact_payload_invalid"]
        return state
    state.update({
        "available": True,
        "sha256": _sha256(path),
        "payload": payload,
    })
    return state


def _bound_source_issues(source: dict[str, Any]) -> list[str]:
    path_text = _text(source.get("path"))
    expected_sha = _text(source.get("sha256"))
    if (
        source.get("available") is not True
        or source.get("immutable_binding_ready") is not True
        or not path_text
        or not _sha_text(expected_sha)
    ):
        return ["result_source_binding_missing"]
    path = Path(path_text)
    if not path.is_file():
        return ["result_source_missing"]
    if _sha256(path) != expected_sha:
        return ["result_source_hash_mismatch"]
    return []


def _deduplicated_records(
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for record in records:
        case_id = record["case_id"]
        if case_id in seen:
            rejected.append({
                "context_key": case_id,
                "issues": ["duplicate_case_identity"],
            })
            continue
        seen.add(case_id)
        result.append(record)
    return result


def render_shadow_calibration_case_index_markdown(
    payload: dict[str, Any],
) -> str:
    lines = [
        "# DEAN-OS Shadow Calibration Case Index",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Accepted prediction cases: {payload.get('record_count')}",
        (
            "- Rejected contexts: "
            f"{payload.get('rejected_context_count')}"
        ),
        "- Calibration executed: False",
        "- Consensus weight changed: False",
        "- Can trade: False",
        "",
        "| Case | Ticker | Timeframe | Target | As of | Realized at | Return |",
        "|---|---|---|---|---|---|---:|",
    ]
    for record in payload.get("records", []):
        identity = _mapping(record.get("identity"))
        prediction = _mapping(record.get("prediction"))
        realization = _mapping(record.get("realization"))
        lines.append(
            "| {case} | {ticker} | {timeframe} | {target} | "
            "{as_of} | {end} | {ret} |".format(
                case=record.get("case_id"),
                ticker=identity.get("ticker"),
                timeframe=identity.get("timeframe"),
                target=identity.get("target_name"),
                as_of=prediction.get("as_of"),
                end=realization.get("observed_at"),
                ret=realization.get("realized_return"),
            )
        )
    if payload.get("source_issues"):
        lines.extend(["", "## Source Blockers", ""])
        lines.extend(
            f"- `{issue}`"
            for issue in payload.get("source_issues", [])
        )
    if payload.get("rejected_contexts"):
        lines.extend(["", "## Rejected Contexts", ""])
        for item in payload["rejected_contexts"]:
            lines.append(
                f"- `{item.get('context_key')}`: "
                + ", ".join(item.get("issues", []))
            )
    lines.extend([
        "",
        "Cases are review-only realized-outcome bindings. They do not "
        "change agent weights, learning memory, recommendations, or trades.",
    ])
    return "\n".join(lines) + "\n"


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key != "payload"
    }


def _empty_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "timeframe", "observed_at", "close"]
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(
    value: Any,
    *,
    upper: bool = False,
    lower: bool = False,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if upper:
        return text.upper()
    if lower:
        return text.lower()
    return text


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha_text(value: Any) -> bool:
    text = _text(value)
    return bool(
        text
        and len(text) == 64
        and all(char in "0123456789abcdef" for char in text.lower())
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"shadow_calibration_case_index_{stamp}"
