from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.targets.timeframe_contract import target_applies_to_timeframe

TIMEFRAME_SECONDS = {
    "15m": 15 * 60,
    "60m": 60 * 60,
    "1d": 24 * 60 * 60,
}
HORIZON_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}


class PredictionTargetSemanticsRegistry:
    """Resolve target period/unit/class meaning without guessing output scale."""

    def __init__(
        self,
        config_path: str | Path = "src/config/targets.yaml",
    ):
        self.config_path = Path(config_path)
        raw = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        ) or {}
        self.targets = raw.get("targets", {})
        self.config_sha256 = hashlib.sha256(
            self.config_path.read_bytes()
        ).hexdigest()

    def resolve(
        self,
        *,
        target_name: str | None,
        timeframe: str | None,
        prediction_as_of: str | None = None,
        model_output_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = str(target_name or "")
        normalized_timeframe = _normalize_timeframe(timeframe)
        target = self.targets.get(name)
        if not isinstance(target, dict):
            return _unavailable(
                name=name,
                timeframe=normalized_timeframe,
                config_path=self.config_path,
                config_sha256=self.config_sha256,
                issue="target_not_found_in_canonical_config",
            )
        params = (
            target.get("params")
            if isinstance(target.get("params"), dict)
            else {}
        )
        horizon = _text_or_none(params.get("horizon"))
        shift_bars, horizon_seconds, period_issue = _period(
            params=params,
            timeframe=normalized_timeframe,
            horizon=horizon,
        )
        timeframe_compatible = bool(
            normalized_timeframe
            and target_applies_to_timeframe(
                {"name": name, **target},
                normalized_timeframe,
            )
        )
        issues = []
        if not normalized_timeframe:
            issues.append("prediction_timeframe_missing")
        elif not timeframe_compatible:
            issues.append("target_timeframe_mismatch")
        if period_issue:
            issues.append(period_issue)
        target_type = str(target.get("type") or "unknown")
        output_contract = _model_output_contract_state(
            model_output_contract,
            target_name=name,
            target_type=target_type,
        )
        threshold = _float_or_none(params.get("threshold"))
        class_semantics = _class_semantics(
            target_type=target_type,
            base_col=params.get("base_col"),
            threshold=threshold,
            thresholds=params.get("thresholds"),
        )
        target_unit = _target_unit(
            target_type=target_type,
            name=name,
            base_col=params.get("base_col"),
        )
        as_of = _parse_timestamp(prediction_as_of)
        realization_end = (
            as_of + timedelta(seconds=horizon_seconds)
            if as_of is not None and horizon_seconds is not None
            else None
        )
        if prediction_as_of is None:
            issues.append("prediction_as_of_missing")
        if realization_end is None:
            issues.append("realization_window_unresolved")
        numeric_metadata_complete = all(
            (
                target_type != "unknown",
                target_unit is not None,
                shift_bars is not None,
                horizon_seconds is not None,
                normalized_timeframe is not None,
            )
        )
        if not numeric_metadata_complete:
            issues.append("numeric_period_metadata_incomplete")
        return {
            "schema_version": "dean_prediction_target_semantics_v1",
            "status": (
                "target_semantics_ready"
                if not issues
                else "target_semantics_partial"
            ),
            "target_name": name,
            "target_type": target_type,
            "description": params.get("description"),
            "base_column": params.get("base_col"),
            "target_unit": target_unit,
            "timeframe": normalized_timeframe,
            "timeframe_compatible": timeframe_compatible,
            "horizon": horizon,
            "shift_bars": shift_bars,
            "horizon_seconds": horizon_seconds,
            "realization_window": {
                "prediction_as_of": prediction_as_of,
                "expected_end": (
                    realization_end.isoformat()
                    if realization_end
                    else None
                ),
                "uses_future_observation": (
                    realization_end is not None
                    and as_of is not None
                    and realization_end > as_of
                ),
            },
            "threshold": {
                "value": threshold,
                "unit": "return_ratio"
                if threshold is not None
                else None,
                "percent": (
                    threshold * 100.0
                    if threshold is not None
                    else None
                ),
            },
            "class_semantics": class_semantics,
            "stage5_scalar_semantics": output_contract[
                "final_output_scale"
            ],
            "model_output_contract_validation": output_contract,
            "numeric_metadata_complete": (
                numeric_metadata_complete
            ),
            "issues": sorted(set(issues)),
            "source_provenance": {
                "config_path": str(self.config_path),
                "config_sha256": self.config_sha256,
                "source_contract": (
                    "targets_yaml_plus_target_timeframe_contract"
                ),
            },
            "calibration": {
                "target_direction_semantics_known": bool(
                    class_semantics
                ),
                "model_output_scale_known": output_contract[
                    "model_output_scale_known"
                ],
                "realized_outcome_supplied": False,
                "directional_inference_allowed": False,
                "calibration_eligible_now": False,
            },
            "decision_influence": False,
            "can_trade": False,
        }


def _period(
    *,
    params: dict[str, Any],
    timeframe: str | None,
    horizon: str | None,
) -> tuple[int | None, int | None, str | None]:
    timeframe_seconds = TIMEFRAME_SECONDS.get(timeframe or "")
    if horizon:
        horizon_seconds = HORIZON_SECONDS.get(horizon.lower())
        if horizon_seconds is None:
            return None, None, "unsupported_target_horizon"
        if timeframe_seconds is None:
            return None, horizon_seconds, "unsupported_prediction_timeframe"
        ratio = horizon_seconds / timeframe_seconds
        if ratio < 1 or ratio != int(ratio):
            return None, horizon_seconds, "horizon_not_exact_bar_multiple"
        return int(ratio), horizon_seconds, None
    try:
        shift_bars = abs(int(params.get("shift")))
    except (TypeError, ValueError):
        return None, None, "target_shift_missing"
    if shift_bars < 1:
        return None, None, "target_shift_invalid"
    if timeframe_seconds is None:
        return shift_bars, None, "unsupported_prediction_timeframe"
    return shift_bars, timeframe_seconds * shift_bars, None


def _class_semantics(
    *,
    target_type: str,
    base_col: Any,
    threshold: float | None,
    thresholds: Any,
) -> dict[str, Any]:
    if target_type == "classification_binary":
        return {
            "negative_class": 0,
            "positive_class": 1,
            "positive_condition": (
                f"future_{base_col}_return > {threshold}"
                if threshold is not None
                else "configured_binary_condition_is_true"
            ),
            "positive_direction": (
                "up" if base_col == "close" else "event_present"
            ),
        }
    if (
        target_type == "classification_multiclass"
        and isinstance(thresholds, list)
        and len(thresholds) == 2
    ):
        return {
            "classes": {
                "0": f"return <= {thresholds[0]}",
                "1": (
                    f"{thresholds[0]} < return < "
                    f"{thresholds[1]}"
                ),
                "2": f"return >= {thresholds[1]}",
            }
        }
    return {}


def _target_unit(
    *,
    target_type: str,
    name: str,
    base_col: Any,
) -> str | None:
    if target_type.startswith("classification"):
        return "class"
    if "return" in name:
        return "return_ratio"
    if "volatility" in name:
        return "range_or_volatility_proxy"
    if target_type == "indicator_prediction":
        return f"{base_col}_native_unit" if base_col else None
    if target_type == "regression":
        return f"{base_col}_derived_value" if base_col else None
    return None


def _model_output_contract_state(
    contract: dict[str, Any] | None,
    *,
    target_name: str,
    target_type: str,
) -> dict[str, Any]:
    if not isinstance(contract, dict) or not contract:
        return {
            "status": "model_output_contract_missing",
            "model_output_scale_known": False,
            "final_output_scale": "model_output_scale_not_declared",
            "issues": ["model_output_contract_missing"],
        }

    issues: list[str] = []
    if contract.get("schema_version") != (
        "dean_stage5_model_output_contract_v1"
    ):
        issues.append("model_output_contract_schema_mismatch")
    if contract.get("status") != "model_output_contract_ready":
        issues.append("model_output_contract_not_ready")
    if str(contract.get("target_name") or "") != target_name:
        issues.append("model_output_contract_target_mismatch")
    if _target_type_family(contract.get("target_type")) != (
        _target_type_family(target_type)
    ):
        issues.append("model_output_contract_target_type_mismatch")
    if contract.get("prediction_method") != "predict":
        issues.append("model_output_contract_prediction_method_mismatch")

    final_output = (
        contract.get("final_output")
        if isinstance(contract.get("final_output"), dict)
        else {}
    )
    final_scale = _text_or_none(final_output.get("scale"))
    if not final_scale:
        issues.append("model_output_contract_final_scale_missing")
    if final_output.get("model_output_scale_known") is not True:
        issues.append("model_output_contract_scale_not_known")
    if final_output.get("directional_inference_allowed") is not False:
        issues.append("unsafe_directional_inference_claim")
    if contract.get("decision_influence") is not False:
        issues.append("unsafe_model_output_decision_influence")

    valid = not issues
    return {
        "status": (
            "model_output_contract_validated"
            if valid
            else "model_output_contract_rejected"
        ),
        "model_output_scale_known": valid,
        "final_output_scale": (
            final_scale
            if valid
            else "model_output_scale_not_declared"
        ),
        "positive_class_probability": (
            final_output.get("positive_class_probability") is True
            if valid
            else False
        ),
        "issues": sorted(set(issues)),
    }


def _target_type_family(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("classification"):
        return "classification"
    if normalized in {"regression", "indicator_prediction"}:
        return "regression"
    return normalized


def _unavailable(
    *,
    name: str,
    timeframe: str | None,
    config_path: Path,
    config_sha256: str,
    issue: str,
) -> dict[str, Any]:
    return {
        "schema_version": "dean_prediction_target_semantics_v1",
        "status": "target_semantics_unavailable",
        "target_name": name or None,
        "timeframe": timeframe,
        "numeric_metadata_complete": False,
        "issues": [issue],
        "source_provenance": {
            "config_path": str(config_path),
            "config_sha256": config_sha256,
        },
        "calibration": {
            "target_direction_semantics_known": False,
            "model_output_scale_known": False,
            "realized_outcome_supplied": False,
            "directional_inference_allowed": False,
            "calibration_eligible_now": False,
        },
        "decision_influence": False,
        "can_trade": False,
    }


def _parse_timestamp(value: Any) -> datetime | None:
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


def _normalize_timeframe(value: Any) -> str | None:
    text = _text_or_none(value)
    if not text:
        return None
    normalized = text.lower()
    return {
        "15min": "15m",
        "1h": "60m",
        "60min": "60m",
        "daily": "1d",
    }.get(normalized, normalized)


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None
