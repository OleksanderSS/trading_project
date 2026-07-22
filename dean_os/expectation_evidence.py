from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from dean_os.context_evidence_provenance import parse_timezone_aware


CONTRACT = "dean_expectation_evidence_v1"
EXPECTATION_TYPES = {
    "analyst_consensus",
    "management_guidance",
    "survey_consensus",
    "market_implied_probability",
    "rates_implied_path",
    "options_implied_volatility",
    "credit_spread_signal",
    "positioning",
}


def validate_expectation_evidence(
    payload: Any, *, as_of: str
) -> dict[str, Any]:
    """Validate one expectation observation without interpreting its direction."""
    cutoff = parse_timezone_aware(as_of)
    if cutoff is None:
        raise ValueError("expectation as_of must be timezone-aware")
    if not isinstance(payload, dict):
        return _result([], ["expectation_evidence_not_structured"])

    reasons: list[str] = []
    expectation_type = str(payload.get("expectation_type") or "")
    if payload.get("contract") != CONTRACT:
        reasons.append("expectation_contract_missing_or_invalid")
    if expectation_type not in EXPECTATION_TYPES:
        reasons.append("expectation_type_invalid")

    actual = _observation(payload.get("actual"), "actual", cutoff)
    expected = _observation(payload.get("expected"), "expected", cutoff)
    reasons.extend(actual.pop("reasons"))
    reasons.extend(expected.pop("reasons"))

    if actual.get("unit") and expected.get("unit") and actual["unit"] != expected["unit"]:
        reasons.append("actual_expected_unit_mismatch")
    actual_time = parse_timezone_aware(str(actual.get("available_at") or ""))
    expected_time = parse_timezone_aware(str(expected.get("available_at") or ""))
    if actual_time and expected_time and expected_time > actual_time:
        reasons.append("expectation_available_after_actual")

    std = payload.get("expectation_std")
    if std is not None and (not _finite(std) or float(std) <= 0):
        reasons.append("expectation_std_invalid")

    normalized = {
        "contract": CONTRACT,
        "expectation_type": expectation_type,
        "actual": actual,
        "expected": expected,
        "expectation_std": float(std) if _finite(std) else None,
        "as_of": cutoff.isoformat(),
    }
    normalized["evidence_sha256"] = _sha256(normalized)
    return _result([normalized] if not reasons else [], reasons, candidate=normalized)


def _observation(raw: Any, role: str, cutoff: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"role": role, "reasons": [f"{role}_observation_not_structured"]}
    value = raw.get("value")
    if not _finite(value):
        reasons.append(f"{role}_value_invalid")
    unit = str(raw.get("unit") or "").strip()
    if not unit:
        reasons.append(f"{role}_unit_missing")
    available_at = parse_timezone_aware(str(raw.get("available_at") or ""))
    if available_at is None:
        reasons.append(f"{role}_available_at_invalid")
    elif available_at > cutoff:
        reasons.append(f"{role}_available_after_as_of")
    locator = str(raw.get("source_locator") or "").strip()
    if not locator:
        reasons.append(f"{role}_source_locator_missing")
    source_hash = str(raw.get("source_sha256") or "").lower()
    if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        reasons.append(f"{role}_source_sha256_invalid")
    return {
        "role": role,
        "value": float(value) if _finite(value) else None,
        "unit": unit or None,
        "available_at": available_at.isoformat() if available_at else None,
        "source_locator": locator or None,
        "source_sha256": source_hash or None,
        "vintage": raw.get("vintage"),
        "revised": bool(raw.get("revised", False)),
        "reasons": reasons,
    }


def _result(
    accepted: list[dict[str, Any]],
    reasons: list[str],
    *,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "status": "point_in_time_ready" if accepted else "not_quantifiable",
        "quantitative_gap_allowed": bool(accepted),
        "accepted": accepted,
        "candidate": candidate,
        "reasons": sorted(set(reasons)),
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["CONTRACT", "EXPECTATION_TYPES", "validate_expectation_evidence"]
