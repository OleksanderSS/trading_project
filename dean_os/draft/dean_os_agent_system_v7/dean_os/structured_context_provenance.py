from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware

STRUCTURED_CONTEXT_CONTRACT = (
    "dean_structured_context_point_in_time_v1"
)

AVAILABILITY_FIELDS = (
    "available_at",
    "published_at",
    "released_at",
    "filed_at",
    "filing_date",
    "retrieved_at",
    "ingested_at",
    "timestamp",
)
PERIOD_FIELDS = (
    "period",
    "reporting_period",
    "fiscal_period",
    "observation_date",
    "effective_date",
    "period_end",
)
SOURCE_LOCATOR_FIELDS = (
    "source_locator",
    "source_url",
    "source_uri",
    "url",
    "uri",
    "source_id",
    "source_hash",
    "accession_number",
    "citation",
    "source_citation",
    "source",
)
SEMANTIC_FIELDS = (
    "evidence_type",
    "required_lane_eligible",
    "stance_hint",
)
RESERVED_KEYS = {
    "metadata",
    "provenance",
    "_provenance",
    "_dean_structured_provenance",
    "metrics",
    "observations",
    "series",
    "values",
    "units",
    "periods",
    *AVAILABILITY_FIELDS,
    *PERIOD_FIELDS,
    *SOURCE_LOCATOR_FIELDS,
    *SEMANTIC_FIELDS,
}


def audit_structured_context(
    *,
    fundamentals: dict[str, Any] | None,
    macro: dict[str, Any] | None,
    sector_data: dict[str, Any] | None,
    as_of: str,
) -> dict[str, Any]:
    as_of_dt = parse_timezone_aware(as_of)
    if as_of_dt is None:
        raise ValueError(
            "structured context as_of must be a timezone-aware ISO-8601 "
            "timestamp"
        )

    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    accepted_context: dict[str, dict[str, Any]] = {
        "fundamentals": {},
        "macro": {},
        "sector_data": {},
    }
    seen_hashes: set[str] = set()

    for ticker, payload in sorted((fundamentals or {}).items()):
        family = "fundamental"
        if not isinstance(payload, dict):
            exclusions.append(
                _shape_exclusion(family, str(ticker), "ticker_payload")
            )
            continue
        candidates = _candidate_map(payload)
        if not candidates:
            exclusions.append(
                _shape_exclusion(family, str(ticker), "metrics")
            )
            continue
        accepted_values: dict[str, Any] = {}
        accepted_provenance: dict[str, Any] = {}
        for metric_name, raw in sorted(candidates.items()):
            observation, reasons = _normalize_observation(
                family=family,
                scope=str(ticker).upper(),
                name=str(metric_name),
                raw=raw,
                parent=payload,
                as_of=as_of_dt,
                numeric_required=True,
            )
            _record_result(
                observation=observation,
                reasons=reasons,
                observations=observations,
                exclusions=exclusions,
                seen_hashes=seen_hashes,
                accepted_values=accepted_values,
                accepted_provenance=accepted_provenance,
            )
        if accepted_values:
            accepted_context["fundamentals"][
                str(ticker).upper()
            ] = {
                **accepted_values,
                "_dean_structured_provenance": {
                    "contract": STRUCTURED_CONTEXT_CONTRACT,
                    "as_of": as_of_dt.isoformat(),
                    "observations": accepted_provenance,
                },
            }

    for family, payload, target_key, numeric_required in (
        ("macro", macro or {}, "macro", True),
        ("sector", sector_data or {}, "sector_data", False),
    ):
        if payload and not isinstance(payload, dict):
            exclusions.append(
                _shape_exclusion(family, family, "observation_map")
            )
            continue
        candidates = _candidate_map(payload)
        if payload and not candidates:
            exclusions.append(
                _shape_exclusion(family, family, "observations")
            )
        accepted_values = {}
        accepted_provenance = {}
        for name, raw in sorted(candidates.items()):
            observation, reasons = _normalize_observation(
                family=family,
                scope=family,
                name=str(name),
                raw=raw,
                parent=payload,
                as_of=as_of_dt,
                numeric_required=numeric_required,
            )
            _record_result(
                observation=observation,
                reasons=reasons,
                observations=observations,
                exclusions=exclusions,
                seen_hashes=seen_hashes,
                accepted_values=accepted_values,
                accepted_provenance=accepted_provenance,
            )
        if accepted_values:
            accepted_context[target_key] = {
                **accepted_values,
                "_dean_structured_provenance": {
                    "contract": STRUCTURED_CONTEXT_CONTRACT,
                    "as_of": as_of_dt.isoformat(),
                    "observations": accepted_provenance,
                },
            }

    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion.get("reasons", [])
    )
    fingerprint = _canonical_sha256(
        [
            {
                key: observation.get(key)
                for key in (
                    "family",
                    "scope",
                    "name",
                    "value",
                    "unit",
                    "period",
                    "available_at",
                    "source_locator",
                    "observation_sha256",
                )
            }
            for observation in observations
        ]
    )
    input_count = len(observations) + len(exclusions)
    return {
        "contract": STRUCTURED_CONTEXT_CONTRACT,
        "status": (
            "point_in_time_ready_with_exclusions"
            if observations and exclusions
            else "point_in_time_ready"
            if observations
            else "no_structured_context"
            if input_count == 0
            else "blocked_no_point_in_time_structured_context"
        ),
        "as_of": as_of_dt.isoformat(),
        "input_count": input_count,
        "accepted_count": len(observations),
        "excluded_count": len(exclusions),
        "family_counts": dict(
            sorted(
                Counter(
                    observation["family"]
                    for observation in observations
                ).items()
            )
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "accepted_observations": observations,
        "accepted_context": accepted_context,
        "exclusions": exclusions,
        "accepted_fingerprint": fingerprint,
        "semantic_rule": (
            "Every structured observation requires an explicit value, unit, "
            "period, point-in-time availability timestamp, and stable source "
            "locator. Raw table inventory is not evidence."
        ),
    }


def audit_market_context_structured(context: Any) -> dict[str, Any]:
    fundamentals = dict(getattr(context, "fundamentals", {}) or {})
    macro = dict(getattr(context, "macro", {}) or {})
    sector_data = dict(getattr(context, "sector_data", {}) or {})
    as_of = getattr(context, "as_of", None)
    input_families = {
        "fundamentals": fundamentals,
        "macro": macro,
        "sector_data": sector_data,
    }
    populated = {
        key: value for key, value in input_families.items() if value
    }
    if not as_of:
        return _blocked_context_audit(
            populated,
            reason="context_as_of_missing",
            as_of=None,
        )
    try:
        return audit_structured_context(
            fundamentals=fundamentals,
            macro=macro,
            sector_data=sector_data,
            as_of=str(as_of),
        )
    except ValueError:
        return _blocked_context_audit(
            populated,
            reason="context_as_of_invalid",
            as_of=str(as_of),
        )


def apply_market_context_structured_boundary(
    context: Any,
) -> dict[str, Any]:
    audit = audit_market_context_structured(context)
    accepted = audit.get("accepted_context", {})
    context.fundamentals = dict(accepted.get("fundamentals", {}))
    context.macro = dict(accepted.get("macro", {}))
    context.sector_data = dict(accepted.get("sector_data", {}))
    context.metadata["structured_context_point_in_time_audit"] = {
        key: value
        for key, value in audit.items()
        if key not in {"accepted_context", "accepted_observations"}
    }
    return audit


def _candidate_map(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "observations", "series", "values"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return {
                str(name): raw
                for name, raw in nested.items()
                if not str(name).startswith("_")
            }
    return {
        str(name): raw
        for name, raw in payload.items()
        if name not in RESERVED_KEYS and not str(name).startswith("_")
    }


def _normalize_observation(
    *,
    family: str,
    scope: str,
    name: str,
    raw: Any,
    parent: dict[str, Any],
    as_of: Any,
    numeric_required: bool,
) -> tuple[dict[str, Any], list[str]]:
    child = raw if isinstance(raw, dict) else {}
    previous = (
        parent.get("_dean_structured_provenance", {})
        .get("observations", {})
        .get(name, {})
    )
    layers = _metadata_layers(child, previous, parent)
    value = child.get("value") if isinstance(raw, dict) else raw
    unit = _first_layer_value(layers, ("unit",))
    if _is_empty(unit):
        units = parent.get("units")
        unit = units.get(name) if isinstance(units, dict) else unit
    period = _first_layer_value(layers, PERIOD_FIELDS)
    if _is_empty(period):
        periods = parent.get("periods")
        period = (
            periods.get(name) if isinstance(periods, dict) else period
        )
    timestamp_field, available_at = _first_timestamp(
        layers,
        AVAILABILITY_FIELDS,
    )
    locator_field, locator = _first_nonempty(
        layers,
        SOURCE_LOCATOR_FIELDS,
    )
    if previous:
        if (
            locator_field == "source_locator"
            and previous.get("source_locator_field")
        ):
            locator_field = previous["source_locator_field"]
        if (
            timestamp_field == "available_at"
            and previous.get("availability_timestamp_field")
        ):
            timestamp_field = previous[
                "availability_timestamp_field"
            ]

    reasons: list[str] = []
    if not _valid_value(value, numeric_required=numeric_required):
        reasons.append("structured_value_missing_or_invalid")
    if _is_empty(unit):
        reasons.append("structured_unit_missing")
    if _is_empty(period):
        reasons.append("structured_period_missing")
    if available_at is None:
        reasons.append(
            "structured_availability_timestamp_missing_or_invalid"
        )
    elif available_at > as_of:
        reasons.append("structured_availability_after_as_of")
    if _is_empty(locator):
        reasons.append("structured_source_locator_missing")

    normalized_value = (
        float(value)
        if numeric_required
        and _valid_value(value, numeric_required=True)
        else value
    )
    canonical = {
        "contract": STRUCTURED_CONTEXT_CONTRACT,
        "family": family,
        "scope": scope,
        "name": name,
        "value": normalized_value,
        "unit": str(unit) if not _is_empty(unit) else None,
        "period": str(period) if not _is_empty(period) else None,
        "available_at": (
            available_at.isoformat() if available_at else None
        ),
        "availability_timestamp_field": timestamp_field,
        "source_locator": (
            str(locator) if not _is_empty(locator) else None
        ),
        "source_locator_field": locator_field,
        "as_of": as_of.isoformat(),
    }
    evidence_type = _first_layer_value(layers, ("evidence_type",))
    if not _is_empty(evidence_type):
        canonical["evidence_type"] = str(evidence_type)
    required_lane_eligible = _first_layer_value(
        layers,
        ("required_lane_eligible",),
    )
    if isinstance(required_lane_eligible, bool):
        canonical["required_lane_eligible"] = required_lane_eligible
    stance_hint = _first_layer_value(layers, ("stance_hint",))
    if stance_hint in {
        "positive",
        "negative",
        "neutral",
        "mixed",
        "unknown",
    }:
        canonical["stance_hint"] = stance_hint
    canonical["observation_sha256"] = _canonical_sha256(canonical)
    return canonical, reasons


def _record_result(
    *,
    observation: dict[str, Any],
    reasons: list[str],
    observations: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    seen_hashes: set[str],
    accepted_values: dict[str, Any],
    accepted_provenance: dict[str, Any],
) -> None:
    observation_hash = observation["observation_sha256"]
    if observation_hash in seen_hashes:
        reasons.append("duplicate_structured_observation")
    if reasons:
        exclusions.append(
            {
                "family": observation["family"],
                "scope": observation["scope"],
                "name": observation["name"],
                "status": "excluded",
                "reasons": sorted(set(reasons)),
                "provenance": observation,
            }
        )
        return
    seen_hashes.add(observation_hash)
    observation["status"] = "point_in_time_compatible"
    observations.append(observation)
    accepted_values[observation["name"]] = observation["value"]
    accepted_provenance[observation["name"]] = {
        key: value
        for key, value in observation.items()
        if key not in {"value", "status"}
    }


def _metadata_layers(
    child: dict[str, Any],
    previous: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for item in (child, previous, parent):
        if not isinstance(item, dict):
            continue
        layers.append(item)
        for key in ("provenance", "_provenance", "metadata"):
            nested = item.get(key)
            if isinstance(nested, dict):
                layers.append(nested)
    return layers


def _first_timestamp(
    layers: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[str | None, Any | None]:
    for layer in layers:
        for field in fields:
            if field not in layer:
                continue
            parsed = parse_timezone_aware(layer.get(field))
            if parsed is not None:
                return field, parsed
    return None, None


def _first_nonempty(
    layers: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[str | None, Any | None]:
    for layer in layers:
        for field in fields:
            value = layer.get(field)
            if not _is_empty(value):
                return field, value
    return None, None


def _first_layer_value(
    layers: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> Any | None:
    return _first_nonempty(layers, fields)[1]


def _valid_value(value: Any, *, numeric_required: bool) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if numeric_required:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _shape_exclusion(
    family: str,
    scope: str,
    expected: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "scope": scope,
        "status": "excluded",
        "reasons": [f"structured_{expected}_missing_or_invalid"],
    }


def _blocked_context_audit(
    populated: dict[str, Any],
    *,
    reason: str,
    as_of: str | None,
) -> dict[str, Any]:
    exclusions = [
        {
            "family": family,
            "scope": family,
            "status": "excluded",
            "reasons": [reason],
        }
        for family in populated
    ]
    return {
        "contract": STRUCTURED_CONTEXT_CONTRACT,
        "status": (
            "no_structured_context"
            if not populated
            else f"blocked_{reason}"
        ),
        "as_of": as_of,
        "input_count": len(populated),
        "accepted_count": 0,
        "excluded_count": len(exclusions),
        "family_counts": {},
        "reason_counts": (
            {reason: len(exclusions)} if exclusions else {}
        ),
        "accepted_observations": [],
        "accepted_context": {
            "fundamentals": {},
            "macro": {},
            "sector_data": {},
        },
        "exclusions": exclusions,
        "accepted_fingerprint": _canonical_sha256([]),
        "semantic_rule": (
            "Structured context is unavailable without an explicit "
            "timezone-aware analysis cutoff."
        ),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
