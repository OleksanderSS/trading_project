from __future__ import annotations

__all__ = [
    'AVAILABILITY_FIELDS',
    'DEFAULT_SERIES_REGISTRY',
    'OBSERVATION_FIELDS',
    'SAVED_MACRO_PRODUCER_CONTRACT',
    'SERIES_FIELDS',
    'SavedMacroEvidenceProducer',
    'load_verified_macro_context_fragment',
    'render_saved_macro_evidence_markdown',
]

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import (
    audit_structured_context,
)

SAVED_MACRO_PRODUCER_CONTRACT = (
    "dean_saved_macro_evidence_producer_v1"
)
DEFAULT_SERIES_REGISTRY = (
    Path(__file__).parent / "config" / "macro_series_registry.yaml"
)
OBSERVATION_FIELDS = ("datetime", "date", "timestamp")
SERIES_FIELDS = ("series_id", "series", "indicator")
AVAILABILITY_FIELDS = (
    "available_at",
    "published_at",
    "released_at",
    "realtime_start",
)


class SavedMacroEvidenceProducer:
    """Normalize a saved long-form macro snapshot into review evidence.

    The producer is offline and fail-closed. It does not collect data, infer
    missing units, or treat file modification time as a source release/vintage
    timestamp.
    """

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_macro_evidence_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        source_path: str | Path,
        as_of: str,
        registry_path: str | Path = DEFAULT_SERIES_REGISTRY,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "macro producer as_of must be a timezone-aware ISO-8601 "
                "timestamp"
            )
        source = Path(source_path)
        registry_source = Path(registry_path)
        run_id = _run_id()
        source_audit = _source_audit(source)
        registry = _load_registry(registry_source)
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_macro_evidence_producer",
            "producer_contract": SAVED_MACRO_PRODUCER_CONTRACT,
            "inputs": {
                "source_path": str(source),
                "registry_path": str(registry_source),
                "as_of": as_of_dt.isoformat(),
            },
            "source_provenance": source_audit,
            "registry": {
                "registry_version": registry.get("registry_version"),
                "review_status": registry.get("review_status"),
                "source_family": registry.get("source_family"),
                "series_count": len(registry.get("series", {})),
                "path": str(registry_source),
                "sha256": (
                    _sha256_file(registry_source)
                    if registry_source.exists()
                    else None
                ),
            },
        }

        if not source_audit["exists"]:
            payload.update(
                _blocked_payload(
                    "blocked_source_missing",
                    ["macro_source_missing"],
                )
            )
            return self._finish(payload, save=save)
        if not registry.get("series"):
            payload.update(
                _blocked_payload(
                    "blocked_registry_missing_or_empty",
                    ["macro_series_registry_missing_or_empty"],
                )
            )
            return self._finish(payload, save=save)

        try:
            frame = _load_frame(source)
        except (ValueError, OSError, ImportError) as exc:
            payload.update(
                _blocked_payload(
                    "blocked_source_unreadable",
                    ["macro_source_unreadable"],
                )
            )
            payload["source_provenance"]["load_error"] = str(exc)
            return self._finish(payload, save=save)

        normalized = _normalize_frame(
            frame,
            as_of=as_of_dt,
            registry=registry["series"],
            source_sha256=source_audit["sha256"],
        )
        macro = {
            item["context_key"]: {
                "value": item["value"],
                "unit": item["unit"],
                "period": item["period"],
                "available_at": item["available_at"],
                "source_url": item["source_locator"],
                "metadata": {
                    "series_id": item["series_id"],
                    "series_name": item["series_name"],
                    "vintage_at": item["vintage_at"],
                    "availability_basis": item[
                        "availability_basis"
                    ],
                    "source_artifact_sha256": source_audit[
                        "sha256"
                    ],
                    "source_row_sha256": item["source_row_sha256"],
                    "release_at": None,
                },
            }
            for item in normalized["selected_observations"]
        }
        structured_audit = audit_structured_context(
            fundamentals={},
            macro=macro,
            sector_data={},
            as_of=as_of_dt.isoformat(),
        )
        accepted_macro = structured_audit["accepted_context"]["macro"]
        status = (
            "macro_evidence_ready_with_exclusions"
            if accepted_macro and normalized["exclusions"]
            else "macro_evidence_ready"
            if accepted_macro
            else "blocked_no_admissible_macro_evidence"
        )
        payload.update(
            {
                "status": status,
                "summary": {
                    "source_row_count": len(frame),
                    "eligible_row_count": normalized[
                        "eligible_row_count"
                    ],
                    "selected_series_count": len(
                        normalized["selected_observations"]
                    ),
                    "not_selected_eligible_row_count": normalized[
                        "not_selected_eligible_row_count"
                    ],
                    "excluded_row_count": len(
                        normalized["exclusions"]
                    ),
                    "accepted_series_count": structured_audit[
                        "accepted_count"
                    ],
                    "accepted_fingerprint": structured_audit[
                        "accepted_fingerprint"
                    ],
                    "reason_counts": normalized["reason_counts"],
                    "can_enter_market_context_review": bool(
                        accepted_macro
                    ),
                    "can_become_pipeline_feature": False,
                    "can_influence_prediction": False,
                    "can_trade": False,
                },
                "schema_mapping": normalized["schema_mapping"],
                "selected_observations": normalized[
                    "selected_observations"
                ],
                "exclusions": normalized["exclusions"],
                "structured_context_audit": {
                    key: value
                    for key, value in structured_audit.items()
                    if key
                    not in {
                        "accepted_context",
                        "accepted_observations",
                    }
                },
                "market_context_fragment": {
                    "as_of": as_of_dt.isoformat(),
                    "macro": accepted_macro,
                    "metadata": {
                        "saved_macro_producer_run_id": run_id,
                        "saved_macro_source_sha256": source_audit[
                            "sha256"
                        ],
                        "saved_macro_accepted_fingerprint": (
                            structured_audit["accepted_fingerprint"]
                        ),
                    },
                },
                "integration_boundary": {
                    "review_only": True,
                    "raw_source_retained_separately": True,
                    "realtime_start_semantics": (
                        "conservative snapshot-vintage availability, not "
                        "asserted original release time"
                    ),
                    "missing_vintage_fallback_to_file_mtime": False,
                    "missing_unit_inference_allowed": False,
                    "pipeline_feature_promotion_allowed": False,
                    "automatic_agent_context_mutation": False,
                },
                "safety": _safety(),
            }
        )
        return self._finish(payload, save=save)

    def _finish(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload.setdefault("integration_boundary", {})
        payload.setdefault("safety", _safety())
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_saved_macro_evidence_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return payload


def load_verified_macro_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("producer_contract") != SAVED_MACRO_PRODUCER_CONTRACT:
        raise ValueError("unsupported saved macro producer contract")
    if payload.get("status") not in {
        "macro_evidence_ready",
        "macro_evidence_ready_with_exclusions",
    }:
        raise ValueError("saved macro producer artifact is not ready")
    summary = payload.get("summary", {})
    safety = payload.get("safety", {})
    if (
        summary.get("can_enter_market_context_review") is not True
        or summary.get("can_trade") is not False
        or safety.get("review_only") is not True
        or safety.get("pipeline_run_performed") is not False
        or safety.get("live_execution_performed") is not False
    ):
        raise ValueError("saved macro producer safety boundary is invalid")

    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("saved macro producer fragment is missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("saved macro fragment as_of is invalid")
    input_as_of = parse_timezone_aware(
        payload.get("inputs", {}).get("as_of")
    )
    if input_as_of != fragment_as_of:
        raise ValueError("saved macro fragment as_of does not match input")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError(
                "saved macro fragment does not match expected as_of"
            )

    source = Path(payload.get("source_provenance", {}).get("path", ""))
    expected_source_sha = payload.get("source_provenance", {}).get(
        "sha256"
    )
    if (
        not source.exists()
        or not expected_source_sha
        or _sha256_file(source) != expected_source_sha
    ):
        raise ValueError("saved macro source artifact hash mismatch")
    registry_source = Path(
        payload.get("registry", {}).get("path", "")
    )
    expected_registry_sha = payload.get("registry", {}).get("sha256")
    if (
        not registry_source.exists()
        or not expected_registry_sha
        or _sha256_file(registry_source) != expected_registry_sha
    ):
        raise ValueError("saved macro registry hash mismatch")

    macro = fragment.get("macro")
    if not isinstance(macro, dict):
        raise ValueError("saved macro fragment payload is invalid")
    audit = audit_structured_context(
        fundamentals={},
        macro=macro,
        sector_data={},
        as_of=fragment_as_of.isoformat(),
    )
    expected_fingerprint = summary.get("accepted_fingerprint")
    if (
        audit["accepted_count"]
        != summary.get("accepted_series_count")
        or audit["accepted_fingerprint"] != expected_fingerprint
        or audit["excluded_count"] != 0
    ):
        raise ValueError(
            "saved macro fragment fingerprint or evidence count mismatch"
        )
    return {
        "as_of": fragment_as_of.isoformat(),
        "macro": audit["accepted_context"]["macro"],
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_macro_producer_artifact_path": str(path),
            "saved_macro_producer_artifact_sha256": _sha256_file(path),
            "saved_macro_verified": True,
        },
    }


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    as_of: Any,
    registry: dict[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    observation_column = _first_column(frame, OBSERVATION_FIELDS)
    series_column = _first_column(frame, SERIES_FIELDS)
    availability_column = _first_column(
        frame,
        AVAILABILITY_FIELDS,
    )
    value_column = "value" if "value" in frame.columns else None
    schema_mapping = {
        "observation_column": observation_column,
        "series_column": series_column,
        "value_column": value_column,
        "availability_column": availability_column,
    }
    missing = [
        name
        for name, column in schema_mapping.items()
        if column is None
    ]
    if missing:
        return {
            "schema_mapping": schema_mapping,
            "eligible_row_count": 0,
            "not_selected_eligible_row_count": 0,
            "selected_observations": [],
            "exclusions": [
                {
                    "index": None,
                    "status": "excluded",
                    "reasons": [
                        f"macro_schema_missing_{name}"
                        for name in missing
                    ],
                }
            ],
            "reason_counts": {
                f"macro_schema_missing_{name}": 1 for name in missing
            },
        }

    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for index, row in frame.iterrows():
        series_id = str(row.get(series_column) or "").strip()
        observation_at = _parse_timestamp(
            row.get(observation_column),
            end_of_day_for_date=False,
        )
        available_at = _parse_timestamp(
            row.get(availability_column),
            end_of_day_for_date=True,
        )
        value = _finite_float(row.get(value_column))
        series_metadata = registry.get(series_id)
        reasons: list[str] = []
        if not series_id:
            reasons.append("macro_series_id_missing")
        if observation_at is None:
            reasons.append("macro_observation_time_missing_or_invalid")
        elif observation_at > as_of:
            reasons.append("macro_observation_after_as_of")
        if available_at is None:
            reasons.append("macro_vintage_time_missing_or_invalid")
        elif available_at > as_of:
            reasons.append("macro_vintage_after_as_of")
        if value is None:
            reasons.append("macro_value_missing_or_invalid")
        if not isinstance(series_metadata, dict):
            reasons.append("macro_series_registry_entry_missing")
            series_metadata = {}
        unit = series_metadata.get("unit")
        if not unit:
            reasons.append("macro_series_unit_missing")
        context_key = str(
            series_metadata.get("context_key") or ""
        ).strip()
        if not context_key:
            reasons.append("macro_context_key_missing")
        source_locator = series_metadata.get(
            "source_url",
            (
                f"https://fred.stlouisfed.org/series/{series_id}"
                if series_id
                else None
            ),
        )
        if not source_locator:
            reasons.append("macro_source_locator_missing")
        canonical = {
            "series_id": series_id,
            "observation_at": (
                observation_at.isoformat()
                if observation_at is not None
                else None
            ),
            "available_at": (
                available_at.isoformat()
                if available_at is not None
                else None
            ),
            "value": value,
            "source_artifact_sha256": source_sha256,
        }
        row_sha = _canonical_sha256(canonical)
        if row_sha in seen_rows:
            reasons.append("duplicate_macro_source_row")
        if reasons:
            exclusions.append(
                {
                    "index": _json_scalar(index),
                    "series_id": series_id or None,
                    "status": "excluded",
                    "reasons": sorted(set(reasons)),
                    "source_row_sha256": row_sha,
                }
            )
            continue
        seen_rows.add(row_sha)
        eligible.append(
            {
                "series_id": series_id,
                "context_key": context_key,
                "series_name": series_metadata.get(
                    "name",
                    series_id,
                ),
                "value": value,
                "unit": str(unit),
                "period": observation_at.date().isoformat(),
                "observation_at": observation_at.isoformat(),
                "available_at": available_at.isoformat(),
                "vintage_at": available_at.isoformat(),
                "availability_basis": (
                    f"{availability_column}_conservative_end_of_day"
                    if _is_date_only(row.get(availability_column))
                    else availability_column
                ),
                "source_locator": str(source_locator),
                "source_row_sha256": row_sha,
            }
        )

    selected_by_series: dict[str, dict[str, Any]] = {}
    for item in sorted(
        eligible,
        key=lambda value: (
            value["series_id"],
            value["observation_at"],
            value["available_at"],
            value["source_row_sha256"],
        ),
    ):
        selected_by_series[item["series_id"]] = item
    selected = [
        selected_by_series[key] for key in sorted(selected_by_series)
    ]
    context_key_counts = Counter(
        item["context_key"] for item in selected
    )
    duplicate_context_keys = {
        key for key, count in context_key_counts.items() if count > 1
    }
    if duplicate_context_keys:
        retained: list[dict[str, Any]] = []
        for item in selected:
            if item["context_key"] in duplicate_context_keys:
                exclusions.append(
                    {
                        "index": None,
                        "series_id": item["series_id"],
                        "status": "excluded",
                        "reasons": [
                            "duplicate_macro_context_key"
                        ],
                        "source_row_sha256": item[
                            "source_row_sha256"
                        ],
                    }
                )
            else:
                retained.append(item)
        selected = retained
    reason_counts = Counter(
        reason
        for exclusion in exclusions
        for reason in exclusion["reasons"]
    )
    return {
        "schema_mapping": schema_mapping,
        "eligible_row_count": len(eligible),
        "not_selected_eligible_row_count": (
            len(eligible) - len(selected)
        ),
        "selected_observations": selected,
        "exclusions": exclusions,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def render_saved_macro_evidence_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Saved Macro Evidence Producer",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Source: `{payload.get('inputs', {}).get('source_path')}`",
        f"- Source SHA256: `{payload.get('source_provenance', {}).get('sha256')}`",
        f"- As-of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Source rows: {summary.get('source_row_count', 0)}",
        f"- Eligible rows: {summary.get('eligible_row_count', 0)}",
        f"- Selected series: {summary.get('selected_series_count', 0)}",
        f"- Accepted series: {summary.get('accepted_series_count', 0)}",
        f"- Excluded rows: {summary.get('excluded_row_count', 0)}",
        f"- Accepted fingerprint: `{summary.get('accepted_fingerprint')}`",
        f"- Can trade: {summary.get('can_trade', False)}",
        "",
        "## Selected Observations",
        "",
    ]
    selected = payload.get("selected_observations", [])
    if selected:
        lines.extend(
            (
                f"- `{item['series_id']}` value=`{item['value']}` "
                f"context_key=`{item['context_key']}` "
                f"unit=`{item['unit']}` period=`{item['period']}` "
                f"available_at=`{item['available_at']}`"
            )
            for item in selected
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This artifact is review-only.",
            "- `realtime_start` is treated as conservative snapshot-vintage availability, not asserted original release time.",
            "- Missing vintage timestamps do not fall back to file modification time.",
            "- No pipeline feature, prediction, learning, configuration, paper, or trading action is produced.",
            "",
        ]
    )
    return "\n".join(lines)


def _source_audit(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "sha256": None,
            "captured_at": None,
        }
    captured_at = pd.Timestamp(
        path.stat().st_mtime,
        unit="s",
        tz="UTC",
    )
    return {
        "exists": True,
        "path": str(path),
        "sha256": _sha256_file(path),
        "captured_at": captured_at.isoformat(),
        "size_bytes": path.stat().st_size,
    }


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("records", "items", "data", "macro"):
                values = payload.get(key)
                if isinstance(values, list):
                    return pd.DataFrame(values)
        raise ValueError("JSON macro source has no record list")
    raise ValueError(f"unsupported macro source suffix: {suffix}")


def _parse_timestamp(
    value: Any,
    *,
    end_of_day_for_date: bool,
) -> Any | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
    except (ValueError, TypeError, OverflowError):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        return None
    if end_of_day_for_date and _is_date_only(value):
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(
            microseconds=1
        )
    return parsed.to_pydatetime()


def _is_date_only(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip())
    )


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    return next(
        (candidate for candidate in candidates if candidate in frame.columns),
        None,
    )


def _blocked_payload(
    status: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": {
            "source_row_count": 0,
            "eligible_row_count": 0,
            "selected_series_count": 0,
            "accepted_series_count": 0,
            "excluded_row_count": 0,
            "reason_counts": dict.fromkeys(reasons, 1),
            "can_enter_market_context_review": False,
            "can_become_pipeline_feature": False,
            "can_influence_prediction": False,
            "can_trade": False,
        },
        "selected_observations": [],
        "exclusions": [
            {
                "status": "excluded",
                "reasons": reasons,
            }
        ],
    }


def _safety() -> dict[str, bool]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "collector_run_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "paper_execution_performed": False,
        "live_execution_performed": False,
        "can_trade": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _run_id() -> str:
    return (
        "saved_macro_evidence_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )
