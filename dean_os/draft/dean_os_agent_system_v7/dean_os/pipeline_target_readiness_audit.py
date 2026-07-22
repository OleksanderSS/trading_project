from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    timeframe_lineage_report,
)
from src.targets.timeframe_contract import target_applies_to_timeframe


class PipelineTargetReadinessAudit:
    """Audit target semantics and coverage before Stage 4."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_target_readiness_audit"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        targets_path: str | Path,
        tickers: list[str],
        timeframe: str,
        features_path: str | Path | None = None,
        batch_metadata_path: str | Path | None = None,
        target_registry_path: str | Path = "src/config/targets.yaml",
        minimum_non_null_ratio: float = 0.50,
        save: bool = True,
    ) -> dict[str, Any]:
        target_path = Path(targets_path)
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        frame = pd.read_parquet(target_path)
        requested_tickers = sorted(
            {
                str(ticker).strip().upper()
                for ticker in tickers
                if str(ticker).strip()
            }
        )
        resolved_timeframe = normalize_timeframe(timeframe)
        if not requested_tickers or not resolved_timeframe:
            raise ValueError("Tickers and timeframe are required")
        if not 0.0 < minimum_non_null_ratio <= 1.0:
            raise ValueError(
                "minimum_non_null_ratio must be in (0, 1]"
            )

        registry_path = Path(target_registry_path)
        registry = _load_registry(registry_path)
        identity_checks = _identity_checks(
            frame,
            tickers=requested_tickers,
            timeframe=resolved_timeframe,
        )
        target_columns = sorted(
            column
            for column in frame.columns
            if str(column).startswith("target_")
        )
        reports = [
            _target_report(
                frame,
                target_name=target_name,
                config=registry.get(target_name),
                tickers=requested_tickers,
                timeframe=resolved_timeframe,
                minimum_non_null_ratio=minimum_non_null_ratio,
            )
            for target_name in target_columns
        ]
        lineage = _lineage_bindings(
            target_path=target_path,
            features_path=(
                Path(features_path) if features_path else None
            ),
            metadata_path=(
                Path(batch_metadata_path)
                if batch_metadata_path
                else None
            ),
        )
        hard_blocking_reasons = [
            check["code"]
            for check in identity_checks
            if check["status"] == "fail"
        ]
        target_exclusions = [
            f"target_not_ready:{report['target_name']}"
            for report in reports
            if report["status"] != "target_ready"
        ]
        hard_blocking_reasons.extend(lineage["errors"])
        if not target_columns:
            hard_blocking_reasons.append("target_columns_missing")
        hard_blocking_reasons = sorted(set(hard_blocking_reasons))
        target_exclusions = sorted(set(target_exclusions))
        ready_count = sum(
            report["status"] == "target_ready"
            for report in reports
        )
        can_use_for_stage4 = not hard_blocking_reasons and ready_count > 0
        all_targets_ready = can_use_for_stage4 and ready_count == len(reports)
        blocked = not can_use_for_stage4
        blocking_reasons = list(hard_blocking_reasons)
        if blocked:
            blocking_reasons.extend(target_exclusions)
        ready_target_names = [
            report["target_name"]
            for report in reports
            if report["status"] == "target_ready"
        ]
        blocked_target_names = [
            report["target_name"]
            for report in reports
            if report["status"] != "target_ready"
        ]
        payload = {
            "run_id": _run_id(
                "pipeline_target_readiness_audit"
            ),
            "created_at": utc_now_iso(),
            "mode": "pipeline_target_readiness_audit",
            "schema_version": (
                "dean_pipeline_target_readiness_audit_v1"
            ),
            "status": (
                "pipeline_target_readiness_ready"
                if all_targets_ready
                else "pipeline_target_readiness_ready_with_gaps"
                if can_use_for_stage4
                else "pipeline_target_readiness_blocked"
            ),
            "source_artifact": {
                "path": str(target_path),
                "sha256": _sha256(target_path),
                "row_count": len(frame),
            },
            "scope": {
                "tickers": requested_tickers,
                "timeframe": resolved_timeframe,
                "minimum_non_null_ratio": minimum_non_null_ratio,
            },
            "identity_checks": identity_checks,
            "lineage_bindings": lineage,
            "target_reports": reports,
            "summary": {
                "target_count": len(reports),
                "ready_target_count": ready_count,
                "blocked_target_count": len(reports) - ready_count,
                "ready_target_names": ready_target_names,
                "blocked_target_names": blocked_target_names,
                "ticker_count": len(requested_tickers),
                "row_count": len(frame),
                "can_use_for_stage4": can_use_for_stage4,
                "can_use_for_stage5": False,
                "can_promote_model": False,
                "can_create_ticker_forecast": False,
                "can_trade": False,
            },
            "blocking_reasons": blocking_reasons,
            "target_exclusions": target_exclusions,
            "safety": {
                "review_only": True,
                "decision_influence": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "can_trade": False,
            },
        }
        payload = json_ready(payload)
        if save:
            paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_pipeline_target_readiness_audit(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = paths
        return payload


def _identity_checks(
    frame: pd.DataFrame,
    *,
    tickers: list[str],
    timeframe: str,
) -> list[dict[str, Any]]:
    required = {"ticker", "datetime", "interval"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return [
            _check(
                "fail",
                "identity_columns_missing",
                ", ".join(missing),
            )
        ]
    checks = []
    present = sorted(
        set(frame["ticker"].astype(str).str.upper())
    )
    checks.append(
        _check(
            "pass" if present == tickers else "fail",
            "ticker_coverage",
            f"present={present}, requested={tickers}",
        )
    )
    timezone = getattr(
        pd.to_datetime(frame["datetime"], errors="coerce").dt,
        "tz",
        None,
    )
    checks.append(
        _check(
            "pass" if timezone is not None else "fail",
            "datetime_timezone",
            f"timezone={timezone}",
        )
    )
    lineage = timeframe_lineage_report(
        frame[["ticker", "datetime", "interval"]],
        declared_timeframe=timeframe,
    )
    checks.append(
        _check(
            (
                "pass"
                if lineage.get("status")
                == "timeframe_cadence_verified"
                else "fail"
            ),
            "timeframe_cadence",
            (
                f"declared={timeframe}, "
                f"observed={lineage.get('observed_timeframe')}"
            ),
        )
    )
    duplicate_count = int(
        frame.duplicated(
            ["ticker", "datetime", "interval"],
            keep=False,
        ).sum()
    )
    checks.append(
        _check(
            "pass" if duplicate_count == 0 else "fail",
            "unique_target_identity",
            f"duplicate_rows={duplicate_count}",
        )
    )
    return checks


def _target_report(
    frame: pd.DataFrame,
    *,
    target_name: str,
    config: dict[str, Any] | None,
    tickers: list[str],
    timeframe: str,
    minimum_non_null_ratio: float,
) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {
            "target_name": target_name,
            "status": "target_registry_missing",
            "blocking_reasons": ["target_registry_missing"],
        }
    canonical = {"name": target_name, **config}
    applies = target_applies_to_timeframe(
        canonical,
        timeframe,
    )
    series = frame[target_name]
    per_ticker = {}
    coverage_ready = True
    class_ready = True
    target_type = str(config.get("type") or "")
    for ticker in tickers:
        selected = series.loc[
            frame["ticker"].astype(str).str.upper().eq(ticker)
        ]
        non_null = int(selected.notna().sum())
        total = len(selected)
        ratio = non_null / total if total else 0.0
        unique = int(selected.nunique(dropna=True))
        per_ticker[ticker] = {
            "row_count": total,
            "non_null_count": non_null,
            "non_null_ratio": round(ratio, 6),
            "unique_value_count": unique,
        }
        coverage_ready &= ratio >= minimum_non_null_ratio
        if target_type.startswith("classification"):
            class_ready &= unique >= 2
    blockers = []
    if not applies:
        blockers.append("target_not_applicable_to_timeframe")
    if not coverage_ready:
        blockers.append("insufficient_non_null_coverage")
    if not class_ready:
        blockers.append("classification_has_fewer_than_two_classes")
    params = config.get("params") or {}
    return {
        "target_name": target_name,
        "status": "target_ready" if not blockers else "target_blocked",
        "target_type": target_type,
        "target_unit": _target_unit(target_name, target_type),
        "source_timeframe": (
            params.get("source_timeframe") or timeframe
        ),
        "semantic_horizon": params.get("horizon"),
        "configured_shift": params.get("shift"),
        "applies_to_timeframe": applies,
        "per_ticker": per_ticker,
        "blocking_reasons": blockers,
    }


def _lineage_bindings(
    *,
    target_path: Path,
    features_path: Path | None,
    metadata_path: Path | None,
) -> dict[str, Any]:
    errors = []
    metadata = {}
    if metadata_path is not None:
        if not metadata_path.is_file():
            errors.append("batch_metadata_missing")
        else:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
    lineage = metadata.get("lineage") or {}
    actual_target_sha = _sha256(target_path)
    expected_target_sha = lineage.get("targets_sha256")
    if expected_target_sha and expected_target_sha != actual_target_sha:
        errors.append("target_sha_mismatch")

    feature_binding = {}
    if features_path is not None:
        if not features_path.is_file():
            errors.append("features_artifact_missing")
        else:
            actual_feature_sha = _sha256(features_path)
            expected_feature_sha = lineage.get("features_sha256")
            if (
                expected_feature_sha
                and expected_feature_sha != actual_feature_sha
            ):
                errors.append("feature_sha_mismatch")
            feature_binding = {
                "path": str(features_path),
                "sha256": actual_feature_sha,
                "expected_sha256": expected_feature_sha,
                "hash_verified": (
                    bool(expected_feature_sha)
                    and expected_feature_sha == actual_feature_sha
                ),
            }
    return {
        "batch_metadata_path": (
            str(metadata_path) if metadata_path else None
        ),
        "target_sha256": actual_target_sha,
        "expected_target_sha256": expected_target_sha,
        "target_hash_verified": (
            bool(expected_target_sha)
            and expected_target_sha == actual_target_sha
        ),
        "feature_artifact": feature_binding,
        "errors": errors,
    }


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("Target registry has no targets mapping")
    return targets


def _target_unit(target_name: str, target_type: str) -> str:
    if target_type.startswith("classification"):
        return "class_label"
    if "return" in target_name:
        return "return_fraction"
    if "volatility" in target_name:
        return "volatility_fraction"
    return "numeric_target_value"


def _check(
    status: str,
    code: str,
    message: str,
) -> dict[str, str]:
    return {
        "status": status,
        "code": code,
        "message": message,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        f"{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    )


def render_pipeline_target_readiness_audit(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# Pipeline Target Readiness Audit",
        "",
        f"- Status: `{payload.get('status')}`",
        (
            "- Ready targets: "
            f"{summary.get('ready_target_count')}/"
            f"{summary.get('target_count')}"
        ),
        f"- Can use for Stage 4: `{summary.get('can_use_for_stage4')}`",
        "- Can trade: `False`",
        "",
        "## Targets",
        "",
    ]
    lines.extend(
        (
            f"- `{item.get('target_name')}` "
            f"status=`{item.get('status')}` "
            f"type=`{item.get('target_type')}` "
            f"horizon=`{item.get('semantic_horizon')}`"
        )
        for item in payload.get("target_reports", [])
    )
    return "\n".join(lines).strip() + "\n"


__all__ = [
    "PipelineTargetReadinessAudit",
    "render_pipeline_target_readiness_audit",
]
