from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
)
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    timeframe_lineage_report,
)


class PipelineStage4ExactContextReview:
    """Run one hash-bound, development-only Stage 4 review context."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_stage4_exact_context_review"
        ),
    ):
        self.output_dir = Path(output_dir)

    def run(
        self,
        *,
        features_path: str | Path,
        targets_path: str | Path,
        batch_metadata_path: str | Path,
        feature_audit_path: str | Path,
        target_audit_path: str | Path,
        ticker: str,
        timeframe: str,
        target_name: str,
        min_train_rows: int = 160,
        validation_rows: int = 50,
        step_rows: int = 50,
        purge_rows: int = 5,
        max_folds: int = 3,
        max_features: int = 30,
        save: bool = True,
    ) -> dict[str, Any]:
        feature_path = Path(features_path)
        target_path = Path(targets_path)
        metadata_path = Path(batch_metadata_path)
        feature_review_path = Path(feature_audit_path)
        target_review_path = Path(target_audit_path)
        for path in (
            feature_path,
            target_path,
            metadata_path,
            feature_review_path,
            target_review_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        ticker = str(ticker).strip().upper()
        timeframe = normalize_timeframe(timeframe)
        if not ticker or not timeframe or not target_name:
            raise ValueError(
                "Ticker, timeframe, and target_name are required"
            )
        metadata = _load_json(metadata_path)
        feature_audit = _load_json(feature_review_path)
        target_audit = _load_json(target_review_path)
        lineage = _verify_parent_lineage(
            feature_path=feature_path,
            target_path=target_path,
            metadata_path=metadata_path,
            feature_audit_path=feature_review_path,
            target_audit_path=target_review_path,
            metadata=metadata,
            feature_audit=feature_audit,
            target_audit=target_audit,
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
        )
        context_frame = _assemble_exact_context(
            pd.read_parquet(feature_path),
            pd.read_parquet(target_path),
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
        )
        cadence = timeframe_lineage_report(
            context_frame[
                ["ticker", "datetime", "interval"]
            ],
            declared_timeframe=timeframe,
        )
        if cadence.get("status") != "timeframe_cadence_verified":
            raise ValueError(
                "Exact Stage 4 context failed cadence verification"
            )
        config = WalkForwardValidationConfig(
            min_train_rows=min_train_rows,
            validation_rows=validation_rows,
            step_rows=step_rows,
            purge_rows=purge_rows,
            max_folds=max_folds,
            max_features=max_features,
        )
        context_report = {
            "status": "exact_single_timeframe_context",
            "join_direction": "not_applicable_single_timeframe",
            "allow_future_context": False,
            "summary": {
                "base_context_count": 1,
                "output_rows": len(context_frame),
                "future_context_violations": 0,
                "row_identity_preserved": True,
            },
            "base_contexts": [
                {
                    "base_timeframe": timeframe,
                    "row_count": len(context_frame),
                    "cadence_status": cadence.get("status"),
                }
            ],
        }
        candidate = PipelineWalkForwardValidationEvaluator(
            config
        ).evaluate(
            context_frame,
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
            timeframe_context_report=context_report,
            source_lineage=lineage,
        )
        metrics = candidate["metrics"]
        failed_checks = [
            check_name
            for check_name, passed in metrics["checks"].items()
            if not passed
        ]
        run_id = _run_id(
            "pipeline_stage4_exact_context_review"
        )
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_stage4_exact_context_review",
            "schema_version": (
                "dean_pipeline_stage4_exact_context_review_v1"
            ),
            "status": candidate["contract_status"],
            "scope": {
                "ticker": ticker,
                "timeframe": timeframe,
                "target_name": target_name,
                "context_row_count": len(context_frame),
            },
            "validation_config": asdict(config),
            "parent_lineage": lineage,
            "timeframe_lineage": cadence,
            "walk_forward_candidate": candidate,
            "summary": {
                "fold_count": metrics["fold_count"],
                "mean_validation_balanced_accuracy": metrics[
                    "mean_validation_balanced_accuracy"
                ],
                "mean_train_validation_gap": metrics[
                    "mean_train_validation_gap"
                ],
                "mean_feature_stability_score": metrics[
                    "mean_feature_stability_score"
                ],
                "contract_passed": metrics["contract_passed"],
                "failed_contract_checks": failed_checks,
                "can_use_as_locked_test_evidence": False,
                "can_promote_model": False,
                "can_write_production_config": False,
                "can_trade": False,
            },
            "safety": {
                "review_only": True,
                "development_train_validation_only": True,
                "test_rows_loaded": 0,
                "model_persisted": False,
                "hyperparameter_search_performed": False,
                "learning_write_performed": False,
                "decision_influence": False,
                "can_trade": False,
            },
        }
        payload = json_ready(payload)
        if save:
            paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_stage4_exact_context_review(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = paths
        return payload


def _verify_parent_lineage(
    *,
    feature_path: Path,
    target_path: Path,
    metadata_path: Path,
    feature_audit_path: Path,
    target_audit_path: Path,
    metadata: dict[str, Any],
    feature_audit: dict[str, Any],
    target_audit: dict[str, Any],
    ticker: str,
    timeframe: str,
    target_name: str,
) -> dict[str, Any]:
    if feature_audit.get("mode") != "pipeline_feature_timeframe_audit":
        raise ValueError("Invalid feature timeframe audit mode")
    if (
        feature_audit.get("status")
        != "pipeline_feature_timeframe_audit_ready"
    ):
        raise ValueError("Feature timeframe audit is not ready")
    if target_audit.get("mode") != "pipeline_target_readiness_audit":
        raise ValueError("Invalid target readiness audit mode")
    if target_audit.get("status") not in {
        "pipeline_target_readiness_ready",
        "pipeline_target_readiness_ready_with_gaps",
    }:
        raise ValueError("Target readiness audit is not ready")

    feature_sha = _sha256(feature_path)
    target_sha = _sha256(target_path)
    metadata_lineage = metadata.get("lineage") or {}
    feature_audit_sha = (
        (feature_audit.get("inputs") or {}).get(
            "features_sha256"
        )
    )
    target_lineage = target_audit.get("lineage_bindings") or {}
    target_audit_target_sha = target_lineage.get("target_sha256")
    target_audit_feature_sha = (
        (target_lineage.get("feature_artifact") or {}).get(
            "sha256"
        )
    )
    feature_bindings = {
        "metadata_feature_sha": metadata_lineage.get(
            "features_sha256"
        ),
        "feature_audit_sha": feature_audit_sha,
        "target_audit_feature_sha": target_audit_feature_sha,
    }
    target_bindings = {
        "metadata_target_sha": metadata_lineage.get(
            "targets_sha256"
        ),
        "target_audit_target_sha": target_audit_target_sha,
    }
    if any(value != feature_sha for value in feature_bindings.values()):
        raise ValueError("Feature parent SHA bindings do not match")
    if any(value != target_sha for value in target_bindings.values()):
        raise ValueError("Target parent SHA bindings do not match")

    target_report = next(
        (
            item
            for item in target_audit.get("target_reports", [])
            if item.get("target_name") == target_name
        ),
        None,
    )
    if (
        not isinstance(target_report, dict)
        or target_report.get("status") != "target_ready"
    ):
        raise ValueError(
            f"Target readiness missing for {target_name}"
        )
    if target_report.get("applies_to_timeframe") is not True:
        raise ValueError("Target does not apply to requested timeframe")
    if ticker not in (target_report.get("per_ticker") or {}):
        raise ValueError("Target audit does not cover requested ticker")

    return {
        "features": {
            "path": str(feature_path),
            "sha256": feature_sha,
        },
        "targets": {
            "path": str(target_path),
            "sha256": target_sha,
        },
        "batch_metadata": {
            "path": str(metadata_path),
            "sha256": _sha256(metadata_path),
        },
        "feature_timeframe_audit": {
            "path": str(feature_audit_path),
            "sha256": _sha256(feature_audit_path),
        },
        "target_readiness_audit": {
            "path": str(target_audit_path),
            "sha256": _sha256(target_audit_path),
        },
        "scope": {
            "ticker": ticker,
            "timeframe": timeframe,
            "target_name": target_name,
        },
        "all_parent_hashes_verified": True,
    }


def _assemble_exact_context(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    ticker: str,
    timeframe: str,
    target_name: str,
) -> pd.DataFrame:
    identity = ["ticker", "datetime", "interval"]
    for name, frame in (("features", features), ("targets", targets)):
        missing = [
            column
            for column in identity
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"{name} missing identity columns: {missing}"
            )
        if frame.duplicated(identity).any():
            raise ValueError(f"{name} has duplicate exact identities")
    if target_name not in targets.columns:
        raise ValueError(f"Target column is missing: {target_name}")

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[
            frame["ticker"].astype(str).str.upper().eq(ticker)
            & frame["interval"]
            .map(normalize_timeframe)
            .eq(timeframe)
        ].copy()

    feature_context = select(features)
    target_context = select(targets)[identity + [target_name]]
    joined = feature_context.merge(
        target_context,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(feature_context):
        raise ValueError(
            "Feature/target exact identity coverage is incomplete"
        )
    if getattr(joined["datetime"].dt, "tz", None) is None:
        raise ValueError("Exact Stage 4 datetime timezone is unresolved")
    return joined


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


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


def render_stage4_exact_context_review(
    payload: dict[str, Any],
) -> str:
    scope = payload.get("scope") or {}
    summary = payload.get("summary") or {}
    return "\n".join(
        [
            "# Stage 4 Exact-Context Review",
            "",
            f"- Status: `{payload.get('status')}`",
            (
                f"- Context: `{scope.get('ticker')}/"
                f"{scope.get('timeframe')}/"
                f"{scope.get('target_name')}`"
            ),
            f"- Rows: {scope.get('context_row_count')}",
            f"- Folds: {summary.get('fold_count')}",
            (
                "- Validation balanced accuracy: "
                f"{summary.get('mean_validation_balanced_accuracy')}"
            ),
            (
                "- Train-validation gap: "
                f"{summary.get('mean_train_validation_gap')}"
            ),
            (
                "- Feature stability: "
                f"{summary.get('mean_feature_stability_score')}"
            ),
            f"- Contract passed: `{summary.get('contract_passed')}`",
            (
                "- Failed checks: "
                f"`{', '.join(summary.get('failed_contract_checks') or [])}`"
            ),
            "- Can promote model: `False`",
            "- Can trade: `False`",
        ]
    ).strip() + "\n"


__all__ = [
    "PipelineStage4ExactContextReview",
    "render_stage4_exact_context_review",
]
