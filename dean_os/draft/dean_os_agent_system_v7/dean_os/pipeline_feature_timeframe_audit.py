from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.timeframe_lineage import timeframe_lineage_report


class PipelineFeatureTimeframeAudit:
    """Read-only audit of declared feature timeframe versus observed cadence."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_feature_timeframe_audit"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        features_path: str | Path,
        stage5_path: str | Path | None = None,
        tickers: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        feature_path = Path(features_path)
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        frame = pd.read_parquet(feature_path)
        required = {"ticker", "datetime"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Feature artifact missing audit columns: "
                + ", ".join(missing)
            )
        requested = sorted(
            {
                str(ticker).strip().upper()
                for ticker in (tickers or frame["ticker"].unique())
                if str(ticker).strip()
            }
        )
        selected = frame[
            frame["ticker"].astype(str).str.upper().isin(requested)
        ].copy()
        ticker_reports = []
        for ticker in requested:
            ticker_frame = selected[
                selected["ticker"].astype(str).str.upper().eq(ticker)
            ]
            if ticker_frame.empty:
                ticker_reports.append(
                    {
                        "ticker": ticker,
                        "status": "ticker_missing",
                        "row_count": 0,
                        "declared_timeframe_counts": {},
                        "lineage": {},
                    }
                )
                continue
            lineage = timeframe_lineage_report(ticker_frame)
            declared_column = next(
                (
                    column
                    for column in ("interval", "timeframe")
                    if column in ticker_frame.columns
                ),
                None,
            )
            declared_counts = (
                Counter(
                    str(value)
                    for value in ticker_frame[
                        declared_column
                    ].dropna()
                )
                if declared_column
                else Counter()
            )
            ticker_reports.append(
                {
                    "ticker": ticker,
                    "status": lineage.get("status"),
                    "row_count": int(len(ticker_frame)),
                    "first_observed_at": _timestamp_text(
                        ticker_frame["datetime"].min()
                    ),
                    "last_observed_at": _timestamp_text(
                        ticker_frame["datetime"].max()
                    ),
                    "datetime_timezone_aware": (
                        _timezone_aware_series(
                            ticker_frame["datetime"]
                        )
                    ),
                    "declared_timeframe_column": declared_column,
                    "declared_timeframe_counts": dict(
                        sorted(declared_counts.items())
                    ),
                    "lineage": lineage,
                }
            )

        mismatch_tickers = [
            item["ticker"]
            for item in ticker_reports
            if item.get("status")
            in {
                "timeframe_cadence_mismatch",
                "timeframe_cadence_ambiguous",
            }
        ]
        missing_tickers = [
            item["ticker"]
            for item in ticker_reports
            if item.get("status") == "ticker_missing"
        ]
        stage5_binding = _stage5_binding(
            Path(stage5_path) if stage5_path else None,
            feature_path=feature_path,
            requested_tickers=requested,
        )
        if mismatch_tickers:
            status = (
                "pipeline_feature_timeframe_audit_blocked_mismatch"
            )
        elif missing_tickers:
            status = (
                "pipeline_feature_timeframe_audit_blocked_missing_ticker"
            )
        else:
            status = "pipeline_feature_timeframe_audit_ready"
        payload = {
            "run_id": (
                "pipeline_feature_timeframe_audit_"
                + utc_now_iso().replace(":", "").replace("+", "")
            ),
            "created_at": utc_now_iso(),
            "mode": "pipeline_feature_timeframe_audit",
            "schema_version": (
                "dean_pipeline_feature_timeframe_audit_v1"
            ),
            "status": status,
            "inputs": {
                "features_path": str(feature_path),
                "features_sha256": _sha256(feature_path),
                "stage5_path": (
                    str(stage5_path) if stage5_path else None
                ),
                "stage5_sha256": stage5_binding.get("sha256"),
                "requested_tickers": requested,
            },
            "summary": {
                "requested_ticker_count": len(requested),
                "present_ticker_count": (
                    len(requested) - len(missing_tickers)
                ),
                "feature_row_count": int(len(frame)),
                "selected_feature_row_count": int(len(selected)),
                "timeframe_mismatch_ticker_count": len(
                    mismatch_tickers
                ),
                "timeframe_mismatch_tickers": mismatch_tickers,
                "missing_tickers": missing_tickers,
                "timezone_aware_ticker_count": sum(
                    item.get("datetime_timezone_aware") is True
                    for item in ticker_reports
                ),
                "can_use_for_stage4": not (
                    mismatch_tickers or missing_tickers
                ),
                "can_use_for_stage5": not (
                    mismatch_tickers or missing_tickers
                ),
                "can_trade": False,
            },
            "ticker_timeframe_reports": ticker_reports,
            "stage5_candidate_binding": stage5_binding,
            "required_next_actions": (
                [
                    (
                        "regenerate_stage2_stage3_from_saved_source_with_"
                        "cadence_validated_timeframes"
                    ),
                    (
                        "preserve_timezone_aware_datetime_and_feature_"
                        "artifact_hash_in_stage5_parent_lineage"
                    ),
                    (
                        "do_not_reuse_models_or_predictions_trained_on_"
                        "mislabeled_timeframe_features"
                    ),
                ]
                if mismatch_tickers
                else [
                    "bind_feature_sha256_into_future_stage4_stage5_artifacts"
                ]
            ),
            "safety": {
                "read_only": True,
                "supporting_audit_only": True,
                "data_mutation_performed": False,
                "pipeline_run_performed": False,
                "training_performed": False,
                "tuning_performed": False,
                "learning_write_performed": False,
                "can_promote_model": False,
                "can_create_ticker_forecast": False,
                "can_trade": False,
            },
        }
        if save:
            self._save(payload)
        return json_ready(payload)

    def _save(self, payload: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_json = self.output_dir / f"{payload['run_id']}.json"
        run_md = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(run_json),
            "markdown": str(run_md),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = (
            json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
            + "\n"
        )
        rendered_md = render_pipeline_feature_timeframe_audit(payload)
        for path in (run_json, latest_json):
            path.write_text(rendered_json, encoding="utf-8")
        for path in (run_md, latest_md):
            path.write_text(rendered_md, encoding="utf-8")


def render_pipeline_feature_timeframe_audit(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary") or {}
    binding = payload.get("stage5_candidate_binding") or {}
    lines = [
        "# DEAN-OS Pipeline Feature Timeframe Audit",
        "",
        f"- Status: `{payload.get('status')}`",
        (
            "- Feature rows: "
            f"{summary.get('selected_feature_row_count')} selected / "
            f"{summary.get('feature_row_count')} source"
        ),
        (
            "- Timeframe mismatches: "
            f"{summary.get('timeframe_mismatch_ticker_count')} "
            f"{summary.get('timeframe_mismatch_tickers')}"
        ),
        (
            "- Timezone-aware tickers: "
            f"{summary.get('timezone_aware_ticker_count')}/"
            f"{summary.get('requested_ticker_count')}"
        ),
        (
            "- Stage 5 relationship: "
            f"`{binding.get('relationship_status')}`"
        ),
        "",
        "## Tickers",
        "",
    ]
    for item in payload.get("ticker_timeframe_reports", []):
        lineage = item.get("lineage") or {}
        lines.append(
            f"- `{item.get('ticker')}` status=`{item.get('status')}` "
            f"declared=`{lineage.get('declared_timeframe')}` "
            f"observed=`{lineage.get('observed_timeframe')}` "
            f"rows={item.get('row_count')} "
            f"timezone_aware={item.get('datetime_timezone_aware')}"
        )
    lines.extend(["", "## Required Next Actions", ""])
    lines.extend(
        f"- {item}"
        for item in payload.get("required_next_actions", [])
    )
    return "\n".join(lines).strip() + "\n"


def _stage5_binding(
    path: Path | None,
    *,
    feature_path: Path,
    requested_tickers: list[str],
) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "relationship_status": "not_supplied",
            "can_assert_feature_parentage": False,
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("prediction_results")
    if not isinstance(predictions, dict):
        predictions = {}
    selected_predictions = [
        value
        for value in predictions.values()
        if isinstance(value, dict)
        and str(value.get("ticker") or "").upper()
        in requested_tickers
    ]
    models = payload.get("models_metadata")
    if not isinstance(models, dict):
        models = {}
    selected_models = [
        value
        for value in models.values()
        if isinstance(value, dict)
        and str(value.get("ticker") or "").upper()
        in requested_tickers
    ]
    explicit_feature_sha = (
        (payload.get("feature_artifact") or {}).get("sha256")
        if isinstance(payload.get("feature_artifact"), dict)
        else payload.get("features_sha256")
    )
    feature_sha = _sha256(feature_path)
    explicit_match = (
        bool(explicit_feature_sha)
        and explicit_feature_sha == feature_sha
    )
    same_directory = path.parent.resolve() == feature_path.parent.resolve()
    same_batch_name = str(payload.get("batch_name") or "") == (
        feature_path.parent.name
    )
    if explicit_match:
        relationship_status = "feature_parent_hash_verified"
    elif same_directory and same_batch_name:
        relationship_status = (
            "co_located_same_batch_candidate_not_hash_bound"
        )
    else:
        relationship_status = "feature_parent_unverified"
    return {
        "available": True,
        "path": str(path),
        "sha256": _sha256(path),
        "batch_name": payload.get("batch_name"),
        "stage5_created_at": payload.get("timestamp"),
        "source_prediction_context_count": len(predictions),
        "selected_prediction_context_count": len(
            selected_predictions
        ),
        "source_model_metadata_count": len(models),
        "selected_model_metadata_count": len(selected_models),
        "explicit_feature_sha256": explicit_feature_sha,
        "expected_feature_sha256": feature_sha,
        "same_directory": same_directory,
        "same_batch_name": same_batch_name,
        "relationship_status": relationship_status,
        "can_assert_feature_parentage": explicit_match,
        "can_clear_timeframe_mismatch": False,
        "can_promote_model": False,
        "can_create_ticker_forecast": False,
        "can_trade": False,
    }


def _timezone_aware_series(values: pd.Series) -> bool:
    parsed = pd.to_datetime(values, errors="coerce")
    return getattr(parsed.dt, "tz", None) is not None


def _timestamp_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "PipelineFeatureTimeframeAudit",
    "render_pipeline_feature_timeframe_audit",
]
