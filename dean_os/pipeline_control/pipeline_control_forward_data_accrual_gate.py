from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

EXPECTED_MINUTES = {"15m": 15.0, "60m": 60.0, "1d": 1440.0}
RETURN_LIMITS = {"15m": 0.25, "60m": 0.50, "1d": 0.75}
REQUIRED_COLUMNS = {
    "datetime",
    "ticker",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


class PipelineControlForwardDataAccrualGate:
    """Validate a saved source against a registered forward-development boundary."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_control_forward_data_accrual_gate_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        accrual_plan_json: str | Path,
        source_path: str | Path,
        save: bool = True,
    ) -> dict[str, Any]:
        plan_path = Path(accrual_plan_json)
        candidate_path = Path(source_path)
        plan_payload = _load_json(plan_path)
        source = _inspect_source(candidate_path)
        checks, context = _checks(plan_payload, source)
        ready = all(check["status"] == "pass" for check in checks)
        status = (
            "forward_development_artifact_ready"
            if ready
            else "blocked_forward_development_artifact"
        )
        run_id = _run_id("pipeline_control_forward_data_accrual_gate")
        eligible_artifact = (
            _eligible_artifact(
                plan_payload=plan_payload,
                plan_path=plan_path,
                source=source,
                source_path=candidate_path,
                context=context,
            )
            if ready
            else None
        )
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_forward_data_accrual_gate",
            "inputs": {
                "accrual_plan_json": str(plan_path),
                "source_path": str(candidate_path),
            },
            "summary": {
                "gate_status": status,
                "check_pass_count": sum(
                    check["status"] == "pass" for check in checks
                ),
                "check_fail_count": sum(
                    check["status"] == "fail" for check in checks
                ),
                "candidate_new_row_count": context.get(
                    "candidate_new_row_count", 0
                ),
                "eligible_new_row_count": (
                    context.get("candidate_new_row_count", 0) if ready else 0
                ),
                "can_supply_next_development_run": ready,
                "can_use_as_locked_test_evidence": False,
                "can_call_virgin_holdout": False,
                "can_train": False,
                "can_promote_model": False,
                "can_write_production_config": False,
                "can_trade": False,
            },
            "checks": checks,
            "source_inspection": _source_preview(source),
            "eligible_development_artifact": eligible_artifact,
            "next_runner_inputs": {
                "source_path": str(candidate_path) if ready else None,
                "ticker": context.get("ticker") if ready else None,
                "timeframe": context.get("timeframe") if ready else None,
                "target_name": context.get("target_name") if ready else None,
                "start_exclusive": context.get("watermark") if ready else None,
                "accrual_gate_json": (
                    str(self.output_dir / "latest.json") if ready and save else None
                ),
                "can_invoke_development_walk_forward": ready,
            },
            "next_step": (
                "Use only rows strictly after start_exclusive in one predeclared "
                "development refresh; preserve the source SHA and this gate lineage."
                if ready
                else (
                    "Do not pass this source to development walk-forward. Acquire "
                    "a new immutable artifact that satisfies every accrual check."
                )
            ),
            "explicit_non_actions": [
                "No target or feature column is generated.",
                "No test or past-evaluation artifact is loaded.",
                "No collector or external API is started.",
                "No model training, evaluation, replay, backtest, or tuning is run.",
                "No locked holdout, production config, recommendation, order, or trade is created.",
            ],
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_forward_data_accrual_gate_markdown(payload),
                run_id=run_id,
            )
            payload["saved_paths"] = saved_paths
            if ready:
                payload["next_runner_inputs"]["accrual_gate_json"] = saved_paths[
                    "latest_json"
                ]
        return json_ready(payload)


def render_forward_data_accrual_gate_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    artifact = payload.get("eligible_development_artifact") or {}
    lines = [
        "# DEAN-OS Forward Data Accrual Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('gate_status')}`",
        f"- Context: `{artifact.get('context_key')}`",
        f"- Eligible new rows: {summary.get('eligible_new_row_count')}",
        f"- Can supply next development run: {summary.get('can_supply_next_development_run')}",
        f"- Can use as locked test evidence: {summary.get('can_use_as_locked_test_evidence')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Checks",
        "",
    ]
    for check in payload.get("checks", []):
        lines.append(
            f"- `{check.get('check_id')}`: {check.get('status')} — "
            f"{check.get('detail')}"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _inspect_source(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "parquet_magic_valid": False,
        "sha256": None,
        "observed_file_time": None,
        "columns": [],
        "frame": None,
        "load_error": None,
    }
    if not path.is_file():
        return result
    stat = path.stat()
    result["observed_file_time"] = datetime.fromtimestamp(
        max(stat.st_ctime, stat.st_mtime),
        tz=UTC,
    ).isoformat()
    try:
        with path.open("rb") as handle:
            prefix = handle.read(4)
            handle.seek(-4, 2)
            suffix = handle.read(4)
        result["parquet_magic_valid"] = prefix == b"PAR1" and suffix == b"PAR1"
        result["sha256"] = _sha256_file(path)
        if result["parquet_magic_valid"]:
            frame = pd.read_parquet(path)
            result["columns"] = [str(column) for column in frame.columns]
            result["frame"] = frame
    except Exception as exc:
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _checks(
    plan_payload: dict[str, Any] | None,
    source: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    plan_summary = (
        plan_payload.get("summary", {})
        if isinstance(plan_payload, dict)
        else {}
    )
    plan = (
        plan_payload.get("accrual_plan", {})
        if isinstance(plan_payload, dict)
        else {}
    )
    plan = plan if isinstance(plan, dict) else {}
    baseline = plan.get("baseline", {})
    baseline = baseline if isinstance(baseline, dict) else {}
    boundary = plan.get("acceptance_boundary", {})
    boundary = boundary if isinstance(boundary, dict) else {}
    frame = source.get("frame")
    frame = frame.copy() if isinstance(frame, pd.DataFrame) else None

    ticker = str(boundary.get("ticker_must_equal") or "").upper()
    timeframe = str(boundary.get("timeframe_must_equal") or "").lower()
    target_name = str(boundary.get("target_contract_must_equal") or "")
    registered_at = _utc_timestamp(boundary.get("source_artifact_acquired_after"))
    watermark = _utc_timestamp(
        boundary.get("observation_timestamp_strictly_after")
    )
    minimum_rows = _positive_int(
        boundary.get("minimum_new_base_timeframe_rows")
    )
    seen_hashes = {
        str(value)
        for value in baseline.get("seen_development_source_sha256", [])
        if value
    }
    observed_file_time = _utc_timestamp(source.get("observed_file_time"))

    plan_ready = (
        plan_summary.get("plan_status")
        == "forward_development_accrual_plan_ready"
        and plan.get("artifact_class")
        == "pipeline_control_forward_data_accrual_plan"
        and plan.get("lane") == "development_refresh_only"
    )
    boundary_complete = bool(
        ticker
        and timeframe in EXPECTED_MINUTES
        and target_name
        and registered_at is not None
        and watermark is not None
        and minimum_rows is not None
    )
    required_columns_present = bool(
        frame is not None and REQUIRED_COLUMNS.issubset(frame.columns)
    )
    target_like = [
        column
        for column in (source.get("columns") or [])
        if _is_target_like(column)
    ]

    context_frame = pd.DataFrame()
    new_frame = pd.DataFrame()
    parseable = False
    interval_match = False
    if required_columns_present and boundary_complete:
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], errors="coerce", utc=True
        )
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["interval"] = frame["interval"].astype(str).str.lower()
        context_frame = frame.loc[
            frame["ticker"].eq(ticker) & frame["interval"].eq(timeframe)
        ].copy()
        parseable = bool(
            not context_frame.empty and context_frame["datetime"].notna().all()
        )
        interval_match = bool(
            not context_frame.empty
            and set(context_frame["interval"].unique()) == {timeframe}
        )
        if parseable and watermark is not None:
            new_frame = context_frame.loc[
                context_frame["datetime"].gt(watermark)
            ].copy()

    duplicate_rows = (
        int(new_frame.duplicated(["ticker", "interval", "datetime"]).sum())
        if not new_frame.empty
        else 0
    )
    numeric_valid, invalid_ohlcv_rows, max_abs_return = _ohlcv_quality(new_frame)
    cadence_ratio = _cadence_ratio(new_frame, timeframe)
    cadence_ok = bool(
        cadence_ratio is not None and cadence_ratio >= 0.75
    )
    return_limit = RETURN_LIMITS.get(timeframe)
    return_ok = bool(
        max_abs_return is not None
        and return_limit is not None
        and max_abs_return <= return_limit
    )
    acquired_after = bool(
        registered_at is not None
        and observed_file_time is not None
        and observed_file_time > registered_at
    )
    sha_new = bool(
        source.get("sha256")
        and source.get("sha256") not in seen_hashes
    )
    enough_rows = bool(
        minimum_rows is not None and len(new_frame) >= minimum_rows
    )
    cross_ticker_groups = (
        _cross_ticker_identity_groups(frame, watermark, timeframe)
        if required_columns_present and watermark is not None
        else 0
    )

    checks = [
        _check(
            "pass" if plan_ready else "fail",
            "accrual_plan_ready",
            f"plan_status={plan_summary.get('plan_status')}.",
        ),
        _check(
            "pass" if boundary_complete else "fail",
            "acceptance_boundary_complete",
            (
                f"context={ticker}/{timeframe}/{target_name}, "
                f"minimum_rows={minimum_rows}."
            ),
        ),
        _check(
            "pass" if source.get("exists") else "fail",
            "source_artifact_exists",
            f"path={source.get('path')}.",
        ),
        _check(
            "pass" if source.get("parquet_magic_valid") else "fail",
            "source_is_real_parquet",
            f"parquet_magic_valid={source.get('parquet_magic_valid')}.",
        ),
        _check(
            "pass" if source.get("load_error") is None else "fail",
            "source_loadable",
            f"load_error={source.get('load_error')}.",
        ),
        _check(
            "pass" if acquired_after else "fail",
            "source_acquired_after_registration",
            (
                f"observed_file_time={source.get('observed_file_time')}, "
                f"registered_after={_iso(registered_at)}."
            ),
        ),
        _check(
            "pass" if sha_new else "fail",
            "source_sha_is_new",
            (
                f"sha256={source.get('sha256')}, "
                f"seen_hash_count={len(seen_hashes)}."
            ),
        ),
        _check(
            "pass" if required_columns_present else "fail",
            "required_price_columns_present",
            f"missing={sorted(REQUIRED_COLUMNS - set(source.get('columns') or []))}.",
        ),
        _check(
            "pass" if not target_like else "fail",
            "raw_source_has_no_target_columns",
            f"target_like_columns={target_like}.",
        ),
        _check(
            "pass" if parseable and interval_match else "fail",
            "context_rows_parseable",
            (
                f"context_rows={len(context_frame)}, "
                f"timestamps_parseable={parseable}, interval_match={interval_match}."
            ),
        ),
        _check(
            "pass" if enough_rows else "fail",
            "minimum_new_rows_after_watermark",
            (
                f"eligible_rows={len(new_frame)}, minimum_rows={minimum_rows}, "
                f"watermark={_iso(watermark)}."
            ),
        ),
        _check(
            "pass" if duplicate_rows == 0 else "fail",
            "new_row_identity_unique",
            f"duplicate_rows={duplicate_rows}.",
        ),
        _check(
            "pass" if numeric_valid else "fail",
            "new_rows_ohlcv_valid",
            f"invalid_ohlcv_rows={invalid_ohlcv_rows}.",
        ),
        _check(
            "pass" if return_ok else "fail",
            "new_rows_return_limit",
            (
                f"max_abs_return={max_abs_return}, "
                f"limit={return_limit}."
            ),
        ),
        _check(
            "pass" if cadence_ok else "fail",
            "new_rows_timeframe_cadence",
            f"cadence_ratio={cadence_ratio}, minimum=0.75.",
        ),
        _check(
            "pass" if cross_ticker_groups == 0 else "fail",
            "new_rows_no_cross_ticker_ohlcv_copies",
            f"cross_ticker_groups={cross_ticker_groups}.",
        ),
    ]
    context = {
        "ticker": ticker,
        "timeframe": timeframe,
        "target_name": target_name,
        "watermark": _iso(watermark),
        "minimum_rows": minimum_rows,
        "candidate_new_row_count": len(new_frame),
        "eligible_start": (
            new_frame["datetime"].min().isoformat()
            if not new_frame.empty
            else None
        ),
        "eligible_end": (
            new_frame["datetime"].max().isoformat()
            if not new_frame.empty
            else None
        ),
        "cadence_ratio": cadence_ratio,
        "max_abs_return": max_abs_return,
    }
    return checks, context


def _eligible_artifact(
    *,
    plan_payload: dict[str, Any],
    plan_path: Path,
    source: dict[str, Any],
    source_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    plan = plan_payload["accrual_plan"]
    context_key = (
        f"{context['ticker']}/{context['timeframe']}/"
        f"{context['target_name']}"
    )
    return {
        "artifact_class": "pipeline_control_forward_development_artifact",
        "evidence_class": "validated_forward_development_source",
        "lane": "development_refresh_only",
        "context_key": context_key,
        "source_path": str(source_path),
        "source_sha256": source["sha256"],
        "source_observed_file_time": source["observed_file_time"],
        "accrual_plan_json": str(plan_path),
        "accrual_plan_sha256": _sha256_file(plan_path),
        "accrual_plan_id": plan["plan_id"],
        "start_exclusive": context["watermark"],
        "eligible_start": context["eligible_start"],
        "eligible_end": context["eligible_end"],
        "eligible_new_row_count": context["candidate_new_row_count"],
        "cadence_ratio": context["cadence_ratio"],
        "max_abs_return": context["max_abs_return"],
        "target_name": context["target_name"],
        "may_be_used_as_locked_test_evidence": False,
        "may_be_called_virgin_holdout": False,
    }


def _ohlcv_quality(
    frame: pd.DataFrame,
) -> tuple[bool, int, float | None]:
    if frame.empty:
        return False, 0, None
    numeric = frame.copy()
    for column in ("open", "high", "low", "close", "volume"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    finite = np.isfinite(
        numeric[["open", "high", "low", "close", "volume"]].to_numpy(
            dtype=float
        )
    ).all(axis=1)
    valid = (
        pd.Series(finite, index=numeric.index)
        & numeric["low"].le(numeric[["open", "close"]].min(axis=1))
        & numeric["high"].ge(numeric[["open", "close"]].max(axis=1))
        & numeric["volume"].ge(0)
    )
    invalid = int((~valid).sum())
    ordered = numeric.sort_values("datetime")
    returns = ordered["close"].pct_change(fill_method=None).abs().dropna()
    maximum = float(returns.max()) if not returns.empty else 0.0
    return invalid == 0, invalid, maximum if math.isfinite(maximum) else None


def _cadence_ratio(
    frame: pd.DataFrame,
    timeframe: str,
) -> float | None:
    if frame.empty or timeframe not in EXPECTED_MINUTES:
        return None
    ordered = frame.sort_values("datetime")
    deltas = ordered["datetime"].diff().dt.total_seconds().div(60.0).dropna()
    if deltas.empty:
        return 0.0
    expected = EXPECTED_MINUTES[timeframe]
    if timeframe == "1d":
        return float(deltas.between(23 * 60, 3 * 24 * 60).mean())
    return float(deltas.eq(expected).mean())


def _cross_ticker_identity_groups(
    frame: pd.DataFrame,
    watermark: pd.Timestamp,
    timeframe: str,
) -> int:
    candidate = frame.copy()
    candidate["datetime"] = pd.to_datetime(
        candidate["datetime"], errors="coerce", utc=True
    )
    candidate["interval"] = candidate["interval"].astype(str).str.lower()
    candidate["ticker"] = candidate["ticker"].astype(str).str.upper()
    candidate = candidate.loc[
        candidate["datetime"].gt(watermark)
        & candidate["interval"].eq(timeframe)
    ]
    if candidate.empty:
        return 0
    columns = ["datetime", "open", "high", "low", "close", "volume"]
    grouped = candidate.groupby(columns, dropna=False)["ticker"].nunique()
    return int(grouped.gt(1).sum())


def _is_target_like(column: str) -> bool:
    normalized = str(column).strip().lower()
    return (
        normalized.startswith("target")
        or "_target" in normalized
        or normalized.startswith("label")
        or normalized in {"y", "prediction", "predicted_target"}
    )


def _source_preview(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if key != "frame"
    }


def _check(status: str, check_id: str, detail: str) -> dict[str, str]:
    return {"status": status, "check_id": check_id, "detail": detail}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed if not pd.isna(parsed) else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _iso(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "")
    return f"{prefix}_{stamp}"
