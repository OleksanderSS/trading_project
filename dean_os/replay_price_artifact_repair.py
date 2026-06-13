from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.historical_replay import _price_quality_summary, guard_replay_frame
from dean_os.outcome_evaluation import _prepare_market_frame, _read_market_frame
from dean_os.replay_price_normalizer import _normalize_daily_ohlcv, _to_artifact_frame
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class ReplayPriceArtifactRepairPlan:
    """Builds a non-destructive candidate repair for mixed daily replay prices.

    The source cache is never edited. When ``write_artifact`` is enabled, this
    runner writes a new candidate artifact and an audit report that explains
    every quarantine rule used to build it.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_price_artifact_repair",
        artifact_dir: str | Path = "data/dean_os/replay_prices",
    ):
        self.output_dir = Path(output_dir)
        self.artifact_dir = Path(artifact_dir)

    def build(
        self,
        price_data_path: str | Path,
        tickers: list[str] | None = None,
        output_path: str | Path | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        benchmark_ticker: str = "SPY",
        anomaly_threshold: float = 0.30,
        anchor_bridge_threshold: float = 0.15,
        write_artifact: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for replay price artifact repair: {exc}") from exc

        price_path = Path(price_data_path)
        if not price_path.exists():
            raise FileNotFoundError(f"Price data file does not exist: {price_path}")

        raw_frame = _read_market_frame(pd, price_path)
        guard = guard_replay_frame(
            raw_frame,
            required_columns=[datetime_col, close_col, "ticker", "symbol", "Ticker", "Symbol"],
        )
        prepared = _prepare_market_frame(
            pd=pd,
            frame=guard.safe_frame,
            close_col=close_col,
            datetime_col=datetime_col,
        )
        requested_tickers = _normalize_tickers(tickers)
        if requested_tickers and benchmark_ticker.upper() not in requested_tickers:
            requested_tickers = sorted({*requested_tickers, benchmark_ticker.upper()})
        if requested_tickers:
            prepared = prepared[prepared["_dean_ticker"].isin(requested_tickers)].copy()
        else:
            requested_tickers = sorted(
                ticker for ticker in prepared["_dean_ticker"].dropna().unique() if ticker
            )

        repair_result = _build_repaired_candidate(
            pd=pd,
            frame=prepared,
            anomaly_threshold=anomaly_threshold,
            anchor_bridge_threshold=anchor_bridge_threshold,
        )
        artifact_frame = repair_result["artifact_frame"]
        artifact_path = None
        artifact_warnings: list[str] = []
        if write_artifact:
            artifact_path, artifact_warnings = self._write_artifact(pd, artifact_frame, output_path)

        raw_quality = _quality_for_frame(prepared, requested_tickers)
        repaired_quality = _quality_for_frame(repair_result["normalized_internal"], requested_tickers)
        repaired_quality["warnings"] = list(repaired_quality.get("warnings", [])) + artifact_warnings
        learning_gate = _learning_gate(
            repaired_quality=repaired_quality,
            write_artifact=write_artifact,
            quarantined_row_count=len(repair_result["quarantined_rows"]),
        )
        payload = {
            "run_id": _run_id("replay_price_artifact_repair"),
            "created_at": utc_now_iso(),
            "mode": "replay_price_artifact_repair",
            "inputs": {
                "price_data_path": str(price_path),
                "tickers": requested_tickers,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "benchmark_ticker": benchmark_ticker.upper(),
                "anomaly_threshold": anomaly_threshold,
                "anchor_bridge_threshold": anchor_bridge_threshold,
                "write_artifact": write_artifact,
                "output_path": str(output_path) if output_path else None,
            },
            "summary": _summary(
                prepared=prepared,
                artifact_frame=artifact_frame,
                repair_result=repair_result,
                repaired_quality=repaired_quality,
                write_artifact=write_artifact,
            ),
            "artifact": _artifact_summary(artifact_path, artifact_frame),
            "repair_policy": {
                "method": "prefer same-day midnight daily anchor; quarantine non-anchor daily-like rows and unanchored bridge outliers",
                "assumptions": [
                    "Midnight rows in the cached daily file are treated as the canonical daily bar when mixed with intraday-like rows.",
                    "Rows far away from both neighboring anchored daily closes are quarantined instead of interpolated.",
                    "The original source cache is never modified; this creates a candidate replay artifact only.",
                ],
            },
            "quality": {
                "raw": raw_quality,
                "candidate_repaired": repaired_quality,
                "improvement": _quality_improvement(raw_quality, repaired_quality),
            },
            "quarantine": {
                "row_count": len(repair_result["quarantined_rows"]),
                "date_count": len({(item["ticker"], item["date"]) for item in repair_result["quarantined_rows"]}),
                "by_reason": dict(Counter(item["reason"] for item in repair_result["quarantined_rows"]).most_common()),
                "by_ticker": dict(Counter(item["ticker"] for item in repair_result["quarantined_rows"]).most_common()),
                "affected_dates_sample": _affected_dates_sample(repair_result["quarantined_rows"]),
                "rows_sample": repair_result["quarantined_rows"][:50],
            },
            "learning_gate": learning_gate,
            "commands": _commands(artifact_path, benchmark_ticker.upper()),
            "safety": {
                "source_data_mutation_performed": False,
                "candidate_artifact_written": bool(write_artifact and artifact_path),
                "collector_run_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
            "recommendations": _recommendations(learning_gate, write_artifact),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_replay_price_artifact_repair_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path

    def _write_artifact(self, pd: Any, frame: Any, output_path: str | Path | None) -> tuple[Path, list[str]]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_path) if output_path else self.artifact_dir / _default_artifact_name()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = artifact_path.suffix.lower()
        warnings: list[str] = []
        if suffix == ".csv":
            frame.to_csv(artifact_path, index=False)
            return artifact_path, warnings
        if suffix in {".parquet", ".pq"}:
            try:
                frame.to_parquet(artifact_path, index=False)
                return artifact_path, warnings
            except Exception as exc:
                fallback = artifact_path.with_suffix(".csv")
                frame.to_csv(fallback, index=False)
                warnings.append(f"Could not write parquet artifact ({type(exc).__name__}: {exc}); wrote CSV fallback.")
                return fallback, warnings
        raise ValueError(f"Unsupported artifact file type: {artifact_path.suffix}. Use .csv or .parquet.")


def render_replay_price_artifact_repair_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    artifact = payload.get("artifact", {})
    gate = payload.get("learning_gate", {})
    quarantine = payload.get("quarantine", {})
    lines = [
        "# DEAN-OS Replay Price Artifact Repair",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Repair status: `{summary.get('repair_status')}`",
        f"- Artifact written: {summary.get('artifact_written')}",
        f"- Artifact: `{artifact.get('path')}`",
        f"- Input rows: {summary.get('input_rows')}",
        f"- Candidate rows: {summary.get('candidate_rows')}",
        f"- Quarantined rows: {quarantine.get('row_count')}",
        f"- Affected ticker/date pairs: {quarantine.get('date_count')}",
        f"- Learning gate: `{gate.get('status')}`",
        "",
        "## Quarantine Reasons",
        "",
    ]
    for reason, count in quarantine.get("by_reason", {}).items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Quality", ""])
    quality = payload.get("quality", {})
    raw = quality.get("raw", {})
    repaired = quality.get("candidate_repaired", {})
    lines.extend(
        [
            f"- Raw warnings: {len(raw.get('warnings', []))}",
            f"- Candidate warnings: {len(repaired.get('warnings', []))}",
            f"- Raw max rows per ticker/day: {raw.get('max_rows_per_ticker_day')}",
            f"- Candidate max rows per ticker/day: {repaired.get('max_rows_per_ticker_day')}",
            f"- Candidate SPY lookback return: {repaired.get('benchmark_spy_lookback_return')}",
            "",
            "## Recommendations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _build_repaired_candidate(
    pd: Any,
    frame: Any,
    anomaly_threshold: float,
    anchor_bridge_threshold: float,
) -> dict[str, Any]:
    if frame.empty:
        empty = _normalize_daily_ohlcv(pd, frame)
        artifact = _to_artifact_frame(empty)
        return {"normalized_internal": empty, "artifact_frame": artifact, "quarantined_rows": []}

    working = frame.copy().reset_index(drop=True)
    working["_dean_repair_row_id"] = range(len(working))
    working["_dean_repair_date"] = working["_dean_datetime"].dt.date
    quarantined: dict[int, dict[str, Any]] = {}
    group_meta: list[dict[str, Any]] = []

    for (ticker, date_value), group in working.groupby(["_dean_ticker", "_dean_repair_date"], sort=True):
        group = group.sort_values("_dean_datetime")
        anchor_mask = _midnight_mask(group["_dean_datetime"])
        anchor_rows = group[anchor_mask]
        row_ids = [int(value) for value in group["_dean_repair_row_id"].tolist()]
        if not anchor_rows.empty:
            anchor = anchor_rows.iloc[-1]
            anchor_row_id = int(anchor["_dean_repair_row_id"])
            anchor_close = float(anchor["_dean_close"])
            group_meta.append(
                {
                    "ticker": str(ticker),
                    "date": date_value,
                    "has_anchor": True,
                    "selected_close": anchor_close,
                    "row_ids": row_ids,
                }
            )
            for _, row in group.iterrows():
                row_id = int(row["_dean_repair_row_id"])
                if row_id == anchor_row_id:
                    continue
                deviation = _relative_deviation(float(row["_dean_close"]), anchor_close)
                reason = "same_day_anchor_deviation" if deviation > anomaly_threshold else "daily_anchor_preferred"
                quarantined[row_id] = _quarantine_record(
                    row=row,
                    reason=reason,
                    anchor_close=anchor_close,
                    deviation=deviation,
                )
        else:
            group_meta.append(
                {
                    "ticker": str(ticker),
                    "date": date_value,
                    "has_anchor": False,
                    "selected_close": float(group["_dean_close"].iloc[-1]),
                    "row_ids": row_ids,
                }
            )

    _quarantine_unanchored_bridge_outliers(
        working=working,
        group_meta=group_meta,
        quarantined=quarantined,
        anomaly_threshold=anomaly_threshold,
        anchor_bridge_threshold=anchor_bridge_threshold,
    )

    quarantine_ids = set(quarantined)
    kept = working[~working["_dean_repair_row_id"].isin(quarantine_ids)].copy()
    kept = kept.drop(columns=["_dean_repair_row_id", "_dean_repair_date"], errors="ignore")
    normalized = _normalize_daily_ohlcv(pd, kept)
    if "interval" in normalized.columns:
        normalized["interval"] = "1d_repaired"
    artifact = _to_artifact_frame(normalized)
    if not artifact.empty:
        artifact["repair_status"] = "candidate_repaired"
    return {
        "normalized_internal": normalized,
        "artifact_frame": artifact,
        "quarantined_rows": sorted(quarantined.values(), key=lambda item: (item["ticker"], item["date"], item["datetime"])),
    }


def _quarantine_unanchored_bridge_outliers(
    working: Any,
    group_meta: list[dict[str, Any]],
    quarantined: dict[int, dict[str, Any]],
    anomaly_threshold: float,
    anchor_bridge_threshold: float,
) -> None:
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for meta in group_meta:
        by_ticker.setdefault(str(meta["ticker"]), []).append(meta)

    for ticker, items in by_ticker.items():
        ordered = sorted(items, key=lambda item: item["date"])
        anchor_positions = [index for index, item in enumerate(ordered) if item["has_anchor"]]
        if len(anchor_positions) < 2:
            continue
        for index, item in enumerate(ordered):
            if item["has_anchor"]:
                continue
            previous_anchor = _previous_anchor(ordered, anchor_positions, index)
            next_anchor = _next_anchor(ordered, anchor_positions, index)
            if previous_anchor is None or next_anchor is None:
                continue
            bridge_deviation = _relative_deviation(
                float(next_anchor["selected_close"]),
                float(previous_anchor["selected_close"]),
            )
            previous_deviation = _relative_deviation(float(item["selected_close"]), float(previous_anchor["selected_close"]))
            next_deviation = _relative_deviation(float(item["selected_close"]), float(next_anchor["selected_close"]))
            if (
                bridge_deviation <= anchor_bridge_threshold
                and previous_deviation > anomaly_threshold
                and next_deviation > anomaly_threshold
            ):
                rows = working[working["_dean_repair_row_id"].isin(item["row_ids"])]
                for _, row in rows.iterrows():
                    row_id = int(row["_dean_repair_row_id"])
                    quarantined[row_id] = _quarantine_record(
                        row=row,
                        reason="unanchored_price_level_outlier",
                        anchor_close=None,
                        deviation=max(previous_deviation, next_deviation),
                        previous_anchor_close=float(previous_anchor["selected_close"]),
                        next_anchor_close=float(next_anchor["selected_close"]),
                    )


def _previous_anchor(
    ordered: list[dict[str, Any]],
    anchor_positions: list[int],
    index: int,
) -> dict[str, Any] | None:
    candidates = [position for position in anchor_positions if position < index]
    return ordered[candidates[-1]] if candidates else None


def _next_anchor(
    ordered: list[dict[str, Any]],
    anchor_positions: list[int],
    index: int,
) -> dict[str, Any] | None:
    candidates = [position for position in anchor_positions if position > index]
    return ordered[candidates[0]] if candidates else None


def _midnight_mask(series: Any) -> Any:
    return (series.dt.hour == 0) & (series.dt.minute == 0) & (series.dt.second == 0)


def _quarantine_record(
    row: Any,
    reason: str,
    anchor_close: float | None,
    deviation: float,
    previous_anchor_close: float | None = None,
    next_anchor_close: float | None = None,
) -> dict[str, Any]:
    return {
        "ticker": str(row["_dean_ticker"]),
        "date": str(row["_dean_repair_date"]),
        "datetime": row["_dean_datetime"].isoformat(),
        "close": float(row["_dean_close"]),
        "reason": reason,
        "deviation": round(float(deviation), 6),
        "anchor_close": anchor_close,
        "previous_anchor_close": previous_anchor_close,
        "next_anchor_close": next_anchor_close,
    }


def _summary(
    prepared: Any,
    artifact_frame: Any,
    repair_result: dict[str, Any],
    repaired_quality: dict[str, Any],
    write_artifact: bool,
) -> dict[str, Any]:
    quarantined = repair_result["quarantined_rows"]
    warnings = repaired_quality.get("warnings", [])
    if warnings:
        status = "blocked_candidate_has_warnings"
    elif write_artifact:
        status = "candidate_artifact_written"
    elif quarantined:
        status = "dry_run_candidate_ready"
    else:
        status = "dry_run_no_repair_needed"
    return {
        "repair_status": status,
        "artifact_written": bool(write_artifact),
        "input_rows": int(len(prepared)),
        "candidate_rows": int(len(artifact_frame)),
        "quarantined_rows": int(len(quarantined)),
        "quarantined_ticker_count": len({item["ticker"] for item in quarantined}),
        "quarantined_ticker_date_count": len({(item["ticker"], item["date"]) for item in quarantined}),
        "candidate_quality_warning_count": len(warnings),
    }


def _artifact_summary(path: Path | None, frame: Any) -> dict[str, Any]:
    return {
        "path": str(path) if path else None,
        "format": path.suffix.lower().lstrip(".") if path else None,
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if not frame.empty and "ticker" in frame.columns else 0,
        "start": frame["datetime"].min().isoformat() if not frame.empty and "datetime" in frame.columns else None,
        "end": frame["datetime"].max().isoformat() if not frame.empty and "datetime" in frame.columns else None,
        "columns": list(frame.columns),
    }


def _quality_for_frame(frame: Any, tickers: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {
            "warnings": ["Price artifact has no rows."],
            "duplicate_ticker_datetime_count": 0,
            "max_rows_per_ticker_day": 0,
            "multi_row_ticker_day_count": 0,
            "interval_counts": {},
            "benchmark_spy_lookback_return": None,
        }
    return _price_quality_summary(frame, tickers)


def _quality_improvement(raw_quality: dict[str, Any], repaired_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "duplicate_ticker_datetime_delta": int(repaired_quality.get("duplicate_ticker_datetime_count", 0))
        - int(raw_quality.get("duplicate_ticker_datetime_count", 0)),
        "max_rows_per_ticker_day_delta": int(repaired_quality.get("max_rows_per_ticker_day", 0))
        - int(raw_quality.get("max_rows_per_ticker_day", 0)),
        "multi_row_ticker_day_delta": int(repaired_quality.get("multi_row_ticker_day_count", 0))
        - int(raw_quality.get("multi_row_ticker_day_count", 0)),
        "warning_count_delta": len(repaired_quality.get("warnings", [])) - len(raw_quality.get("warnings", [])),
    }


def _learning_gate(
    repaired_quality: dict[str, Any],
    write_artifact: bool,
    quarantined_row_count: int,
) -> dict[str, Any]:
    warnings = repaired_quality.get("warnings", [])
    if warnings:
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "Candidate repaired artifact still has price-quality warnings.",
            "warnings": warnings,
        }
    if not write_artifact:
        return {
            "status": "dry_run_review_required",
            "can_write_learning_memory": False,
            "reason": "Repair candidate is clean in memory, but no artifact was written.",
            "warnings": [],
        }
    return {
        "status": "candidate_ready_for_replay_review",
        "can_write_learning_memory": False,
        "reason": (
            "Candidate artifact passed local price-quality checks. Use it for replay diagnostics only, "
            f"then review outcomes before any learning promotion. Quarantined rows: {quarantined_row_count}."
        ),
        "warnings": [],
    }


def _recommendations(gate: dict[str, Any], write_artifact: bool) -> list[str]:
    recommendations = [
        "Keep the original cached price file unchanged; use only the candidate repaired artifact for replay diagnostics.",
        "Rerun replay price-quality investigation against the candidate artifact before trusting hit/miss results.",
    ]
    if gate.get("status") == "blocked":
        recommendations.append("Do not run learning or calibration from this artifact; price-quality warnings remain.")
    elif write_artifact:
        recommendations.append("Next: rerun historical replay batch and historical research replay batch using the repaired artifact path.")
    else:
        recommendations.append("If the dry-run report looks correct, rerun with --write-artifact to create the candidate artifact.")
    return recommendations


def _commands(artifact_path: Path | None, benchmark_ticker: str) -> dict[str, str]:
    path = str(artifact_path) if artifact_path else "PATH_TO_REPAIRED_ARTIFACT"
    return {
        "inspect_repaired_artifact": (
            "python run_agent_replay_price_quality_investigation.py "
            f"--artifact-only --price-data {path} --benchmark-ticker {benchmark_ticker}"
        ),
        "replay_batch_on_repaired_artifact": (
            "python run_agent_historical_replay_batch.py "
            f"{path} --tickers AMD NVDA MSFT AAPL TSM QQQ {benchmark_ticker} "
            "--start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 "
            "--step-days 30 --lookback-days 180 --horizon-days 30 60"
        ),
    }


def _affected_dates_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter((item["ticker"], item["date"], item["reason"]) for item in rows)
    return [
        {"ticker": ticker, "date": date, "reason": reason, "row_count": count}
        for (ticker, date, reason), count in counter.most_common(25)
    ]


def _relative_deviation(value: float, reference: float) -> float:
    if not reference:
        return 0.0
    return abs(value / reference - 1.0)


def _normalize_tickers(tickers: Iterable[str] | None) -> list[str]:
    return sorted({str(ticker).strip().upper() for ticker in tickers or [] if str(ticker).strip()})


def _default_artifact_name() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"replay_prices_1d_repaired_{stamp}.parquet"


def _run_id(prefix: str) -> str:
    return prefix + "_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
