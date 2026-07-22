from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

EXPECTED_MINUTES = {"15m": 15.0, "60m": 60.0, "1d": 1440.0}
RUNNER_TARGETS = {
    "15m": "target_intraday_up_15m",
    "1d": "target_up_1d",
}


class PipelineControlSavedDataCoverage:
    """Inventory saved price and macro coverage before bounded evidence runs."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_saved_data_coverage_current",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        assets_yaml: str | Path = "src/config/assets.yaml",
        price_paths: list[str | Path] | None = None,
        macro_paths: list[str | Path] | None = None,
        min_rows: int = 180,
        max_rows: int = 600,
        max_abs_return: float = 0.25,
        min_cadence_ratio: float = 0.75,
        save: bool = True,
    ) -> dict[str, Any]:
        assets_path = Path(assets_yaml)
        configured_assets, sector_by_asset = _load_assets(assets_path)
        resolved_price_paths = (
            [Path(path) for path in price_paths]
            if price_paths is not None
            else _latest_price_snapshots(Path("data/processed"))
        )
        resolved_macro_paths = (
            [Path(path) for path in macro_paths]
            if macro_paths is not None
            else _default_macro_candidates(Path("data/processed"))
        )
        macro_sources = [_analyze_macro_source(path) for path in resolved_macro_paths]
        recommended_macro = _recommended_macro_source(macro_sources)

        price_sources: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        for path in resolved_price_paths:
            source, source_contexts = _analyze_price_source(
                path,
                configured_assets=configured_assets,
                sector_by_asset=sector_by_asset,
                min_rows=min_rows,
                max_rows=max_rows,
                max_abs_return=max_abs_return,
                min_cadence_ratio=min_cadence_ratio,
                macro_source=recommended_macro,
            )
            price_sources.append(source)
            contexts.extend(source_contexts)

        eligible = [
            context
            for context in contexts
            if context.get("evidence_eligible")
        ]
        configured_with_data = sorted(
            {
                context["ticker"]
                for context in contexts
                if context.get("row_count", 0) > 0
            }
        )
        summary = {
            "coverage_status": (
                "saved_data_coverage_ready"
                if eligible
                else "saved_data_coverage_blocked"
            ),
            "configured_asset_count": len(configured_assets),
            "configured_assets_with_price_data": len(configured_with_data),
            "configured_assets_missing_price_data": sorted(
                set(configured_assets) - set(configured_with_data)
            ),
            "price_source_count": len(price_sources),
            "context_count": len(contexts),
            "eligible_context_count": len(eligible),
            "eligible_15m_context_count": sum(
                1 for context in eligible if context.get("timeframe") == "15m"
            ),
            "macro_source_count": len(macro_sources),
            "recommended_macro_source": (
                recommended_macro.get("path") if recommended_macro else None
            ),
            "latest_processed_macro_snapshot_empty": any(
                source.get("is_latest_processed_snapshot") and source.get("row_count") == 0
                for source in macro_sources
            ),
            "can_train": False,
            "can_tune": False,
            "can_trade": False,
        }
        payload = {
            "run_id": _run_id("pipeline_control_saved_data_coverage"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_saved_data_coverage",
            "inputs": {
                "assets_yaml": str(assets_path),
                "price_paths": [str(path) for path in resolved_price_paths],
                "macro_paths": [str(path) for path in resolved_macro_paths],
                "min_rows": min_rows,
                "max_rows": max_rows,
                "max_abs_return": max_abs_return,
                "min_cadence_ratio": min_cadence_ratio,
            },
            "summary": summary,
            "configured_assets": configured_assets,
            "price_sources": price_sources,
            "macro_sources": macro_sources,
            "recommended_macro_source": recommended_macro,
            "contexts": contexts,
            "eligible_contexts": eligible,
            "known_contract_blocks": [
                {
                    "timeframe": "60m",
                    "code": "hourly_target_shift_mismatch",
                    "detail": (
                        "The current target_hourly_up_1h uses shift=-4, which assumes 15m bars; "
                        "it is not valid for a true 60m source without a target-contract change."
                    ),
                }
            ],
            "explicit_non_actions": [
                "No collector or external API was called.",
                "No model was trained or evaluated.",
                "No test window was consumed.",
                "No recommendation, order, paper trade, or live trade was created.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_saved_data_coverage_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_saved_data_coverage_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Saved Data Coverage",
        "",
        f"- Status: `{summary.get('coverage_status')}`",
        f"- Configured assets: {summary.get('configured_asset_count')}",
        f"- Assets with price data: {summary.get('configured_assets_with_price_data')}",
        f"- Eligible contexts: {summary.get('eligible_context_count')}",
        f"- Eligible 15m contexts: {summary.get('eligible_15m_context_count')}",
        f"- Recommended macro source: `{summary.get('recommended_macro_source')}`",
        f"- Latest processed macro snapshot empty: {summary.get('latest_processed_macro_snapshot_empty')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Contexts",
        "",
    ]
    for context in payload.get("contexts", []):
        lines.append(
            f"- `{context.get('ticker')}/{context.get('timeframe')}`: "
            f"rows={context.get('row_count')} clean_suffix={context.get('clean_suffix_row_count')} "
            f"cadence={context.get('clean_suffix_cadence_ratio')} "
            f"max_return={context.get('clean_suffix_max_abs_return')} "
            f"status={context.get('status')}"
        )
    lines.extend(["", "## Macro Sources", ""])
    for source in payload.get("macro_sources", []):
        lines.append(
            f"- `{source.get('path')}`: rows={source.get('row_count')} "
            f"series={source.get('series_count')} status={source.get('status')}"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _load_assets(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assets = payload.get("assets", {})
    active_preset = assets.get("active_preset")
    presets = assets.get("presets", {})
    configured = [
        str(ticker).upper()
        for ticker in presets.get(active_preset, {}).get("tickers", [])
        if str(ticker).strip()
    ]
    sector_by_asset: dict[str, list[str]] = {ticker: [] for ticker in configured}
    for sector, item in assets.get("sectors", {}).items():
        for ticker in item.get("assets", []):
            sector_by_asset.setdefault(str(ticker).upper(), []).append(str(sector))
    return list(dict.fromkeys(configured)), sector_by_asset


def _latest_price_snapshots(directory: Path) -> list[Path]:
    paths = []
    for timeframe in EXPECTED_MINUTES:
        candidates = list(directory.glob(f"prices_{timeframe}_*.parquet"))
        if candidates:
            paths.append(max(candidates, key=lambda path: path.stat().st_mtime))
    return paths


def _default_macro_candidates(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    processed_snapshots = list(directory.glob("macro_data_*.parquet"))
    if processed_snapshots:
        candidates.append(max(processed_snapshots, key=lambda path: path.stat().st_mtime))
    enriched_macro = directory / "features" / "macro_data.parquet"
    if enriched_macro.exists():
        candidates.append(enriched_macro)
    return list(dict.fromkeys(candidates))


def _analyze_price_source(
    path: Path,
    *,
    configured_assets: list[str],
    sector_by_asset: dict[str, list[str]],
    min_rows: int,
    max_rows: int,
    max_abs_return: float,
    min_cadence_ratio: float,
    macro_source: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timeframe = _timeframe_from_path(path)
    source = {
        "path": str(path),
        "timeframe": timeframe,
        "exists": path.exists(),
        "row_count": 0,
        "status": "missing",
    }
    if not path.exists():
        return source, []
    frame = _load_table(path)
    source["row_count"] = len(frame)
    required = {"datetime", "ticker", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        source.update({"status": "blocked_schema", "missing_columns": missing})
        return source, []

    frame = frame.copy()
    frame["_coverage_datetime"] = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
        utc=True,
    )
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    contexts = []
    for ticker in configured_assets:
        group = frame.loc[frame["ticker"].eq(ticker)].copy()
        contexts.append(
            _analyze_context(
                group,
                path=path,
                ticker=ticker,
                timeframe=timeframe,
                sectors=sector_by_asset.get(ticker, []),
                min_rows=min_rows,
                max_rows=max_rows,
                max_abs_return=max_abs_return,
                min_cadence_ratio=min_cadence_ratio,
                macro_source=macro_source,
            )
        )
    source.update(
        {
            "status": "analyzed",
            "available_tickers": sorted(frame["ticker"].dropna().unique().tolist()),
            "configured_ticker_count": len(configured_assets),
            "eligible_context_count": sum(
                1 for context in contexts if context.get("evidence_eligible")
            ),
        }
    )
    return source, contexts


def _analyze_context(
    frame: pd.DataFrame,
    *,
    path: Path,
    ticker: str,
    timeframe: str | None,
    sectors: list[str],
    min_rows: int,
    max_rows: int,
    max_abs_return: float,
    min_cadence_ratio: float,
    macro_source: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "source_path": str(path),
        "ticker": ticker,
        "timeframe": timeframe,
        "sectors": sectors,
        "row_count": len(frame),
        "evidence_eligible": False,
    }
    if frame.empty or timeframe not in EXPECTED_MINUTES:
        return {**base, "status": "blocked_missing_or_unknown_timeframe"}

    frame = frame.dropna(subset=["_coverage_datetime"]).sort_values(
        "_coverage_datetime"
    ).drop_duplicates("_coverage_datetime", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = close.pct_change(fill_method=None).abs()
    bad_positions = np.flatnonzero((returns > max_abs_return).fillna(False).to_numpy())
    clean_start_position = int(bad_positions[-1] + 1) if len(bad_positions) else 0
    clean = frame.iloc[clean_start_position:].copy()
    clean_close = pd.to_numeric(clean["close"], errors="coerce")
    clean_returns = clean_close.pct_change(fill_method=None).abs().dropna()
    cadence = clean["_coverage_datetime"].diff().dt.total_seconds().div(60.0).dropna()
    expected_minutes = EXPECTED_MINUTES[timeframe]
    cadence_ratio = float((cadence == expected_minutes).mean()) if not cadence.empty else 0.0
    finite_rows = int(np.isfinite(clean_close).sum())
    clean_max_return = float(clean_returns.max()) if not clean_returns.empty else math.inf
    target_name = RUNNER_TARGETS.get(timeframe)
    target_contract_ready = target_name is not None
    effective_start = clean["_coverage_datetime"].min() if not clean.empty else None
    if macro_source and macro_source.get("captured_at") and effective_start is not None:
        macro_capture = pd.to_datetime(macro_source["captured_at"], utc=True)
        effective_start = max(effective_start, macro_capture)
    rows_after_start = (
        int((clean["_coverage_datetime"] >= effective_start).sum())
        if effective_start is not None
        else 0
    )
    eligible = bool(
        target_contract_ready
        and rows_after_start >= min_rows
        and finite_rows == len(clean)
        and math.isfinite(clean_max_return)
        and clean_max_return <= max_abs_return
        and cadence_ratio >= min_cadence_ratio
    )
    blockers = []
    if rows_after_start < min_rows:
        blockers.append("insufficient_clean_rows")
    if finite_rows != len(clean):
        blockers.append("non_finite_close")
    if not math.isfinite(clean_max_return) or clean_max_return > max_abs_return:
        blockers.append("extreme_return")
    if cadence_ratio < min_cadence_ratio:
        blockers.append("timeframe_cadence_mismatch")
    if not target_contract_ready:
        blockers.append("target_contract_not_ready")
    return {
        **base,
        "row_count": len(frame),
        "start": _iso(frame["_coverage_datetime"].min()),
        "end": _iso(frame["_coverage_datetime"].max()),
        "clean_suffix_row_count": len(clean),
        "clean_suffix_start": _iso(clean["_coverage_datetime"].min()) if not clean.empty else None,
        "clean_suffix_end": _iso(clean["_coverage_datetime"].max()) if not clean.empty else None,
        "clean_suffix_max_abs_return": clean_max_return,
        "clean_suffix_cadence_ratio": cadence_ratio,
        "expected_cadence_minutes": expected_minutes,
        "effective_start": _iso(effective_start),
        "rows_after_effective_start": rows_after_start,
        "target_name": target_name,
        "target_contract_ready": target_contract_ready,
        "max_rows": min(max_rows, rows_after_start),
        "status": "eligible" if eligible else "blocked",
        "blockers": blockers,
        "evidence_eligible": eligible,
    }


def _analyze_macro_source(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": 0,
        "series_count": 0,
        "status": "missing",
        "is_latest_processed_snapshot": path.parent.name == "processed"
        and path.name.startswith("macro_data_"),
    }
    if not path.exists():
        return result
    frame = _load_table(path)
    result["row_count"] = len(frame)
    result["captured_at"] = pd.Timestamp(
        path.stat().st_mtime,
        unit="s",
        tz="UTC",
    ).isoformat()
    if frame.empty:
        result["status"] = "blocked_empty"
        return result
    date_column = next(
        (column for column in ("datetime", "date", "timestamp") if column in frame.columns),
        None,
    )
    series_column = next(
        (column for column in ("series_id", "series") if column in frame.columns),
        None,
    )
    if not date_column or not series_column or "value" not in frame.columns:
        result.update(
            {
                "status": "blocked_schema",
                "columns": [str(column) for column in frame.columns],
            }
        )
        return result
    timestamps = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    values = pd.to_numeric(frame["value"], errors="coerce")
    result.update(
        {
            "status": "usable_saved_macro"
            if timestamps.notna().any() and np.isfinite(values).any()
            else "blocked_invalid_values",
            "date_column": date_column,
            "series_column": series_column,
            "series_count": int(frame[series_column].astype(str).nunique()),
            "observation_start": _iso(timestamps.min()),
            "observation_end": _iso(timestamps.max()),
            "finite_value_count": int(np.isfinite(values).sum()),
            "availability_basis": (
                "realtime_start"
                if "realtime_start" in frame.columns
                else "artifact_mtime_conservative"
            ),
        }
    )
    return result


def _recommended_macro_source(sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [source for source in sources if source.get("status") == "usable_saved_macro"]
    if not usable:
        return None
    return max(
        usable,
        key=lambda source: (
            pd.to_datetime(source.get("observation_end"), errors="coerce", utc=True),
            int(source.get("series_count", 0)),
        ),
    )


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Coverage source must be parquet or CSV: {path}")


def _timeframe_from_path(path: Path) -> str | None:
    lowered = path.name.lower()
    for timeframe in EXPECTED_MINUTES:
        if f"_{timeframe}_" in lowered or lowered.startswith(f"prices_{timeframe}"):
            return timeframe
    return None


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
